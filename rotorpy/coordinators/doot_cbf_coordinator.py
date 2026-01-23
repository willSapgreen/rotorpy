import numpy as np
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Dict, Any, Tuple
import math
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator


@dataclass(frozen=True)
class DootConfig:
    """
    Static Distributed Online Optimal Transport (DOOT) configuration (no runtime state).
    """
    num_neighbors: int                 # kNN neighbors excluding self
    max_iter_primaldual: int           # inner primal-dual iterations per outer step

    use_random_sampling: bool = False  # Using _FIXED_SEED to have deterministic sampling result
    num_trial_move_samples: int = 300
    mean_trial_move: Optional[List[float]] = None   # len-3
    var_trial_move: Optional[List[float]] = None    # len-3 (per-axis variance)
    min_displacement_norm: float = 0.1

    # Option A: axis-aligned planar detection.
    # If any axis has std <= this threshold across agents, treat swarm as planar
    # and interpolate on the best 2 axes (largest std).
    planar_std_threshold: float = 0.1

    # TODO (Option B): replace axis-aligned detection with SVD/rank-based
    # subspace detection to support rotated planes robustly.


@dataclass(frozen=True)
class CbfConfig:
    """
    Configuration for density-based Control Barrier Function (CBF).
    This config is purely static (no runtime state).
    """

    # === Density time-derivative gains ===
    # Enforces upper density bound: rho <= density_upper_bound
    density_upper_gain: float

    # Enforces lower density bound: rho >= density_lower_bound
    density_lower_gain: float

    # === Density thresholds ===
    # Maximum allowed local density (crowding avoidance)
    density_upper_bound: float

    # Minimum allowed local density (anti-fragmentation)
    density_lower_bound: float

    # === Kernel Density Estimation (KDE) parameters ===
    # Gaussian kernel bandwidth
    kde_bandwidth: float

    # Threshold for ||∇rho|| below which CBF projection is skipped
    grad_norm_eps: float = 1e-6

    # === Derived quantities (computed once) ===
    # Effective interaction radii induced by KDE truncation
    kde_radius: float = field(init=False)
    kde_radius_bar: float = field(init=False)

    # Radius actually used for neighbor selection
    interaction_radius: float = field(init=False)

    def __post_init__(self):
        bw = float(self.kde_bandwidth)

        if bw <= 0.0:
            raise ValueError("kde_bandwidth must be positive.")

        # Formulas:
        #   Rh     = sqrt(-2*bw^2 * log(2*pi*bw^3))
        #   Rh_bar = sqrt(-2*bw^2 * log(2*pi*bw^5))
        try:
            Rh = math.sqrt(-2.0 * bw * bw * math.log(2.0 * math.pi * bw**3))
            Rh_bar = math.sqrt(-2.0 * bw * bw * math.log(2.0 * math.pi * bw**5))
        except ValueError as e:
            raise ValueError(
                "Invalid KDE radius computation; check kde_bandwidth."
            ) from e

        object.__setattr__(self, "kde_radius", Rh)
        object.__setattr__(self, "kde_radius_bar", Rh_bar)
        object.__setattr__(self, "interaction_radius", Rh_bar)

class DootCbfCoordinator:
    """
    Coordinator that stores per-vehicle commanded velocities and exposes
    per-vehicle callables transport_vel_fn(cur_time) -> radius^3.

    This class hosts DOOT (and CBF later) in `step(...)`.

    Assumptions:
      - The simulator calls coordinator.step(t_global, states_subset),
        where states_subset is ONLY the states of vehicles assigned to this coordinator.
      - All vehicles share the same time base.
    """

    # Fixed seed for reproducible sampling when use_random_sampling=True
    _FIXED_SEED: int = 20220610

    def __init__(
        self,
        *,
        vehicles: List[Any],
        velocity_max: Optional[float] = None,
        targeted_positions: List,
        doot_config: DootConfig,
        apply_cbf: bool = False,
        cbf_config: Optional[CbfConfig] = None,
    ):
        # Membership / core
        self.vehicles = vehicles
        self.num_vehicles = len(self.vehicles)
        if self.num_vehicles < 1:
            raise ValueError("vehicles must be a non-empty list.")
        self.velocity_max = velocity_max  # scalar cap (m/s) for all vehicles in this coordinator

        # Target samples (num_targeted_pos,3)
        self.targeted_positions = np.asarray(targeted_positions, dtype=float)
        if self.targeted_positions.ndim != 2 or self.targeted_positions.shape[1] != 3:
            raise ValueError(
                f"targeted_positions must have shape (num_targeted_pos,3), got {self.targeted_positions.shape}"
            )

        # DOOT config
        self.doot_config = doot_config
        if self.doot_config.num_neighbors < 1:
            raise ValueError("doot_config.num_neighbors must be >= 1 (excluding self).")
        if self.doot_config.max_iter_primaldual < 1:
            raise ValueError("doot_config.max_iter_primaldual must be >= 1.")
        if self.doot_config.use_random_sampling:
            if self.doot_config.num_trial_move_samples < 1:
                raise ValueError("doot_config.num_trial_move_samples must be >= 1.")
            if self.doot_config.min_displacement_norm < 0.0:
                raise ValueError("doot_config.min_displacement_norm must be >= 0.")

        # Trial-move distribution defaults (3D)
        if self.doot_config.mean_trial_move is None:
            mean_trial_move = [0.0, 0.0, 0.0]
        else:
            mean_trial_move = self.doot_config.mean_trial_move

        if self.doot_config.var_trial_move is None:
            var_trial_move = [0.05, 0.05, 0.05]
        else:
            var_trial_move = self.doot_config.var_trial_move

        self.mean_trial_move = np.asarray(mean_trial_move, dtype=float).reshape(3,)
        self.var_trial_move = np.asarray(var_trial_move, dtype=float).reshape(3,)
        if np.any(self.var_trial_move < 0.0):
            raise ValueError("var_trial_move must be nonnegative per axis.")

        # DOOT runtime state
        self.phi = np.zeros((self.num_vehicles,), dtype=float)  # dual potential at agent locations
        self.last_time: Optional[float] = None

        # RNG
        # - use_random_sampling=True  -> non-deterministic seed (system entropy)
        # - use_random_sampling=False -> deterministic seed (_FIXED_SEED)
        seed = None if self.doot_config.use_random_sampling else self._FIXED_SEED
        self.rng: np.random.Generator = np.random.default_rng(seed)


        # Command storage
        self._transport_vel = np.zeros((self.num_vehicles, 3), dtype=float)
        self._transport_vel_fns = [self._make_transport_vel_fn(i) for i in range(self.num_vehicles)]

        # CBF config
        self.apply_cbf = bool(apply_cbf)
        self.cbf_config = cbf_config
        if self.apply_cbf and self.cbf_config is None:
            raise ValueError("apply_cbf=True requires a valid CbfConfig.")

        # CBF runtime state (previous positions only)
        self._positions_prev_cbf: Optional[np.ndarray] = None

    def _make_transport_vel_fn(self, vehicle_idx: int) -> Callable[[float], np.ndarray]:

        # transport_vel_fn(cur_time) returns the latest _transport_vel[vehicle_idx]
        def transport_vel_fn(cur_time: float) -> np.ndarray:
            return self._transport_vel[vehicle_idx]
        return transport_vel_fn

    def get_transport_vel_fns(self) -> List[Callable[[float], np.ndarray]]:
        return self._transport_vel_fns

    def set_transport_vel_batch(self, vel_batch: np.ndarray) -> None:
        vel_batch = np.asarray(vel_batch, dtype=float)
        if vel_batch.shape != (self.num_vehicles, 3):
            raise ValueError(f"vel_batch must have shape {(self.num_vehicles, 3)}, got {vel_batch.shape}")

        if self.velocity_max is not None:
            speeds = np.linalg.norm(vel_batch, axis=1)
            mask = speeds > self.velocity_max
            if np.any(mask):
                vel_batch[mask] *= (self.velocity_max / speeds[mask])[:, None]

        self._transport_vel[:, :] = vel_batch

    def _compute_kde_density_and_gradient(
        self,
        positions_prev: np.ndarray,
        vehicle_idx: int,
        *,
        planar: bool,
        axes_2d: Optional[Tuple[int, int]],
    ) -> Tuple[float, np.ndarray]:
        """
        KDE rho and grad_rho at agent vehicle_idx using previous positions, consistent with DOOT plane.

        If planar=True:
        - compute KDE in the DOOT plane defined by axes_2d (two indices in {0,1,2})
        - return a (3,) gradient with nonzeros only on axes_2d

        If planar=False:
        - compute KDE in 3D and return full (3,) gradient
        """
        if self.cbf_config is None:
            raise RuntimeError("_compute_kde_density_and_gradient called but cbf_config is None.")

        positions_prev = np.asarray(positions_prev, dtype=float)
        if positions_prev.shape != (self.num_vehicles, 3):
            raise ValueError(f"positions_prev must have shape {(self.num_vehicles, 3)}, got {positions_prev.shape}")
        if not (0 <= vehicle_idx < self.num_vehicles):
            raise IndexError(f"vehicle_idx out of range: {vehicle_idx}")

        cfg = self.cbf_config
        bw = float(cfg.kde_bandwidth)
        radius = float(cfg.interaction_radius)

        if planar:
            if axes_2d is None or len(axes_2d) != 2:
                raise ValueError("planar=True requires axes_2d=(i,j).")

            # --- work in the selected 2D plane ---
            positions_2d = positions_prev[:, axes_2d]                      # (N,2)
            position_2d = positions_prev[vehicle_idx, axes_2d]            # (2,)
            diffs = positions_2d - position_2d                              # (N,2)
            dists = np.linalg.norm(diffs, axis=1)        # (N,)

            neigh_mask = (dists <= radius) & (dists >= 0.0)


            # C = 2*pi*bw^2 * (1 - exp(-R^2 / (2*bw^2)));
            kde_norm_const_2d_truncated = (
                2.0 * np.pi * bw**2 * (1.0 - np.exp(-radius**2 / (2*bw**2)))
            )

            if not np.any(neigh_mask):
                rho_m = 0.0
                grad_2d = np.zeros((2,), dtype=float)
            else:
                diffs_k = diffs[neigh_mask]                          # (K,2)
                dists_k = dists[neigh_mask]                          # (K,)
                w = np.exp(-(dists_k ** 2) / (2.0 * (bw ** 2)))      # (K,)

                rho_m = float(np.sum(w) / (float(self.num_vehicles) * kde_norm_const_2d_truncated))

                grad_2d = -(1.0 / (float(self.num_vehicles) * kde_norm_const_2d_truncated * (bw ** 2))) * np.sum(
                    diffs_k * w[:, None], axis=0
                )

            grad_3d = np.zeros((3,), dtype=float)
            grad_3d[axes_2d[0]] = grad_2d[0]
            grad_3d[axes_2d[1]] = grad_2d[1]
            return rho_m, grad_3d

        else:
            # --- 3D KDE ---
            diffs = positions_prev - positions_prev[vehicle_idx, :]                 # (N,3)
            dists = np.linalg.norm(diffs, axis=1)         # (N,)
            neigh_mask = (dists <= radius) & (dists > 0.0)

            # 3D Gaussian normalization
            kde_norm_const_3d_truncated = (2.0 * np.pi) ** (1.5) * (bw ** 3)

            if not np.any(neigh_mask):
                rho_m = 0.0
                grad_3d = np.zeros((3,), dtype=float)
            else:
                diffs_k = diffs[neigh_mask]                          # (K,3)
                dists_k = dists[neigh_mask]                          # (K,)
                w = np.exp(-(dists_k ** 2) / (2.0 * (bw ** 2)))      # (K,)

                rho_m = float(np.sum(w) / (float(self.num_vehicles) * kde_norm_const_3d_truncated))

                grad_3d = -(1.0 / (float(self.num_vehicles) * kde_norm_const_3d_truncated * (bw ** 2))) * np.sum(
                    diffs_k * w[:, None], axis=0
                )

            return rho_m, grad_3d

    def _apply_cbf_projection(
        self,
        vel_nominal: np.ndarray,
        density: float,
        density_grad: np.ndarray,
    ) -> np.ndarray:
        """
        Project a nominal velocity onto density-based control barrier function
        (CBF) constraints for a single agent.

        The projection enforces affine bounds on the density time derivative:
            rho_dot >= alpha_upper * (rho - rho_upper)
            rho_dot <= alpha_lower * (rho - rho_lower)

        where:
            rho_dot := (density_grad)^T * vel

        The constraints are applied sequentially via orthogonal projection
        along the density gradient direction.

        After projection, the velocity is saturated to unit norm to enforce
        a maximum speed constraint.
        """
        if self.cbf_config is None:
            raise RuntimeError("_apply_cbf_projection called but cbf_config is None.")

        cfg = self.cbf_config

        vel = np.asarray(vel_nominal, dtype=float).reshape(3,).copy()
        grad_rho = np.asarray(density_grad, dtype=float).reshape(3,)

        # Bounds on rho_dot := grad_rho^T * vel
        rho_dot_min = float(cfg.density_upper_gain) * (float(density) - float(cfg.density_upper_bound))
        rho_dot_max = float(cfg.density_lower_gain) * (float(density) - float(cfg.density_lower_bound))

        grad_norm = float(np.linalg.norm(grad_rho))
        if grad_norm > float(cfg.grad_norm_eps):
            grad_sq_norm = float(grad_rho @ grad_rho)   # ||grad_rho||^2
            rho_dot = float(grad_rho @ vel)             # density time derivative

            # Enforce lower bound on rho_dot
            if rho_dot < rho_dot_min:
                vel = vel + ((rho_dot_min - rho_dot) / grad_sq_norm) * grad_rho
                rho_dot = rho_dot_min

            # Enforce upper bound on rho_dot
            if rho_dot > rho_dot_max:
                vel = vel + ((rho_dot_max - rho_dot) / grad_sq_norm) * grad_rho

        # Enforce maximum speed constraint
        vel_norm = float(np.linalg.norm(vel))
        if vel_norm > 1.0:
            vel = vel * (1.0 / vel_norm)

        return vel


    def step(self, cur_time: float, states: List[Dict[str, Any]]) -> None:
        """
        DOOT step (CBF optional via self.apply_cbf).

        Inputs:
        cur_time: global simulation time (seconds)
        states: list of length self.num_vehicles, only for member vehicles; each must include 'x' (3,)

        Behavior:
        - compute dt = cur_time - last_time
        - update phi via primal-dual iterations
        - build interpolant evaluate_phi_with_fallback at current agent positions
        - choose trial move per agent: argmin(||d|| + evaluate_phi_with_fallback(x+d)) with ||d|| >= min_displacement_norm
        - convert displacement to velocity v = d/dt
        - optionally apply CBF filter to v_nom
        - store v in _transport_vel
        """
        if len(states) != self.num_vehicles:
            raise ValueError(f"states must have length num_vehicles={self.num_vehicles}, got {len(states)}")

        # Initialize time on first call
        if self.last_time is None:
            self.last_time = float(cur_time)
            self.set_transport_vel_batch(np.zeros((self.num_vehicles, 3), dtype=float))
            return

        dt = float(cur_time) - float(self.last_time)
        self.last_time = float(cur_time)

        if (not np.isfinite(dt)) or dt <= 0.0:
            self.set_transport_vel_batch(np.zeros((self.num_vehicles, 3), dtype=float))
            return

        # Extract vehicles' positions (num_vehicles,3)
        positions = np.zeros((self.num_vehicles, 3), dtype=float)
        for i, s in enumerate(states):
            if "x" not in s:
                raise KeyError("Each state must contain key 'x' with shape (3,).")
            positions[i, :] = np.asarray(s["x"], dtype=float).reshape(3,)

        # CBF runtime state init (previous positions)
        if self._positions_prev_cbf is None:
            self._positions_prev_cbf = positions.copy()

        # ------------------------------------------------------------
        # Planar detection (Option A): axis-aligned std check.
        # If planar, choose the 2 axes with the largest std and drop the smallest.
        # TODO (Option B): use SVD/rank-based plane detection for rotated planes.
        # ------------------------------------------------------------
        std_xyz = np.std(positions, axis=0)  # (3,)
        planar = bool(np.any(std_xyz <= float(self.doot_config.planar_std_threshold)))

        if planar:
            axes_2d = tuple(np.argsort(std_xyz)[-2:].tolist())  # e.g. (0,2) for XZ plane
            dropped_axis = int(np.argsort(std_xyz)[0])
            positions_2d = positions[:, axes_2d]                           # (num_vehicles,2)
            targeted_positions = self.targeted_positions[:, axes_2d]     # (num_targeted_pos,2)
        else:
            axes_2d = None
            dropped_axis = None
            positions_2d = positions                                       # (num_vehicles,3)
            targeted_positions = self.targeted_positions                 # (num_targeted_pos,3)

        # ------------------------------------------------------------
        # 1) count: assign each target sample to nearest agent
        # ------------------------------------------------------------
        num_targeted_pos = int(self.targeted_positions.shape[0])
        diff_ta = targeted_positions[:, None, :] - positions_2d[None, :, :]        # (num_targeted_pos,num_vehicles,dim)
        d2_ta = np.sum(diff_ta * diff_ta, axis=2)        # (num_targeted_pos,num_vehicles)
        nearest_agent = np.argmin(d2_ta, axis=1)         # (num_targeted_pos,) nearest_agent[i] = j means i-th sample is assigned to j-th agent
        count = np.bincount(nearest_agent, minlength=self.num_vehicles).astype(float) / float(num_targeted_pos)

        # ------------------------------------------------------------
        # 2) Neighbor graph + Laplacian
        # ------------------------------------------------------------
        n_neigh = int(self.doot_config.num_neighbors)
        if n_neigh < 2:
            raise ValueError("doot_config.num_neighbors must be >= 2 to (includes self then drops).")

	# TODO: If the agents’ distribution is non-planar, the difference in the z-coordinate must be taken into account.
        # K including self (MATLAB knnsearch K=n_neigh includes self)
        K_including_self = min(n_neigh, self.num_vehicles)
        k = K_including_self - 1  # neighbors excluding self (MATLAB uses 2:n_neigh)

        # Pairwise squared distances INCLUDING self
        diff_aa = positions_2d[:, None, :] - positions_2d[None, :, :]        # (N,N,dim)
        d2_aa = np.sum(diff_aa * diff_aa, axis=2)                            # (N,N)

        # Ensure self is always in the K slice
        np.fill_diagonal(d2_aa, -np.inf)

        # Unordered K-candidate set (includes self); then we sort within the set by distance
        kth = min(K_including_self - 1, self.num_vehicles - 1)
        nn_full = np.argpartition(d2_aa, kth=kth, axis=1)[:, :K_including_self]  # (N,K_including_self)

        # Drop self and keep the closest k non-self neighbors (MATLAB columns 2..n_neigh)
        nn_idx_list = []
        for i in range(self.num_vehicles):
            row = nn_full[i]
            row = row[np.argsort(d2_aa[i, row])]       # sort candidates by distance
            row_wo_self = row[row != i]               # drop self
            nn_idx_list.append(row_wo_self[:k])       # take k neighbors

        nn_idx = np.stack(nn_idx_list, axis=0)        # (N,k)

        # Directed adjacency A (N x N)
        adjacency_directed = np.zeros((self.num_vehicles, self.num_vehicles), dtype=float)
        rows = np.repeat(np.arange(self.num_vehicles), k)
        cols = nn_idx.reshape(-1)
        adjacency_directed[rows, cols] = 1.0

        W = 0.5 * (adjacency_directed + adjacency_directed.T)                # values in {0, 0.5, 1}

        # UNWEIGHTED Laplacian of structural adjacency
        A_struct = (W > 0.0).astype(float)                                   # {0,1}
        degree_matrix = np.diag(np.sum(A_struct, axis=1))
        laplacian = degree_matrix - A_struct

        # DEBUG: cache what step() actually used
        self._dbg_last_count = count.copy()
        self._dbg_last_laplacian = laplacian.copy()

        # ------------------------------------------------------------
        # 3) Primal–dual inner iterations updating phi
        # ------------------------------------------------------------
        normalization = 1.0 / float(self.num_vehicles)
        gain = 1.0 / float(n_neigh + 1)

        phi = self.phi
        for _ in range(int(self.doot_config.max_iter_primaldual)):
            phi = phi - gain * (laplacian @ phi) + normalization * np.ones((self.num_vehicles,), dtype=float) - count
        self.phi = phi

        # ------------------------------------------------------------
        # 4) Scattered interpolant evaluate_phi_with_fallback in dim=2 (planar) or dim=3 (full)
        # ------------------------------------------------------------
        phi_lin = LinearNDInterpolator(positions_2d, self.phi, fill_value=np.nan)
        phi_nn = NearestNDInterpolator(positions_2d, self.phi)

        def evaluate_phi_with_fallback(pos: np.ndarray) -> np.ndarray:
            v = phi_lin(pos)
            v = np.asarray(v, dtype=float)
            mask = ~np.isfinite(v)

            # Use nearest-neighbor if linear interpolation is undefined
            if np.any(mask):
                v[mask] = phi_nn(pos[mask])
            return v

        # ------------------------------------------------------------
        # 5) Transport step via trial moves (nominal)
        # ------------------------------------------------------------
        v_nom = np.zeros((self.num_vehicles, 3), dtype=float)

        disp = self.rng.normal(
            loc=self.mean_trial_move,
            scale=np.sqrt(self.var_trial_move),
            size=(int(self.doot_config.num_trial_move_samples), 3),
        )

        if planar:
            # Keep trial moves within the detected plane
            disp[:, dropped_axis] = 0.0

        norms = np.linalg.norm(disp, axis=1)
        keep = norms >= float(self.doot_config.min_displacement_norm)
        disp_kept = disp[keep]
        norms_kept = norms[keep]

        if disp_kept.shape[0] == 0:
            self.set_transport_vel_batch(np.zeros((self.num_vehicles, 3), dtype=float))
            self._positions_prev_cbf = positions.copy()
            return

        for vehicle_idx in range(self.num_vehicles):
            pos_trial = positions[vehicle_idx, :][None, :] + disp_kept  # (K,3)

            if planar:
                pos_eval = pos_trial[:, axes_2d]      # (K,2)
            else:
                pos_eval = pos_trial                  # (K,3)

            cost = norms_kept + evaluate_phi_with_fallback(pos_eval)
            ind = int(np.argmin(cost))
            v_disp = pos_trial[ind, :] - positions[vehicle_idx, :]
            v_nom[vehicle_idx, :] = v_disp / dt

        # ------------------------------------------------------------
        # 6) Optional CBF filtering
        # ------------------------------------------------------------
        if not self.apply_cbf:
            self.set_transport_vel_batch(v_nom)
            self._positions_prev_cbf = positions.copy()
            return

        if self.cbf_config is None:
            raise RuntimeError("apply_cbf=True but cbf_config is None.")

        v = v_nom.copy()
        positions_prev = self._positions_prev_cbf

        for vehicle_idx in range(self.num_vehicles):
            rho_m, grad_rho_m = self._compute_kde_density_and_gradient(
                positions_prev, vehicle_idx, planar=planar, axes_2d=axes_2d)
            v[vehicle_idx, :] = self._apply_cbf_projection(v_nom[vehicle_idx, :], rho_m, grad_rho_m)

        self.set_transport_vel_batch(v)
        self._positions_prev_cbf = positions.copy()


def compute_kde_density_and_gradient(
        positions_prev: np.ndarray,
        vehicle_idx: int,
        num_vehicles: int,
        bw: float,
        radius: float,
        *,
        planar: bool,
        axes_2d: Optional[Tuple[int, int]],
    ) -> Tuple[float, np.ndarray]:
        """
        KDE rho and grad_rho at agent vehicle_idx using previous positions, consistent with DOOT plane.

        If planar=True:
        - compute KDE in the DOOT plane defined by axes_2d (two indices in {0,1,2})
        - return a (3,) gradient with nonzeros only on axes_2d

        If planar=False:
        - compute KDE in 3D and return full (3,) gradient
        """
        positions_prev = np.asarray(positions_prev, dtype=float)
        if positions_prev.shape != (num_vehicles, 3):
            raise ValueError(f"positions_prev must have shape {(num_vehicles, 3)}, got {positions_prev.shape}")
        if not (0 <= vehicle_idx < num_vehicles):
            raise IndexError(f"vehicle_idx out of range: {vehicle_idx}")

        if planar:
            if axes_2d is None or len(axes_2d) != 2:
                raise ValueError("planar=True requires axes_2d=(i,j).")

            # --- work in the selected 2D plane ---
            positions_2d = positions_prev[:, axes_2d]                      # (N,2)
            position_2d = positions_prev[vehicle_idx, axes_2d]            # (2,)
            diffs = positions_2d - position_2d                              # (N,2)
            dists = np.linalg.norm(diffs, axis=1)        # (N,)

            neigh_mask = (dists <= radius ) & (dists >= 0.0)

            kde_norm_const_2d_truncated = (
                2.0 * np.pi * bw**2 * (1.0 - np.exp(-radius**2 / (2*bw**2)))
            )

            if not np.any(neigh_mask):
                rho_m = 0.0
                grad_2d = np.zeros((2,), dtype=float)
            else:
                diffs_k = diffs[neigh_mask]                          # (K,2)
                dists_k = dists[neigh_mask]                          # (K,)
                w = np.exp(-(dists_k ** 2) / (2.0 * (bw ** 2)))      # (K,)

                rho_m = float(np.sum(w) / (float(num_vehicles) * kde_norm_const_2d_truncated))

                grad_2d = -(1.0 / (float(num_vehicles) * kde_norm_const_2d_truncated * (bw ** 2))) * np.sum(
                    diffs_k * w[:, None], axis=0
                )

            grad_3d = np.zeros((3,), dtype=float)
            grad_3d[axes_2d[0]] = grad_2d[0]
            grad_3d[axes_2d[1]] = grad_2d[1]
            return rho_m, grad_3d

        else:
            # --- 3D KDE ---
            diffs = positions_prev - positions_prev[vehicle_idx, :]                 # (N,3)
            dists = np.linalg.norm(diffs, axis=1)         # (N,)
            neigh_mask = (dists <= radius) & (dists > 0.0)

            # 3D Gaussian normalization
            kde_norm_const_3d_truncated = (2.0 * np.pi) ** (1.5) * (bw ** 3)

            if not np.any(neigh_mask):
                rho_m = 0.0
                grad_3d = np.zeros((3,), dtype=float)
            else:
                diffs_k = diffs[neigh_mask]                          # (K,3)
                dists_k = dists[neigh_mask]                          # (K,)
                w = np.exp(-(dists_k ** 2) / (2.0 * (bw ** 2)))      # (K,)

                rho_m = float(np.sum(w) / (float(num_vehicles) * kde_norm_const_3d_truncated))

                grad_3d = -(1.0 / (float(num_vehicles) * kde_norm_const_3d_truncated * (bw ** 2))) * np.sum(
                    diffs_k * w[:, None], axis=0
                )

            return rho_m, grad_3d


def unit_test_01():
    import copy
    import os

    import numpy as np
    import matplotlib.pyplot as plt
    import cv2

    # Initialize the "world"
    num_vehicles = 20
    vehicles = [None] * num_vehicles

    targeted_positions = [[0.0, 0.0, 0.0]]

    sim_time = 200
    dt = 1
    intervals = int(sim_time / dt)

    # Initialize the vehicles' initial states
    bounds = np.array([[-5, 5], [-5, 5], [-0.5, 3]], dtype=float)
    mean_init = np.array([4.0, 4.0, 0.0], dtype=float)
    std_init  = np.sqrt(np.array([1.0, 3.0, 0.0], dtype=float))

    rng = np.random.default_rng(0)
    x0_positions = np.empty((0, 3), dtype=float)

    while x0_positions.shape[0] < num_vehicles:
        batch = rng.normal(loc=mean_init, scale=std_init, size=(num_vehicles, 3))
        batch[:, 2] = mean_init[2]
        ok = np.all((batch >= bounds[:, 0]) & (batch <= bounds[:, 1]), axis=1)
        x0_positions = np.vstack([x0_positions, batch[ok]])

    x0_positions = x0_positions[:num_vehicles]

    x0s = []
    for i in range(num_vehicles):
        x0s.append(
            {
                "x": x0_positions[i].astype(float),
                "v": np.zeros(3),
                "q": np.array([0.0, 0.0, 0.0, 1.0]),
                "w": np.zeros(3),
                "wind": np.zeros(3),
                "rotor_speeds": np.full(4, 1788.53),
            }
        )

    doot_config = DootConfig(
        num_neighbors=min(5, num_vehicles - 1),
        max_iter_primaldual=10,
        use_random_sampling=False,
        num_trial_move_samples=300,
        min_displacement_norm=0.1,
        planar_std_threshold=0.1,
    )

    coordinator = DootCbfCoordinator(
        vehicles=vehicles,
        velocity_max=1.0,
        targeted_positions=targeted_positions,
        doot_config=doot_config,
    )

    # Output path: same directory as doot_cbf_coordinator.py
    out_dir = os.path.dirname(os.path.abspath(__file__))
    avi_path = os.path.join(out_dir, "doot_cbf_coordinator_ut01.avi")

    cur_time = 0.0
    cur_x = copy.deepcopy(x0s)

    # --- Matplotlib 2D plot setup ---
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(bounds[0, 0], bounds[0, 1])
    ax.set_ylim(bounds[1, 0], bounds[1, 1])
    ax.set_aspect("equal")
    ax.grid(True)
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    target_xy = np.array(targeted_positions, dtype=float)[:, :2]
    ax.plot(target_xy[:, 0], target_xy[:, 1], "rx", markersize=10)

    # Initial xy
    xy0 = np.array([cur_x[i]["x"][:2] for i in range(num_vehicles)], dtype=float)

    # Random per-vehicle colors (different each run is OK)
    colors = np.random.rand(num_vehicles, 3)  # RGB in [0,1]

    scat = ax.scatter(xy0[:, 0], xy0[:, 1], s=20, c=colors)

    fig.canvas.draw()

    # --- OpenCV VideoWriter ---
    width, height = fig.canvas.get_width_height()
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    video = cv2.VideoWriter(avi_path, fourcc, 10, (width, height))

    for step in range(intervals):
        coordinator.step(cur_time, cur_x)
        transport_vel_fns = coordinator.get_transport_vel_fns()

        for i in range(num_vehicles):
            v = np.asarray(transport_vel_fns[i](cur_time), dtype=float).reshape(3,)
            cur_x[i]["x"] += v * dt

        xy = np.array([cur_x[i]["x"][:2] for i in range(num_vehicles)], dtype=float)
        scat.set_offsets(xy)
        ax.set_title(f"cur_time = {cur_time:.2f} s")

        fig.canvas.draw()

        # Backend-safe capture for TkAgg: ARGB buffer
        buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
        buf = buf.reshape(height, width, 4)

        # ARGB -> BGR for OpenCV (drop alpha)
        frame = buf[:, :, [3, 2, 1]]
        video.write(frame)

        cur_time += dt

    video.release()
    plt.close(fig)


def unit_test_02():
    import copy
    import os

    import numpy as np
    import matplotlib.pyplot as plt
    import cv2
    from scipy.stats import multivariate_normal

    # =========================
    # Helper: planar detection (same rule as coordinator.step)
    # =========================
    def _detect_planar_axes(positions: np.ndarray, planar_std_threshold: float):
        std_xyz = np.std(positions, axis=0)  # (3,)
        planar = bool(np.any(std_xyz <= float(planar_std_threshold)))
        if planar:
            axes_2d = tuple(np.argsort(std_xyz)[-2:].tolist())  # keep best 2 axes
        else:
            axes_2d = None
        return planar, axes_2d

    # =========================
    # Setup
    # =========================
    num_vehicles = 200
    vehicles = [None] * num_vehicles

    num_targets = 500
    components = 20
    dom_size = 5

    sim_time = 200
    dt = 1
    intervals = int(sim_time / dt)

    rng = np.random.default_rng(10)

    # =========================
    # Target distribution
    # =========================
    mean_des = 8.0 * (rng.random((components, 2)) - 0.5)
    var_des = np.array([0.25, 0.25], dtype=float)
    cov_des = np.diag(var_des)

    comp_ids = rng.integers(low=0, high=components, size=num_targets)
    samples_xy = np.empty((num_targets, 2), dtype=float)
    for k in range(components):
        idx = np.where(comp_ids == k)[0]
        if idx.size > 0:
            samples_xy[idx, :] = rng.multivariate_normal(
                mean=mean_des[k], cov=cov_des, size=idx.size
            )

    targeted_positions = np.hstack([samples_xy, np.zeros((num_targets, 1), dtype=float)]).tolist()

    # =========================
    # Initial agent distribution
    # =========================
    mean_init = np.array([5.0, 5.0], dtype=float)
    var_init = np.array([2.0, 6.0], dtype=float)
    cov_init = np.diag(var_init)

    pos_init_xy = rng.multivariate_normal(mean=mean_init, cov=cov_init, size=num_vehicles)

    x0s = []
    for i in range(num_vehicles):
        x0s.append(
            {
                "x": np.array([pos_init_xy[i, 0], pos_init_xy[i, 1], 0.0], dtype=float),
                "v": np.zeros(3),
                "q": np.array([0.0, 0.0, 0.0, 1.0]),
                "w": np.zeros(3),
                "wind": np.zeros(3),
                "rotor_speeds": np.full(4, 1788.53),
            }
        )

    # =========================
    # Configs
    # =========================
    doot_config = DootConfig(
        num_neighbors=min(10, num_vehicles - 1),
        max_iter_primaldual=20,
        use_random_sampling=False,
        num_trial_move_samples=300,
        min_displacement_norm=0.1,
        planar_std_threshold=0.1,
    )

    cbf_config = CbfConfig(
        density_upper_gain=100.0,
        density_lower_gain=1000.0,
        density_upper_bound=0.045,
        density_lower_bound=0.011,
        kde_bandwidth=0.3,
    )

    # Compute DOOT once per step (apply_cbf=False here on purpose)
    coordinator = DootCbfCoordinator(
        vehicles=vehicles,
        velocity_max=1.0,
        targeted_positions=targeted_positions,
        doot_config=doot_config,
        apply_cbf=False,
        cbf_config=cbf_config,
    )

    # =========================
    # Output video
    # =========================
    out_dir = os.path.dirname(os.path.abspath(__file__))
    avi_path = os.path.join(out_dir, "doot_cbf_coordinator_ut02_side_by_side.avi")

    cur_nom = copy.deepcopy(x0s)
    cur_cbf = copy.deepcopy(x0s)

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(12, 6))
    for ax in (ax_l, ax_r):
        ax.set_aspect("equal")
        ax.grid(True)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.scatter(samples_xy[:, 0], samples_xy[:, 1], s=8, c="k", alpha=0.15, linewidths=0)

    colors = rng.random((num_vehicles, 3))
    xy0 = np.array([cur_nom[i]["x"][:2] for i in range(num_vehicles)], dtype=float)

    # Agent scatters
    scatL = ax_l.scatter(xy0[:, 0], xy0[:, 1], s=20, c=colors)
    scatR = ax_r.scatter(xy0[:, 0], xy0[:, 1], s=20, c=colors)

    # Markers:
    # - yellow squares: sparse (rho < epsilon_min)
    # - red circles (open): crowded / unsafe (rho >= epsilon)
    sparse_l = ax_l.scatter(
        [], [], s=120, marker="s",
        facecolors=(1.0, 0.8, 0.2),
        edgecolors=(1.0, 0.6, 0.0),
        linewidths=1.5,
        alpha=0.4,
        zorder=3,
    )
    sparse_r = ax_r.scatter(
        [], [], s=120, marker="s",
        facecolors=(1.0, 0.8, 0.2),
        edgecolors=(1.0, 0.6, 0.0),
        linewidths=1.5,
        alpha=0.4,
        zorder=3,
    )

    crowd_l = ax_l.scatter(
        [], [], s=140, marker="o",
        facecolors="none", edgecolors="r",
        linewidths=1.5,
        zorder=4,
    )
    crowd_r = ax_r.scatter(
        [], [], s=140, marker="o",
        facecolors="none", edgecolors="r",
        linewidths=1.5,
        zorder=4,
    )

    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    video = cv2.VideoWriter(
        avi_path,
        cv2.VideoWriter_fourcc(*"XVID"),
        10,
        (width, height),
    )

    eps = float(cbf_config.density_upper_bound)
    eps_min = float(cbf_config.density_lower_bound)

    cur_time = 0.0

    # Initialize previous positions
    x_prev_nom = np.array([cur_nom[i]["x"] for i in range(num_vehicles)], dtype=float)
    x_prev_cbf = np.array([cur_cbf[i]["x"] for i in range(num_vehicles)], dtype=float)

    for step in range(intervals):
        # Previous positions for this step
        x_prev_nom = np.array([cur_nom[i]["x"] for i in range(num_vehicles)], dtype=float)
        x_prev_cbf = np.array([cur_cbf[i]["x"] for i in range(num_vehicles)], dtype=float)

        # Planar detection for KDE, consistent with the same rule used in coordinator.step
        planar_nom, axes_2d_nom = _detect_planar_axes(x_prev_nom, doot_config.planar_std_threshold)
        planar_cbf, axes_2d_cbf = _detect_planar_axes(x_prev_cbf, doot_config.planar_std_threshold)

        # ---- DOOT (nominal) computed once per step ----
        coordinator.step(cur_time, cur_nom)
        v_nom = np.array([coordinator.get_transport_vel_fns()[i](cur_time) for i in range(num_vehicles)], dtype=float)
        v_nom[:, 2] = 0.0  # enforce 2D for the unit test

        # ---- CBF projection branch: v_cbf = projection(v_nom) using x_prev_cbf ----
        v_cbf = v_nom.copy()
        for vehicle_idx in range(num_vehicles):
            rho_m, grad_rho_m = coordinator._compute_kde_density_and_gradient(
                x_prev_cbf, vehicle_idx,
                planar=planar_cbf, axes_2d=axes_2d_cbf
            )
            v_cbf[vehicle_idx, :] = coordinator._apply_cbf_projection(v_nom[vehicle_idx, :], rho_m, grad_rho_m)
        v_cbf[:, 2] = 0.0

        # ---- Integrate positions (two separate worlds) ----
        for i in range(num_vehicles):
            cur_nom[i]["x"] += dt * v_nom[i]
            cur_cbf[i]["x"] += dt * v_cbf[i]
            cur_nom[i]["x"][2] = 0.0
            cur_cbf[i]["x"][2] = 0.0

        xy_nom = np.array([cur_nom[i]["x"][:2] for i in range(num_vehicles)], dtype=float)
        xy_cbf = np.array([cur_cbf[i]["x"][:2] for i in range(num_vehicles)], dtype=float)

        scatL.set_offsets(xy_nom)
        scatR.set_offsets(xy_cbf)

        # Extract the position for KDE density calculation
        x_cur_nom = np.array([cur_nom[i]["x"] for i in range(num_vehicles)], dtype=float)
        x_cur_cbf = np.array([cur_cbf[i]["x"] for i in range(num_vehicles)], dtype=float)

        # ---- KDE-based overlays (computed from X_prev_*) ----
        rho_nom = np.array(
            [
                compute_kde_density_and_gradient(
                    x_cur_nom, vehicle_idx, num_vehicles,
                    cbf_config.kde_bandwidth, cbf_config.kde_radius_bar,
                    planar=planar_nom, axes_2d=axes_2d_nom
                )[0]
                for vehicle_idx in range(num_vehicles)
            ],
            dtype=float,
        )
        rho_cbf = np.array(
            [
                compute_kde_density_and_gradient(
                    x_cur_cbf, vehicle_idx, num_vehicles,
                    cbf_config.kde_bandwidth, cbf_config.kde_radius_bar,
                    planar=planar_cbf, axes_2d=axes_2d_cbf
                )[0]
                for vehicle_idx in range(num_vehicles)
            ],
            dtype=float,
        )

        sparse_l.set_offsets(xy_nom[rho_nom < eps_min])
        sparse_r.set_offsets(xy_cbf[rho_cbf < eps_min])

        crowd_l.set_offsets(xy_nom[rho_nom >= eps])
        crowd_r.set_offsets(xy_cbf[rho_cbf >= eps])

        ax_l.set_title(f"Nominal (no CBF)   cur_time = {cur_time:.2f} s")
        ax_r.set_title(f"With CBF          cur_time = {cur_time:.2f} s")

        # Write frame
        fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
        buf = buf.reshape(height, width, 4)
        frame = buf[:, :, [3, 2, 1]]  # ARGB -> BGR
        video.write(frame)

        cur_time += dt

    video.release()
    plt.close(fig)


if __name__ == "__main__":
    import time

    print("==== Start DootCbfCoordinator Unit Tests ====")

    t0 = time.perf_counter()
    unit_test_01()
    t1 = time.perf_counter()
    print(f"    Complete multiple vehicles - one target unit test in {t1 - t0:.3f} seconds")

    t2 = time.perf_counter()
    unit_test_02()
    t3 = time.perf_counter()
    print(f"    Complete multiple vehicles - multiple targets w/ and w/o CBF test in {t3 - t2:.3f} seconds")

    print("==== Finish DootCbfCoordinator Unit Tests ====")
