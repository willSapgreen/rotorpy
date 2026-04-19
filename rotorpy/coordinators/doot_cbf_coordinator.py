from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Dict, Any, Tuple, Union

import numpy as np
import torch
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
    Coordinator that stores per-vehicle commanded velocities.

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
        idxs: Optional[List[int]] = None,
    ):
        # Initialize vehicles setting
        self._vehicles = vehicles
        self._num_vehicles = len(self._vehicles)
        if self._num_vehicles < 1:
            raise ValueError("vehicles must be a non-empty list.")
        self._velocity_max = velocity_max  # scalar cap (m/s) for all vehicles in this coordinator

        # Initialize target samples, (num_targeted_pos,3), setting
        self._targeted_positions = np.asarray(targeted_positions, dtype=float)
        if self._targeted_positions.ndim != 2 or self._targeted_positions.shape[1] != 3:
            raise ValueError(
                f"targeted_positions must have shape (num_targeted_pos,3), got {self._targeted_positions.shape}"
            )

        # Set up DOOT config
        self._doot_config = doot_config
        if self._doot_config.num_neighbors < 1:
            raise ValueError("doot_config.num_neighbors must be >= 1 (excluding self).")
        if self._doot_config.max_iter_primaldual < 1:
            raise ValueError("doot_config.max_iter_primaldual must be >= 1.")
        if self._doot_config.use_random_sampling:
            if self._doot_config.num_trial_move_samples < 1:
                raise ValueError("doot_config.num_trial_move_samples must be >= 1.")
            if self._doot_config.min_displacement_norm < 0.0:
                raise ValueError("doot_config.min_displacement_norm must be >= 0.")

        # Set up trial-move distribution defaults (3D)
        if self._doot_config.mean_trial_move is None:
            mean_trial_move = [0.0, 0.0, 0.0]
        else:
            mean_trial_move = self._doot_config.mean_trial_move

        if self._doot_config.var_trial_move is None:
            var_trial_move = [0.05, 0.05, 0.05]
        else:
            var_trial_move = self._doot_config.var_trial_move

        self._mean_trial_move = np.asarray(mean_trial_move, dtype=float).reshape(3,)
        self._var_trial_move = np.asarray(var_trial_move, dtype=float).reshape(3,)
        if np.any(self._var_trial_move < 0.0):
            raise ValueError("var_trial_move must be nonnegative per axis.")

        # Set up DOOT runtime state
        self._phi = np.zeros((self._num_vehicles,), dtype=float)  # dual potential at agent locations
        self._last_time: Optional[float] = None

        # Set up random number generator (RNG)
        # - use_random_sampling=True  -> non-deterministic seed (system entropy)
        # - use_random_sampling=False -> deterministic seed (_FIXED_SEED)
        seed = None if self._doot_config.use_random_sampling else self._FIXED_SEED
        self._rng: np.random.Generator = np.random.default_rng(seed)

        # Set up command storage (latest velocity commands, Nx3)
        self._vel_cmds: np.ndarray = np.zeros((self._num_vehicles, 3), dtype=float)

        # CBF config
        self._apply_cbf = bool(apply_cbf)
        self._cbf_config = cbf_config
        if self._apply_cbf and self._cbf_config is None:
            raise ValueError("apply_cbf=True requires a valid CbfConfig.")

        # CBF runtime state (previous positions only)
        self._positions_prev_cbf: Optional[np.ndarray] = None

        # Store indices for this coordinator (subset of the full swarm)
        if idxs is None:
            raise ValueError(
                "idxs must be provided when using multiple coordinators (global indexing required)."
            )

        self._idxs = np.asarray(idxs, dtype=int).reshape(-1)

        if self._idxs.size < 1:
            raise ValueError("idxs must be non-empty.")
        if np.any(self._idxs < 0):
            raise ValueError("idxs must be nonnegative.")

        # KEY CHECK: subset sizes must match
        if self._idxs.size != len(vehicles):
            raise ValueError(
                f"idxs length ({self._idxs.size}) must match len(vehicles) ({len(vehicles)})."
            )

        # OPTIONAL (recommended): indices must be unique within this coordinator
        if np.unique(self._idxs).size != self._idxs.size:
            raise ValueError("idxs must not contain duplicates within a coordinator.")


    def _set_vel_cmds(self, vel_batch: np.ndarray) -> None:
        vel_batch = np.asarray(vel_batch, dtype=float)

        # Enforce global velocity cap
        if self._velocity_max is not None:
            speeds = np.linalg.norm(vel_batch, axis=1)
            mask = speeds > self._velocity_max
            if np.any(mask):
                vel_batch = vel_batch.copy()
                vel_batch[mask] *= (self._velocity_max / speeds[mask])[:, None]

        self._vel_cmds = vel_batch


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
        if self._cbf_config is None:
            raise RuntimeError("_compute_kde_density_and_gradient called but cbf_config is None.")

        positions_prev = np.asarray(positions_prev, dtype=float)
        if positions_prev.shape != (self._num_vehicles, 3):
            raise ValueError(f"positions_prev must have shape {(self._num_vehicles, 3)}, got {positions_prev.shape}")
        if not (0 <= vehicle_idx < self._num_vehicles):
            raise IndexError(f"vehicle_idx out of range: {vehicle_idx}")

        cfg = self._cbf_config
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

                rho_m = float(np.sum(w) / (float(self._num_vehicles) * kde_norm_const_2d_truncated))

                grad_2d = -(1.0 / (float(self._num_vehicles) * kde_norm_const_2d_truncated * (bw ** 2))) * np.sum(
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
            neigh_mask = (dists <= radius) & (dists >= 0.0)

            # 3D Gaussian normalization
            kde_norm_const_3d_truncated = (2.0 * np.pi) ** (1.5) * (bw ** 3)

            if not np.any(neigh_mask):
                rho_m = 0.0
                grad_3d = np.zeros((3,), dtype=float)
            else:
                diffs_k = diffs[neigh_mask]                          # (K,3)
                dists_k = dists[neigh_mask]                          # (K,)
                w = np.exp(-(dists_k ** 2) / (2.0 * (bw ** 2)))      # (K,)

                rho_m = float(np.sum(w) / (float(self._num_vehicles) * kde_norm_const_3d_truncated))

                grad_3d = -(1.0 / (float(self._num_vehicles) * kde_norm_const_3d_truncated * (bw ** 2))) * np.sum(
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
        if self._cbf_config is None:
            raise RuntimeError("_apply_cbf_projection called but cbf_config is None.")

        cfg = self._cbf_config

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


    def get_indices(self) -> np.ndarray:
        return self._idxs.copy()


    def step(self, cur_time: float, states: List[Dict[str, Any]]) -> None:
        """
        DOOT step (CBF optional via self._apply_cbf).

        Inputs:
        cur_time: global simulation time (seconds)
        states: list of length self._num_vehicles, only for member vehicles; each must include 'x' (3,)

        Behavior:
        - compute dt = cur_time - last_time
        - update phi via primal-dual iterations
        - build interpolant evaluate_phi_with_fallback at current agent positions
        - choose trial move per agent: argmin(||d|| + evaluate_phi_with_fallback(x+d)) with ||d|| >= min_displacement_norm
        - convert displacement to velocity v = d/dt
        - optionally apply CBF filter to v_nom
        - store v in _vel_cmds
        """
        if len(states) != self._num_vehicles:
            raise ValueError(f"states must have length num_vehicles={self._num_vehicles}, got {len(states)}")

        # Initialize time on first call
        if self._last_time is None:
            self._last_time = float(cur_time)
            self._set_vel_cmds(np.zeros((self._num_vehicles, 3)))
            return

        dt = float(cur_time) - float(self._last_time)
        self._last_time = float(cur_time)

        if (not np.isfinite(dt)) or dt <= 0.0:
            self._set_vel_cmds(np.zeros((self._num_vehicles, 3)))
            return

        # Extract vehicles' positions (num_vehicles,3)
        positions = np.zeros((self._num_vehicles, 3), dtype=float)
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
        planar = bool(np.any(std_xyz <= float(self._doot_config.planar_std_threshold)))

        if planar:
            axes_2d = tuple(np.argsort(std_xyz)[-2:].tolist())  # e.g. (0,2) for XZ plane
            dropped_axis = int(np.argsort(std_xyz)[0])
            positions_2d = positions[:, axes_2d]                           # (num_vehicles,2)
            targeted_positions = self._targeted_positions[:, axes_2d]     # (num_targeted_pos,2)
        else:
            axes_2d = None
            dropped_axis = None
            positions_2d = positions                                       # (num_vehicles,3)
            targeted_positions = self._targeted_positions                 # (num_targeted_pos,3)

        # ------------------------------------------------------------
        # 1) count: assign each target sample to nearest agent
        # ------------------------------------------------------------
        num_targeted_pos = int(self._targeted_positions.shape[0])
        diff_ta = targeted_positions[:, None, :] - positions_2d[None, :, :]        # (num_targeted_pos,num_vehicles,dim)
        d2_ta = np.sum(diff_ta * diff_ta, axis=2)        # (num_targeted_pos,num_vehicles)
        nearest_agent = np.argmin(d2_ta, axis=1)         # (num_targeted_pos,) nearest_agent[i] = j means i-th sample is assigned to j-th agent
        count = np.bincount(nearest_agent, minlength=self._num_vehicles).astype(float) / float(num_targeted_pos)

        # ------------------------------------------------------------
        # 2) Neighbor graph + Laplacian
        # ------------------------------------------------------------
        n_neigh = int(self._doot_config.num_neighbors)
        if n_neigh < 2:
            raise ValueError("doot_config.num_neighbors must be >= 2 for including self then dropping.")

        # TODO: If the agents’ distribution is non-planar, the difference in the z-coordinate must be taken into account.
        # K including self
        K_including_self = min(n_neigh, self._num_vehicles)
        k = K_including_self - 1  # neighbors excluding self

        # Pairwise squared distances INCLUDING self
        diff_aa = positions_2d[:, None, :] - positions_2d[None, :, :]        # (N,N,dim)
        d2_aa = np.sum(diff_aa * diff_aa, axis=2)                            # (N,N)

        # Ensure self is always in the K slice
        np.fill_diagonal(d2_aa, -np.inf)

        # Unordered K-candidate set (includes self); then we sort within the set by distance
        kth = min(K_including_self - 1, self._num_vehicles - 1)
        nn_full = np.argpartition(d2_aa, kth=kth, axis=1)[:, :K_including_self]  # (N,K_including_self)

        # Drop self and keep the closest k non-self neighbors
        nn_idx_list = []
        for i in range(self._num_vehicles):
            row = nn_full[i]
            row = row[np.argsort(d2_aa[i, row])]       # sort candidates by distance
            row_wo_self = row[row != i]               # drop self
            nn_idx_list.append(row_wo_self[:k])       # take k neighbors

        nn_idx = np.stack(nn_idx_list, axis=0)        # (N,k)

        # Directed adjacency A (N x N)
        adjacency_directed = np.zeros((self._num_vehicles, self._num_vehicles), dtype=float)
        rows = np.repeat(np.arange(self._num_vehicles), k)
        cols = nn_idx.reshape(-1)
        adjacency_directed[rows, cols] = 1.0

        W = 0.5 * (adjacency_directed + adjacency_directed.T)                # values in {0, 0.5, 1}

        # Create UNWEIGHTED Laplacian of structural adjacency
        A_struct = (W > 0.0).astype(float)                                   # {0,1}
        degree_matrix = np.diag(np.sum(A_struct, axis=1))
        laplacian = degree_matrix - A_struct

        # DEBUG: cache what step() actually used
        self._dbg_last_count = count.copy()
        self._dbg_last_laplacian = laplacian.copy()

        # ------------------------------------------------------------
        # 3) Primal–dual inner iterations updating phi
        # ------------------------------------------------------------
        normalization = 1.0 / float(self._num_vehicles)
        gain = 1.0 / float(n_neigh + 1)

        phi = self._phi
        for _ in range(int(self._doot_config.max_iter_primaldual)):
            phi = phi - gain * (laplacian @ phi) + normalization * np.ones((self._num_vehicles,), dtype=float) - count
        self._phi = phi

        # ------------------------------------------------------------
        # 4) Scattered interpolant evaluate_phi_with_fallback in dim=2 (planar) or dim=3 (full)
        # ------------------------------------------------------------
        phi_lin = LinearNDInterpolator(positions_2d, self._phi, fill_value=np.nan)
        phi_nn = NearestNDInterpolator(positions_2d, self._phi)

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
        v_nom = np.zeros((self._num_vehicles, 3), dtype=float)

        disp = self._rng.normal(
            loc=self._mean_trial_move,
            scale=np.sqrt(self._var_trial_move),
            size=(int(self._doot_config.num_trial_move_samples), 3),
        )

        if planar:
            # Keep trial moves within the detected plane
            disp[:, dropped_axis] = 0.0

        norms = np.linalg.norm(disp, axis=1)
        keep = norms >= float(self._doot_config.min_displacement_norm)
        disp_kept = disp[keep]
        norms_kept = norms[keep]

        if disp_kept.shape[0] == 0:
            self._set_vel_cmds(np.zeros((self._num_vehicles, 3)))
            self._positions_prev_cbf = positions.copy()
            return

        for vehicle_idx in range(self._num_vehicles):
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
        if not self._apply_cbf:
            self._set_vel_cmds(v_nom)
            self._positions_prev_cbf = positions.copy()
            return

        if self._cbf_config is None:
            raise RuntimeError("apply_cbf=True but cbf_config is None.")

        v = v_nom.copy()
        positions_prev = self._positions_prev_cbf

        for vehicle_idx in range(self._num_vehicles):
            rho_m, grad_rho_m = self._compute_kde_density_and_gradient(
                positions_prev, vehicle_idx, planar=planar, axes_2d=axes_2d)
            v[vehicle_idx, :] = self._apply_cbf_projection(v_nom[vehicle_idx, :], rho_m, grad_rho_m)

        self._set_vel_cmds(v)
        self._positions_prev_cbf = positions.copy()


    def get_vel_cmds(self) -> np.ndarray:
        """
        Get the latest velocity commands for all vehicles.

        Returns:
            np.ndarray: An array of velocity commands
        """
        return self._vel_cmds


class BatchedDootCbfCoordinator:
    """
    Full torch-native version:
      - targets stored as torch.Tensor
      - interpolation via torch kNN inverse-distance weighting
      - outputs vel_cmds as torch.Tensor (N,3)

    Intended wiring:
      dcc.step(t, states)
      v_cmds = dcc.get_vel_cmds()            # torch (N,3)
      batched_vr.set_vel_cmd(v_cmds)         # torch-native VF :contentReference[oaicite:3]{index=3}
    """

    _FIXED_SEED: int = 20220610

    def __init__(
        self,
        *,
        vehicles: List[Any],
        targeted_positions: torch.Tensor,   # (M,3) torch
        doot_config: DootConfig,
        velocity_max: Optional[float] = None,
        apply_cbf: bool = False,
        cbf_config: Optional[CbfConfig] = None,
        dtype: torch.dtype = torch.float64,
        device: Optional[torch.device] = None,
        idxs: Optional[List[int]] = None,
    ) -> None:
        self._vehicles: List[Any] = vehicles
        self._num_vehicles: int = len(vehicles)
        if self._num_vehicles < 1:
            raise ValueError("vehicles must be non-empty.")

        self._dtype = dtype
        self._device = device if device is not None else targeted_positions.device

        self._targeted_positions: torch.Tensor = targeted_positions.to(device=self._device, dtype=self._dtype)
        if self._targeted_positions.ndim != 2 or self._targeted_positions.shape[1] != 3:
            raise ValueError(f"targeted_positions must be shape (M,3), got {tuple(self._targeted_positions.shape)}")

        self._doot = doot_config
        self._velocity_max = velocity_max

        self._apply_cbf = bool(apply_cbf)
        self._cbf = cbf_config
        if self._apply_cbf and self._cbf is None:
            raise ValueError("apply_cbf=True requires cbf_config.")

        # internal state
        self._phi: torch.Tensor = torch.zeros((self._num_vehicles,), device=self._device, dtype=self._dtype)
        self._last_time: Optional[float] = None
        self._positions_prev_cbf: Optional[torch.Tensor] = None  # (N,3)

        self._vel_cmds: torch.Tensor = torch.zeros((self._num_vehicles, 3), device=self._device, dtype=self._dtype)

        seed = None if self._doot.use_random_sampling else self._FIXED_SEED
        self._rng = torch.Generator(device="cpu")
        if seed is not None:
            self._rng.manual_seed(int(seed))

        # trial move distribution
        if self._doot.mean_trial_move is None:
            self._mean = torch.zeros((3,), dtype=self._dtype, device="cpu")
        else:
            self._mean = torch.tensor(self._doot.mean_trial_move, dtype=self._dtype, device="cpu").reshape(3,)

        if self._doot.var_trial_move is None:
            self._var = torch.tensor([0.05, 0.05, 0.05], dtype=self._dtype, device="cpu")
        else:
            self._var = torch.tensor(self._doot.var_trial_move, dtype=self._dtype, device="cpu").reshape(3,)

        if torch.any(self._var < 0.0):
            raise ValueError("var_trial_move must be nonnegative.")

        # Store indices for this coordinator (subset of the full swarm)
        if idxs is None:
            raise ValueError(
                "idxs must be provided when using multiple coordinators (global indexing required)."
            )

        self._idxs = np.asarray(idxs, dtype=int).reshape(-1)

        if self._idxs.size < 1:
            raise ValueError("idxs must be non-empty.")
        if np.any(self._idxs < 0):
            raise ValueError("idxs must be nonnegative.")

        # KEY CHECK: subset sizes must match
        if self._idxs.size != len(vehicles):
            raise ValueError(
                f"idxs length ({self._idxs.size}) must match len(vehicles) ({len(vehicles)})."
            )

        # OPTIONAL (recommended): indices must be unique within this coordinator
        if np.unique(self._idxs).size != self._idxs.size:
            raise ValueError("idxs must not contain duplicates within a coordinator.")


    def get_indices(self) -> np.ndarray:
        return self._idxs.copy()


    def get_vel_cmds(self) -> torch.Tensor:
        return self._vel_cmds


    def _set_vel_cmds(self, v: torch.Tensor) -> None:
        v = v.to(device=self._device, dtype=self._dtype)
        if self._velocity_max is not None:
            vmax = float(self._velocity_max)
            speeds = torch.linalg.norm(v, dim=1)
            mask = speeds > vmax
            if torch.any(mask):
                scale = (vmax / speeds.clamp(min=1e-12))
                v = v * torch.where(mask, scale, torch.ones_like(scale)).unsqueeze(1)
        self._vel_cmds = v


    @staticmethod
    def _planar_axes(positions: torch.Tensor, planar_std_threshold: float) -> Tuple[bool, Optional[Tuple[int, int]], Optional[int]]:
        # positions: (N,3)
        std = torch.std(positions, dim=0)
        planar = bool(torch.any(std <= planar_std_threshold).item())
        if not planar:
            return False, None, None
        order = torch.argsort(std)  # small->large
        dropped = int(order[0].item())
        axes = (int(order[1].item()), int(order[2].item()))
        return True, axes, dropped


    def _knn_interp_phi(
        self,
        query: torch.Tensor,       # (Q,D)
        sites: torch.Tensor,       # (N,D)
        values: torch.Tensor,      # (N,)
        *,
        k: int,
        eps: float,
        p: float,
    ) -> torch.Tensor:
        # distances (Q,N)
        d2 = torch.cdist(query, sites, p=2.0)  # (Q,N)
        kk = min(int(k), int(sites.shape[0]))
        dist, idx = torch.topk(d2, k=kk, dim=1, largest=False)  # (Q,kk)
        v = values[idx]  # (Q,kk)

        w = 1.0 / (dist.clamp(min=eps) ** p)
        wsum = torch.sum(w, dim=1).clamp(min=eps)
        return torch.sum(w * v, dim=1) / wsum  # (Q,)

    def _scipy_interp_phi(
        self,
        query: torch.Tensor,   # (Q, D)
        sites: torch.Tensor,   # (N, D)
        values: torch.Tensor,  # (N,)
    ) -> torch.Tensor:
        """Scipy LinearNDInterpolator + NearestNDInterpolator fallback.

        Exactly replicates the interpolation used in DootCbfCoordinator:
          - Inside convex hull : piecewise-linear (Delaunay triangulation)
          - Outside convex hull: nearest-neighbor (no extrapolation artifacts)

        No gradients are needed through this call (phi interpolation is used
        only to select the best displacement via argmin), so the numpy
        round-trip is safe.
        """
        sites_np  = sites.detach().cpu().numpy()
        values_np = values.detach().cpu().numpy()
        query_np  = query.detach().cpu().numpy()

        phi_lin = LinearNDInterpolator(sites_np, values_np, fill_value=np.nan)
        phi_nn  = NearestNDInterpolator(sites_np, values_np)

        v = phi_lin(query_np).astype(np.float64)
        mask = ~np.isfinite(v)
        if np.any(mask):
            v[mask] = phi_nn(query_np[mask])

        return torch.as_tensor(v, device=query.device, dtype=query.dtype)  # (Q,)

    def _kde_rho_grad(
        self,
        positions_prev: torch.Tensor,  # (N,3)
        planar: bool,
        axes_2d: Optional[Tuple[int, int]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._cbf is None:
            raise RuntimeError("cbf_config is None.")

        bw = float(self._cbf.kde_bandwidth)
        R = float(self._cbf.interaction_radius)

        X = positions_prev  # (N,3)
        N = X.shape[0]

        if planar:
            if axes_2d is None:
                raise ValueError("planar=True requires axes_2d.")
            ax0, ax1 = int(axes_2d[0]), int(axes_2d[1])
            X2 = X[:, (ax0, ax1)]  # (N,2)

            diffs = X2.unsqueeze(0) - X2.unsqueeze(1)  # (N,N,2): diffs[i,j] = x_j - x_i
            d2 = torch.sum(diffs * diffs, dim=2)       # (N,N)
            d = torch.sqrt(d2)

            neigh = (d <= R) & (d >= 0.0)
            w = torch.exp(-d2 / (2.0 * bw * bw)) * neigh.to(X.dtype)

            # truncated Gaussian normalization over disk of radius R
            C = 2.0 * math.pi * bw**2 * (1.0 - math.exp(-R**2 / (2.0 * bw**2)))
            rho = torch.sum(w, dim=1) / (float(N) * C)

            grad2 = -(1.0 / (float(N) * C * (bw**2))) * torch.sum(diffs * w.unsqueeze(2), dim=1)
            grad = torch.zeros((N, 3), device=X.device, dtype=X.dtype)
            grad[:, ax0] = grad2[:, 0]
            grad[:, ax1] = grad2[:, 1]
            return rho, grad

        diffs = X.unsqueeze(0) - X.unsqueeze(1)  # (N,N,3): diffs[i,j] = x_j - x_i
        d2 = torch.sum(diffs * diffs, dim=2)     # (N,N)
        d = torch.sqrt(d2)

        neigh = (d <= R) & (d >= 0.0)
        w = torch.exp(-d2 / (2.0 * bw * bw)) * neigh.to(X.dtype)

        C = (2.0 * math.pi) ** 1.5 * (bw ** 3)
        rho = torch.sum(w, dim=1) / (float(N) * C)
        grad = -(1.0 / (float(N) * C * (bw**2))) * torch.sum(diffs * w.unsqueeze(2), dim=1)
        return rho, grad


    def _cbf_project(
        self,
        v_nom: torch.Tensor,  # (N,3)
        rho: torch.Tensor,    # (N,)
        grad: torch.Tensor,   # (N,3)
    ) -> torch.Tensor:
        if self._cbf is None:
            raise RuntimeError("cbf_config is None.")

        v = v_nom.clone()

        rho_dot_min = float(self._cbf.density_upper_gain) * (rho - float(self._cbf.density_upper_bound))
        rho_dot_max = float(self._cbf.density_lower_gain) * (rho - float(self._cbf.density_lower_bound))

        grad_norm = torch.linalg.norm(grad, dim=1)
        active = grad_norm > float(self._cbf.grad_norm_eps)

        grad_sq = torch.sum(grad * grad, dim=1).clamp(min=1e-12)
        rho_dot = torch.sum(grad * v, dim=1)

        need = active & (rho_dot < rho_dot_min)
        if torch.any(need):
            alpha = (rho_dot_min - rho_dot) / grad_sq
            v = v + (alpha.unsqueeze(1) * grad) * need.to(v.dtype).unsqueeze(1)
            rho_dot = torch.sum(grad * v, dim=1)

        need = active & (rho_dot > rho_dot_max)
        if torch.any(need):
            alpha = (rho_dot_max - rho_dot) / grad_sq
            v = v + (alpha.unsqueeze(1) * grad) * need.to(v.dtype).unsqueeze(1)

        # unit speed saturation (same spirit as your DCC CBF projection)
        n = torch.linalg.norm(v, dim=1).clamp(min=1e-12)
        mask = n > 1.0
        if torch.any(mask):
            v = torch.where(mask.unsqueeze(1), v / n.unsqueeze(1), v)
        return v


    def step(self, cur_time: float, states: List[Dict[str, Any]]) -> None:
        if len(states) != self._num_vehicles:
            raise ValueError(f"states must have length {self._num_vehicles}, got {len(states)}")

        if self._last_time is None:
            self._last_time = float(cur_time)
            self._set_vel_cmds(torch.zeros((self._num_vehicles, 3), device=self._device, dtype=self._dtype))
            return

        dt = float(cur_time) - float(self._last_time)
        self._last_time = float(cur_time)
        if (not math.isfinite(dt)) or dt <= 0.0:
            self._set_vel_cmds(torch.zeros((self._num_vehicles, 3), device=self._device, dtype=self._dtype))
            return

        # positions (N,3) torch
        pos = torch.zeros((self._num_vehicles, 3), device=self._device, dtype=self._dtype)
        for i, s in enumerate(states):
            pos[i, :] = torch.as_tensor(s["x"], device=self._device, dtype=self._dtype).reshape(3,)

        if self._positions_prev_cbf is None:
            self._positions_prev_cbf = pos.clone()

        planar, axes_2d, dropped_axis = self._planar_axes(pos, float(self._doot.planar_std_threshold))
        if planar:
            assert axes_2d is not None
            pos_eval = pos[:, axes_2d]                    # (N,2)
            targ_eval = self._targeted_positions[:, axes_2d]  # (M,2)
        else:
            pos_eval = pos
            targ_eval = self._targeted_positions

        # (1) count: for each target, nearest agent
        # d2: (M,N)
        d2 = torch.cdist(targ_eval, pos_eval, p=2.0)  # (M,N) distances
        nearest = torch.argmin(d2, dim=1)             # (M,)
        count = torch.bincount(nearest, minlength=self._num_vehicles).to(self._dtype)
        count = count / float(targ_eval.shape[0])

        # (2) Laplacian from undirected kNN graph
        n_neigh = int(self._doot.num_neighbors)
        if n_neigh < 2:
            raise ValueError("doot_config.num_neighbors must be >= 2")

        # pairwise distances between agents (N,N)
        dAA = torch.cdist(pos_eval, pos_eval, p=2.0)  # (N,N)
        # exclude self by setting diag to +inf, then take smallest
        inf = torch.tensor(float("inf"), device=self._device, dtype=self._dtype)
        dAA = dAA + torch.diag(inf.repeat(self._num_vehicles))
        k = min(self._num_vehicles - 1, n_neigh - 1)
        _, nn_idx = torch.topk(dAA, k=k, dim=1, largest=False)  # (N,k)

        A_dir = torch.zeros((self._num_vehicles, self._num_vehicles), device=self._device, dtype=self._dtype)
        rows = torch.arange(self._num_vehicles, device=self._device).repeat_interleave(k)
        cols = nn_idx.reshape(-1)
        A_dir[rows, cols] = 1.0

        W = 0.5 * (A_dir + A_dir.t())
        A = (W > 0.0).to(self._dtype)
        deg = torch.diag(torch.sum(A, dim=1))
        L = deg - A

        # (3) primal-dual iterations
        normalization = 1.0 / float(self._num_vehicles)
        gain = 1.0 / float(n_neigh + 1)

        phi = self._phi
        ones = torch.ones((self._num_vehicles,), device=self._device, dtype=self._dtype)
        for _ in range(int(self._doot.max_iter_primaldual)):
            phi = phi - gain * (L @ phi) + normalization * ones - count
        self._phi = phi

        # (4) trial moves (generate on CPU for RNG, then move to device)
        S = int(self._doot.num_trial_move_samples)
        disp_cpu = torch.normal(
            mean=self._mean.expand(S, 3),
            std=torch.sqrt(self._var).expand(S, 3),
            generator=self._rng,
        )  # (S,3) on CPU
        disp = disp_cpu.to(device=self._device, dtype=self._dtype)

        if planar and dropped_axis is not None:
            disp[:, dropped_axis] = 0.0

        norms = torch.linalg.norm(disp, dim=1)  # (S,)
        keep = norms >= float(self._doot.min_displacement_norm)
        if not torch.any(keep):
            self._set_vel_cmds(torch.zeros((self._num_vehicles, 3), device=self._device, dtype=self._dtype))
            self._positions_prev_cbf = pos.clone()
            return

        disp = disp[keep]
        norms = norms[keep]
        K = int(disp.shape[0])

        # (5) evaluate phi at all trials for all agents using torch kNN interpolant
        # build all trial positions: (N,K,3)
        pos_trial = pos.unsqueeze(1) + disp.unsqueeze(0)   # (N,K,3)
        if planar:
            assert axes_2d is not None
            query = pos_trial[:, :, axes_2d].reshape(-1, 2)   # (N*K,2)
            sites = pos_eval                                 # (N,2)
        else:
            query = pos_trial.reshape(-1, 3)                  # (N*K,3)
            sites = pos_eval                                  # (N,3)

        phi_q = self._scipy_interp_phi(
            query=query,
            sites=sites,
            values=self._phi,
        ).reshape(self._num_vehicles, K)  # (N,K)

        cost = norms.unsqueeze(0) + phi_q  # (N,K)
        idx = torch.argmin(cost, dim=1)    # (N,)

        best_disp = disp[idx]              # (N,3)
        v_nom = best_disp / float(dt)      # (N,3)

        # (6) optional CBF in torch
        if not self._apply_cbf:
            self._set_vel_cmds(v_nom)
            self._positions_prev_cbf = pos.clone()
            return

        rho, grad = self._kde_rho_grad(self._positions_prev_cbf, planar=planar, axes_2d=axes_2d)
        v = self._cbf_project(v_nom, rho, grad)

        self._set_vel_cmds(v)
        self._positions_prev_cbf = pos.clone()
