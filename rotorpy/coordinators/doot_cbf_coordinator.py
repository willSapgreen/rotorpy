import numpy as np
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Dict, Any, Tuple
import math
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator


@dataclass(frozen=True)
class DootConfig:
    """
    Static DOOT configuration (no runtime state).
    """
    num_neighbors: int                 # kNN neighbors excluding self
    max_iter_primaldual: int           # inner primal-dual iterations per outer step

    use_random_sampling: bool = True
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
    Configuration for density-based Control Barrier Function (CBF),
    translated from the MATLAB reference implementation.

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

    # Radius actually used for neighbor selection (MATLAB uses Rh_bar)
    interaction_radius: float = field(init=False)

    def __post_init__(self):
        bw = float(self.kde_bandwidth)

        if bw <= 0.0:
            raise ValueError("kde_bandwidth must be positive.")

        # MATLAB formulas:
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
    per-vehicle callables v_cmd_fn(t) -> R^3.

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

        # Target samples (M,3)
        self.targeted_positions = np.asarray(targeted_positions, dtype=float)
        if self.targeted_positions.ndim != 2 or self.targeted_positions.shape[1] != 3:
            raise ValueError(
                f"targeted_positions must have shape (M,3), got {self.targeted_positions.shape}"
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
        self.rng: Optional[np.random.Generator] = None
        if self.doot_config.use_random_sampling:
            self.rng = np.random.default_rng(self._FIXED_SEED)

        # Command storage
        self._v_cmd = np.zeros((self.num_vehicles, 3), dtype=float)
        self._v_cmd_fns = [self._make_v_cmd_fn(i) for i in range(self.num_vehicles)]

        # CBF config
        self.apply_cbf = bool(apply_cbf)
        self.cbf_config = cbf_config
        if self.apply_cbf and self.cbf_config is None:
            raise ValueError("apply_cbf=True requires a valid CbfConfig.")

        # CBF runtime state (previous positions only)
        self._X_prev_cbf: Optional[np.ndarray] = None


    def _make_v_cmd_fn(self, i: int) -> Callable[[float], np.ndarray]:
        def v_cmd_fn(t: float) -> np.ndarray:
            return self._v_cmd[i]
        return v_cmd_fn

    def get_v_cmd_fns(self) -> List[Callable[[float], np.ndarray]]:
        return self._v_cmd_fns

    def set_v_cmd_batch(self, V: np.ndarray) -> None:
        V = np.asarray(V, dtype=float)
        if V.shape != (self.num_vehicles, 3):
            raise ValueError(f"V must have shape {(self.num_vehicles, 3)}, got {V.shape}")

        if self.velocity_max is not None:
            speeds = np.linalg.norm(V, axis=1)
            mask = speeds > self.velocity_max
            if np.any(mask):
                V[mask] *= (self.velocity_max / speeds[mask])[:, None]

        self._v_cmd[:, :] = V

    def _compute_kde_density_and_gradient(
    self,
    X_prev: np.ndarray,
    m: int,
    *,
    planar: bool,
    axes_2d: Optional[Tuple[int, int]],
    ) -> Tuple[float, np.ndarray]:
        """
        KDE rho and grad_rho at agent m using previous positions, consistent with DOOT plane.

        If planar=True:
        - compute KDE in the DOOT plane defined by axes_2d (two indices in {0,1,2})
        - return a (3,) gradient with nonzeros only on axes_2d

        If planar=False:
        - compute KDE in 3D and return full (3,) gradient
        """
        if self.cbf_config is None:
            raise RuntimeError("_compute_kde_density_and_gradient called but cbf_config is None.")

        X_prev = np.asarray(X_prev, dtype=float)
        if X_prev.shape != (self.num_vehicles, 3):
            raise ValueError(f"X_prev must have shape {(self.num_vehicles, 3)}, got {X_prev.shape}")
        if not (0 <= m < self.num_vehicles):
            raise IndexError(f"m out of range: {m}")

        cfg = self.cbf_config
        bw = float(cfg.kde_bandwidth)
        R = float(cfg.interaction_radius)

        if planar:
            if axes_2d is None or len(axes_2d) != 2:
                raise ValueError("planar=True requires axes_2d=(i,j).")

            # --- work in the selected 2D plane ---
            Xp = X_prev[:, axes_2d]                      # (N,2)
            xm = X_prev[m, axes_2d]                      # (2,)
            diffs = Xp - xm                              # (N,2)
            dists = np.linalg.norm(diffs, axis=1)        # (N,)

            neigh_mask = (dists <= R) & (dists > 0.0)

            # MATLAB truncated-2D normalization
            C = 2.0 * np.pi * (bw ** 2) * (1.0 - np.exp(-0.5))

            if not np.any(neigh_mask):
                rho_m = 0.0
                grad_2d = np.zeros((2,), dtype=float)
            else:
                diffs_k = diffs[neigh_mask]                          # (K,2)
                dists_k = dists[neigh_mask]                          # (K,)
                w = np.exp(-(dists_k ** 2) / (2.0 * (bw ** 2)))      # (K,)

                rho_m = float(np.sum(w) / (float(self.num_vehicles) * C))

                grad_2d = -(1.0 / (float(self.num_vehicles) * C * (bw ** 2))) * np.sum(
                    diffs_k * w[:, None], axis=0
                )

            grad_3d = np.zeros((3,), dtype=float)
            grad_3d[axes_2d[0]] = grad_2d[0]
            grad_3d[axes_2d[1]] = grad_2d[1]
            return rho_m, grad_3d

        else:
            # --- 3D KDE (extension beyond MATLAB 2D) ---
            diffs = X_prev - X_prev[m, :]                 # (N,3)
            dists = np.linalg.norm(diffs, axis=1)         # (N,)
            neigh_mask = (dists <= R) & (dists > 0.0)

            # 3D Gaussian normalization (not truncated like MATLAB 2D)
            C = (2.0 * np.pi) ** (1.5) * (bw ** 3)

            if not np.any(neigh_mask):
                rho_m = 0.0
                grad_3d = np.zeros((3,), dtype=float)
            else:
                diffs_k = diffs[neigh_mask]                          # (K,3)
                dists_k = dists[neigh_mask]                          # (K,)
                w = np.exp(-(dists_k ** 2) / (2.0 * (bw ** 2)))      # (K,)

                rho_m = float(np.sum(w) / (float(self.num_vehicles) * C))

                grad_3d = -(1.0 / (float(self.num_vehicles) * C * (bw ** 2))) * np.sum(
                    diffs_k * w[:, None], axis=0
                )

            return rho_m, grad_3d

    def _apply_cbf_projection(
        self,
        v_nom_m: np.ndarray,
        rho_m: float,
        grad_rho_m: np.ndarray,
    ) -> np.ndarray:
        """
        MATLAB-equivalent sequential projection enforcing:
        rho_dot >= alpha1 * (rho - epsilon)
        rho_dot <= alpha2 * (rho - epsilon_min)
        where rho_dot := grad_rho' * v.

        Also applies MATLAB's post-cap: if ||v|| > 1 then v <- v/||v||.
        """
        if self.cbf_config is None:
            raise RuntimeError("_apply_cbf_projection called but cbf_config is None.")

        cfg = self.cbf_config

        v = np.asarray(v_nom_m, dtype=float).reshape(3,).copy()
        a = np.asarray(grad_rho_m, dtype=float).reshape(3,)

        # Bounds on rho_dot := a' * v
        rho_dot_min = float(cfg.density_upper_gain) * (float(rho_m) - float(cfg.density_upper_bound))
        rho_dot_max = float(cfg.density_lower_gain) * (float(rho_m) - float(cfg.density_lower_bound))

        a_norm = float(np.linalg.norm(a))
        if a_norm > float(cfg.grad_norm_eps):
            aa = float(a @ a)          # ||a||^2
            adv = float(a @ v)         # rho_dot

            # Enforce rho_dot >= rho_dot_min
            if adv < rho_dot_min:
                v = v + ((rho_dot_min - adv) / aa) * a
                adv = rho_dot_min

            # Enforce rho_dot <= rho_dot_max
            if adv > rho_dot_max:
                v = v + ((rho_dot_max - adv) / aa) * a

        # MATLAB post-cap to unit speed
        v_norm = float(np.linalg.norm(v))
        if v_norm > 1.0:
            v = v * (1.0 / v_norm)

        return v

    def step(self, t: float, states: List[Dict[str, Any]]) -> None:
        """
        DOOT step (CBF optional via self.apply_cbf).

        Inputs:
        t: global simulation time (seconds)
        states: list of length self.num_vehicles, only for member vehicles; each must include 'x' (3,)

        Behavior:
        - compute dt = t - last_time
        - update phi via primal-dual iterations
        - build interpolant PHI at current agent positions
        - choose trial move per agent: argmin(||d|| + PHI(x+d)) with ||d|| >= min_displacement_norm
        - convert displacement to velocity v = d/dt
        - optionally apply CBF filter to v_nom
        - store v in _v_cmd
        """
        if len(states) != self.num_vehicles:
            raise ValueError(f"states must have length num_vehicles={self.num_vehicles}, got {len(states)}")

        # Initialize time on first call
        if self.last_time is None:
            self.last_time = float(t)
            self.set_v_cmd_batch(np.zeros((self.num_vehicles, 3), dtype=float))
            return

        dt = float(t) - float(self.last_time)
        self.last_time = float(t)

        if (not np.isfinite(dt)) or dt <= 0.0:
            self.set_v_cmd_batch(np.zeros((self.num_vehicles, 3), dtype=float))
            return

        # Extract positions X (num_vehicles,3)
        X = np.zeros((self.num_vehicles, 3), dtype=float)
        for i, s in enumerate(states):
            if "x" not in s:
                raise KeyError("Each state must contain key 'x' with shape (3,).")
            X[i, :] = np.asarray(s["x"], dtype=float).reshape(3,)

        # CBF runtime state init (previous positions)
        if self._X_prev_cbf is None:
            self._X_prev_cbf = X.copy()

        # ------------------------------------------------------------
        # Planar detection (Option A): axis-aligned std check.
        # If planar, choose the 2 axes with the largest std and drop the smallest.
        # TODO (Option B): use SVD/rank-based plane detection for rotated planes.
        # ------------------------------------------------------------
        std_xyz = np.std(X, axis=0)  # (3,)
        planar = bool(np.any(std_xyz <= float(self.doot_config.planar_std_threshold)))

        if planar:
            axes_2d = tuple(np.argsort(std_xyz)[-2:].tolist())  # e.g. (0,2) for XZ plane
            dropped_axis = int(np.argsort(std_xyz)[0])
            Xp = X[:, axes_2d]                           # (num_vehicles,2)
            Tp = self.targeted_positions[:, axes_2d]     # (M,2)
        else:
            axes_2d = None
            dropped_axis = None
            Xp = X                                       # (num_vehicles,3)
            Tp = self.targeted_positions                 # (M,3)

        # ------------------------------------------------------------
        # 1) count: assign each target sample to nearest agent
        # ------------------------------------------------------------
        M = int(self.targeted_positions.shape[0])
        diff_ta = Tp[:, None, :] - Xp[None, :, :]        # (M,num_vehicles,dim)
        d2_ta = np.sum(diff_ta * diff_ta, axis=2)        # (M,num_vehicles)
        nearest_agent = np.argmin(d2_ta, axis=1)         # (M,)
        count = np.bincount(nearest_agent, minlength=self.num_vehicles).astype(float) / float(M)

        # ------------------------------------------------------------
        # 2) Neighbor graph + Laplacian from agent kNN (excluding self)
        # ------------------------------------------------------------
        k = int(self.doot_config.num_neighbors)
        if k >= self.num_vehicles:
            # Excluding self, max meaningful neighbors is num_vehicles-1
            k = self.num_vehicles - 1

        diff_aa = Xp[:, None, :] - Xp[None, :, :]        # (num_vehicles,num_vehicles,dim)
        d2_aa = np.sum(diff_aa * diff_aa, axis=2)        # (num_vehicles,num_vehicles)
        np.fill_diagonal(d2_aa, np.inf)

        # kth index safety when num_vehicles is small
        kth = min(k - 1, self.num_vehicles - 2) if self.num_vehicles >= 2 else 0
        nn_idx = np.argpartition(d2_aa, kth=kth, axis=1)[:, :k]  # (num_vehicles,k)

        A = np.zeros((self.num_vehicles, self.num_vehicles), dtype=float)
        rows = np.repeat(np.arange(self.num_vehicles), k)
        cols = nn_idx.reshape(-1)
        A[rows, cols] = 1.0

        W = 0.5 * (A + A.T)
        D = np.diag(np.sum(W, axis=1))
        L = D - W

        # ------------------------------------------------------------
        # 3) Primal–dual inner iterations updating phi
        # ------------------------------------------------------------
        one_over_N = 1.0 / float(self.num_vehicles)
        gain = 1.0 / float(k + 1)

        phi = self.phi
        for _ in range(int(self.doot_config.max_iter_primaldual)):
            phi = phi - gain * (L @ phi) + one_over_N * np.ones((self.num_vehicles,), dtype=float) - count
        self.phi = phi

        # ------------------------------------------------------------
        # 4) Scattered interpolant PHI in dim=2 (planar) or dim=3 (full)
        # ------------------------------------------------------------
        phi_lin = LinearNDInterpolator(Xp, self.phi, fill_value=np.nan)
        phi_nn = NearestNDInterpolator(Xp, self.phi)

        def PHI(P: np.ndarray) -> np.ndarray:
            v = phi_lin(P)
            v = np.asarray(v, dtype=float)
            mask = ~np.isfinite(v)
            if np.any(mask):
                v[mask] = phi_nn(P[mask])
            return v

        # ------------------------------------------------------------
        # 5) Transport step via trial moves (nominal)
        # ------------------------------------------------------------
        v_nom = np.zeros((self.num_vehicles, 3), dtype=float)

        if not self.doot_config.use_random_sampling:
            self.set_v_cmd_batch(v_nom)
            self._X_prev_cbf = X.copy()
            return

        if self.rng is None:
            raise RuntimeError("use_random_sampling=True but RNG is not initialized.")

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
            self.set_v_cmd_batch(np.zeros((self.num_vehicles, 3), dtype=float))
            self._X_prev_cbf = X.copy()
            return

        for m in range(self.num_vehicles):
            pos_trial = X[m, :][None, :] + disp_kept  # (K,3)

            if planar:
                pos_eval = pos_trial[:, axes_2d]      # (K,2)
            else:
                pos_eval = pos_trial                  # (K,3)

            cost = norms_kept + PHI(pos_eval)
            ind = int(np.argmin(cost))
            v_disp = pos_trial[ind, :] - X[m, :]
            v_nom[m, :] = v_disp / dt

        # ------------------------------------------------------------
        # 6) Optional CBF filtering
        # ------------------------------------------------------------
        if not self.apply_cbf:
            self.set_v_cmd_batch(v_nom)
            self._X_prev_cbf = X.copy()
            return

        if self.cbf_config is None:
            raise RuntimeError("apply_cbf=True but cbf_config is None.")

        v = v_nom.copy()
        X_prev = self._X_prev_cbf

        for m in range(self.num_vehicles):
            rho_m, grad_rho_m = self._compute_kde_density_and_gradient(
                X_prev, m, planar=planar, axes_2d=axes_2d)
            v[m, :] = self._apply_cbf_projection(v_nom[m, :], rho_m, grad_rho_m)

        self.set_v_cmd_batch(v)
        self._X_prev_cbf = X.copy()


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
        use_random_sampling=True,
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

    # Initial XY (use initial states so scatter has N points)
    XY0 = np.array([cur_x[i]["x"][:2] for i in range(num_vehicles)], dtype=float)

    # Random per-vehicle colors (different each run is OK)
    colors = np.random.rand(num_vehicles, 3)  # RGB in [0,1]

    scat = ax.scatter(XY0[:, 0], XY0[:, 1], s=20, c=colors)

    fig.canvas.draw()

    # --- OpenCV VideoWriter (MATLAB-style) ---
    width, height = fig.canvas.get_width_height()
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    video = cv2.VideoWriter(avi_path, fourcc, 10, (width, height))

    for step in range(intervals):
        coordinator.step(cur_time, cur_x)
        v_cmd_fns = coordinator.get_v_cmd_fns()

        for i in range(num_vehicles):
            v = np.asarray(v_cmd_fns[i](cur_time), dtype=float).reshape(3,)
            cur_x[i]["x"] += v * dt

        XY = np.array([cur_x[i]["x"][:2] for i in range(num_vehicles)], dtype=float)
        scat.set_offsets(XY)
        ax.set_title(f"t = {cur_time:.2f} s")

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


# def unit_test_02():
#     import copy
#     import os

#     import numpy as np
#     import matplotlib.pyplot as plt
#     import cv2

#     from scipy.stats import multivariate_normal

#     # Initialize the "world"
#     num_vehicles = 200
#     vehicles = [None] * num_vehicles

#     # --- MATLAB-matching parameters ---
#     num_targets = 500
#     components = 20
#     dom_size = 5  # for the same grid construction as MATLAB

#     sim_time = 200
#     dt = 1
#     intervals = int(sim_time / dt)

#     # RNG (different each run is OK)
#     rng = np.random.default_rng()

#     # ============================================================
#     # Target distribution µ*
#     # ============================================================
#     mean_des = 8.0 * (rng.random((components, 2)) - 0.5)  # in [-4,4] for each component center
#     var_des = np.array([0.25, 0.25], dtype=float)
#     cov_des = np.diag(var_des)

#     comp_ids = rng.integers(low=0, high=components, size=num_targets)
#     samples_xy = np.empty((num_targets, 2), dtype=float)
#     for k in range(components):
#         idx = np.where(comp_ids == k)[0]
#         if idx.size == 0:
#             continue
#         samples_xy[idx, :] = rng.multivariate_normal(mean=mean_des[k], cov=cov_des, size=idx.size)

#     # targeted_positions expected as 3D points in your coordinator interface
#     targeted_positions = np.hstack([samples_xy, np.zeros((num_targets, 1), dtype=float)]).tolist()

#     # Optional: density map Z for visualization parity with MATLAB (not used in control)
#     x = np.arange(-dom_size, dom_size + dom_size / 100.0, dom_size / 100.0)
#     y = np.arange(-dom_size, dom_size + dom_size / 100.0, dom_size / 100.0)
#     Xg, Yg = np.meshgrid(x, y)
#     grid_xy = np.column_stack([Xg.ravel(), Yg.ravel()])

#     Z = np.zeros(grid_xy.shape[0], dtype=float)
#     for k in range(components):
#         Z += multivariate_normal(mean=mean_des[k], cov=cov_des).pdf(grid_xy)
#     Z /= components
#     Z = Z.reshape(Xg.shape)

#     # ============================================================
#     # Initial agent distribution
#     # ============================================================
#     mean_init = np.array([5.0, 5.0], dtype=float)
#     var_init = np.array([2.0, 6.0], dtype=float)
#     cov_init = np.diag(var_init)

#     pos_init_xy = rng.multivariate_normal(mean=mean_init, cov=cov_init, size=num_vehicles)

#     x0s = []
#     for i in range(num_vehicles):
#         x0s.append(
#             {
#                 "x": np.array([pos_init_xy[i, 0], pos_init_xy[i, 1], 0.0], dtype=float),
#                 "v": np.zeros(3),
#                 "q": np.array([0.0, 0.0, 0.0, 1.0]),
#                 "w": np.zeros(3),
#                 "wind": np.zeros(3),
#                 "rotor_speeds": np.full(4, 1788.53),
#             }
#         )

#     doot_config = DootConfig(
#         num_neighbors=min(10, num_vehicles - 1),  # MATLAB n_neigh = 10
#         max_iter_primaldual=10,
#         use_random_sampling=True,
#         num_trial_move_samples=300,               # MATLAB n_trial_samples = 300
#         min_displacement_norm=0.1,                # MATLAB d>=0.1
#         planar_std_threshold=0.1,
#     )

#     # --- Nominal coordinator (no CBF) ---
#     coordinator_nom = DootCbfCoordinator(
#         vehicles=vehicles,
#         velocity_max=1.0,
#         targeted_positions=targeted_positions,
#         doot_config=doot_config,
#         apply_cbf=False,
#         cbf_config=None,
#     )

#     # --- CBF coordinator ---
#     cbf_config = CbfConfig(
#         density_upper_gain=1.0,
#         density_lower_gain=1.0,
#         density_upper_bound=0.045,   # epsilon
#         density_lower_bound=0.01,    # epsilon_min
#         kde_bandwidth=0.3,           # bw
#         grad_norm_eps=1e-6,
#     )

#     coordinator_cbf = DootCbfCoordinator(
#         vehicles=vehicles,
#         velocity_max=1.0,
#         targeted_positions=targeted_positions,
#         doot_config=doot_config,
#         apply_cbf=True,
#         cbf_config=cbf_config,
#     )

#     def compute_rho_all_xy(pos_xy: np.ndarray, cfg: CbfConfig) -> np.ndarray:
#         """
#         KDE density rho at each agent position, using the same radius-truncated Gaussian as MATLAB.
#         pos_xy: (N,2)
#         returns rho: (N,)
#         """
#         N = pos_xy.shape[0]
#         bw = float(cfg.kde_bandwidth)
#         R = float(cfg.interaction_radius)

#         # Normalization constant for truncated Gaussian in 2D (MATLAB)
#         C = 2.0 * np.pi * (bw ** 2) * (1.0 - np.exp(-0.5))

#         diffs = pos_xy[:, None, :] - pos_xy[None, :, :]          # (N,N,2)
#         dists = np.linalg.norm(diffs, axis=2)                    # (N,N)

#         # exclude self and apply radius cutoff
#         mask = (dists > 0.0) & (dists <= R)

#         w = np.exp(-(dists ** 2) / (2.0 * (bw ** 2)))            # (N,N)
#         w *= mask

#         rho = np.sum(w, axis=1) / (N * C)
#         return rho

#     # Output path: same directory as this file
#     out_dir = os.path.dirname(os.path.abspath(__file__))
#     avi_path = os.path.join(out_dir, "doot_cbf_coordinator_ut02_compare.avi")

#     cur_time = 0.0
#     cur_x_nom = copy.deepcopy(x0s)
#     cur_x_cbf = copy.deepcopy(x0s)

#     # --- Matplotlib side-by-side plot setup ---
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
#     for ax in (ax1, ax2):
#         ax.set_aspect("equal")
#         ax.grid(True)
#         ax.set_xlabel("x")
#         ax.set_ylabel("y")
#         ax.scatter(samples_xy[:, 0], samples_xy[:, 1], s=8, c="k", alpha=0.15, linewidths=0)

#     ax1.set_title("Nominal (no CBF)")
#     ax2.set_title("With CBF")

#     XY0 = np.array([cur_x_nom[i]["x"][:2] for i in range(num_vehicles)], dtype=float)
#     colors = rng.random((num_vehicles, 3))  # same colors for both panels

#     scat1 = ax1.scatter(XY0[:, 0], XY0[:, 1], s=20, c=colors)
#     scat2 = ax2.scatter(XY0[:, 0], XY0[:, 1], s=20, c=colors)

#     # --- overlays: crowded (red ring) + sparse (yellow square) ---
#     over_hi1 = ax1.scatter([], [], s=90, facecolors="none", edgecolors="r", linewidths=1.8)
#     over_hi2 = ax2.scatter([], [], s=90, facecolors="none", edgecolors="r", linewidths=1.8)

#     under_lo1 = ax1.scatter([], [], s=120, marker="s", c=[[1.0, 0.8, 0.2]], edgecolors="none", alpha=0.9)
#     under_lo2 = ax2.scatter([], [], s=120, marker="s", c=[[1.0, 0.8, 0.2]], edgecolors="none", alpha=0.9)

#     fig.canvas.draw()

#     # --- OpenCV VideoWriter ---
#     width, height = fig.canvas.get_width_height()
#     fourcc = cv2.VideoWriter_fourcc(*"XVID")
#     video = cv2.VideoWriter(avi_path, fourcc, 10, (width, height))

#     for _ in range(intervals):
#         # step + integrate NOMINAL
#         coordinator_nom.step(cur_time, cur_x_nom)
#         v_cmd_fns_nom = coordinator_nom.get_v_cmd_fns()
#         for i in range(num_vehicles):
#             v = np.asarray(v_cmd_fns_nom[i](cur_time), dtype=float).reshape(3,)
#             v[2] = 0.0
#             cur_x_nom[i]["x"] += v * dt
#             cur_x_nom[i]["x"][2] = 0.0

#         # step + integrate CBF
#         coordinator_cbf.step(cur_time, cur_x_cbf)
#         v_cmd_fns_cbf = coordinator_cbf.get_v_cmd_fns()
#         for i in range(num_vehicles):
#             v = np.asarray(v_cmd_fns_cbf[i](cur_time), dtype=float).reshape(3,)
#             v[2] = 0.0
#             cur_x_cbf[i]["x"] += v * dt
#             cur_x_cbf[i]["x"][2] = 0.0

#         # update plots
#         XY_nom = np.array([cur_x_nom[i]["x"][:2] for i in range(num_vehicles)], dtype=float)
#         XY_cbf = np.array([cur_x_cbf[i]["x"][:2] for i in range(num_vehicles)], dtype=float)

#         scat1.set_offsets(XY_nom)
#         scat2.set_offsets(XY_cbf)

#         # --- overlays from KDE rho at agent positions ---
#         rho_nom = compute_rho_all_xy(XY_nom, cbf_config)
#         rho_cbf = compute_rho_all_xy(XY_cbf, cbf_config)

#         hi_nom = rho_nom >= cbf_config.density_upper_bound
#         hi_cbf = rho_cbf >= cbf_config.density_upper_bound

#         lo_nom = rho_nom < cbf_config.density_lower_bound
#         lo_cbf = rho_cbf < cbf_config.density_lower_bound

#         under_lo1.set_offsets(XY_nom[lo_nom])
#         under_lo2.set_offsets(XY_cbf[lo_cbf])

#         over_hi1.set_offsets(XY_nom[hi_nom])
#         over_hi2.set_offsets(XY_cbf[hi_cbf])

#         ax1.set_title(f"Nominal (no CBF)  t = {cur_time:.2f} s")
#         ax2.set_title(f"With CBF          t = {cur_time:.2f} s")

#         fig.canvas.draw()

#         # ARGB -> BGR for OpenCV
#         buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
#         buf = buf.reshape(height, width, 4)
#         frame = buf[:, :, [3, 2, 1]]
#         video.write(frame)

#         cur_time += dt

#     video.release()
#     plt.close(fig)

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
    def _detect_planar_axes(X: np.ndarray, planar_std_threshold: float):
        std_xyz = np.std(X, axis=0)  # (3,)
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

    rng = np.random.default_rng()

    # =========================
    # Target distribution (MATLAB equivalent)
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
        max_iter_primaldual=10,
        use_random_sampling=True,
        num_trial_move_samples=300,
        min_displacement_norm=0.1,
        planar_std_threshold=0.1,
    )

    cbf_config = CbfConfig(
        density_upper_gain=1.0,
        density_lower_gain=1.0,
        density_upper_bound=0.045,
        density_lower_bound=0.01,
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

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 6))
    for ax in (axL, axR):
        ax.set_aspect("equal")
        ax.grid(True)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.scatter(samples_xy[:, 0], samples_xy[:, 1], s=8, c="k", alpha=0.15, linewidths=0)

    colors = rng.random((num_vehicles, 3))
    XY0 = np.array([cur_nom[i]["x"][:2] for i in range(num_vehicles)], dtype=float)

    # Agent scatters
    scatL = axL.scatter(XY0[:, 0], XY0[:, 1], s=20, c=colors)
    scatR = axR.scatter(XY0[:, 0], XY0[:, 1], s=20, c=colors)

    # MATLAB-style markers:
    # - yellow squares: sparse (rho < epsilon_min)
    # - red circles (open): crowded / unsafe (rho >= epsilon)
    sparseL = axL.scatter(
        [], [], s=120, marker="s",
        facecolors=(1.0, 0.8, 0.2),
        edgecolors=(1.0, 0.6, 0.0),
        linewidths=1.5,
        zorder=3,
    )
    sparseR = axR.scatter(
        [], [], s=120, marker="s",
        facecolors=(1.0, 0.8, 0.2),
        edgecolors=(1.0, 0.6, 0.0),
        linewidths=1.5,
        zorder=3,
    )

    crowdL = axL.scatter(
        [], [], s=140, marker="o",
        facecolors="none", edgecolors="r",
        linewidths=1.5,
        zorder=4,
    )
    crowdR = axR.scatter(
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

    # Initialize previous positions (used for KDE, MATLAB parity)
    X_prev_nom = np.array([cur_nom[i]["x"] for i in range(num_vehicles)], dtype=float)
    X_prev_cbf = np.array([cur_cbf[i]["x"] for i in range(num_vehicles)], dtype=float)

    for step in range(intervals):
        # Previous positions for this step (MATLAB uses pos_prev_* for KDE)
        X_prev_nom = np.array([cur_nom[i]["x"] for i in range(num_vehicles)], dtype=float)
        X_prev_cbf = np.array([cur_cbf[i]["x"] for i in range(num_vehicles)], dtype=float)

        # Planar detection for KDE, consistent with the same rule used in coordinator.step
        planar_nom, axes_2d_nom = _detect_planar_axes(X_prev_nom, doot_config.planar_std_threshold)
        planar_cbf, axes_2d_cbf = _detect_planar_axes(X_prev_cbf, doot_config.planar_std_threshold)

        # ---- DOOT (nominal) computed once per step ----
        coordinator.step(cur_time, cur_nom)
        v_nom = np.array([coordinator.get_v_cmd_fns()[i](cur_time) for i in range(num_vehicles)], dtype=float)
        v_nom[:, 2] = 0.0  # enforce 2D for the unit test

        # ---- CBF projection branch: v_cbf = projection(v_nom) using X_prev_cbf ----
        v_cbf = v_nom.copy()
        for m in range(num_vehicles):
            rho_m, grad_rho_m = coordinator._compute_kde_density_and_gradient(
                X_prev_cbf, m, planar=planar_cbf, axes_2d=axes_2d_cbf
            )
            v_cbf[m, :] = coordinator._apply_cbf_projection(v_nom[m, :], rho_m, grad_rho_m)
        v_cbf[:, 2] = 0.0

        # ---- Integrate positions (two separate worlds) ----
        for i in range(num_vehicles):
            cur_nom[i]["x"] += dt * v_nom[i]
            cur_cbf[i]["x"] += dt * v_cbf[i]
            cur_nom[i]["x"][2] = 0.0
            cur_cbf[i]["x"][2] = 0.0

        XY_nom = np.array([cur_nom[i]["x"][:2] for i in range(num_vehicles)], dtype=float)
        XY_cbf = np.array([cur_cbf[i]["x"][:2] for i in range(num_vehicles)], dtype=float)

        scatL.set_offsets(XY_nom)
        scatR.set_offsets(XY_cbf)

        # ---- KDE-based overlays (computed from X_prev_*) ----
        rho_nom = np.array(
            [
                coordinator._compute_kde_density_and_gradient(
                    X_prev_nom, m, planar=planar_nom, axes_2d=axes_2d_nom
                )[0]
                for m in range(num_vehicles)
            ],
            dtype=float,
        )
        rho_cbf = np.array(
            [
                coordinator._compute_kde_density_and_gradient(
                    X_prev_cbf, m, planar=planar_cbf, axes_2d=axes_2d_cbf
                )[0]
                for m in range(num_vehicles)
            ],
            dtype=float,
        )

        sparseL.set_offsets(XY_nom[rho_nom < eps_min])
        sparseR.set_offsets(XY_cbf[rho_cbf < eps_min])

        crowdL.set_offsets(XY_nom[rho_nom >= eps])
        crowdR.set_offsets(XY_cbf[rho_cbf >= eps])

        axL.set_title(f"Nominal (no CBF)   t = {cur_time:.2f} s")
        axR.set_title(f"With CBF          t = {cur_time:.2f} s")

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
