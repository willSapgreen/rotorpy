import numpy as np
from dataclasses import dataclass
from typing import Callable, List, Optional, Dict, Any

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

    def step(self, t: float, states: List[Dict[str, Any]]) -> None:
        """
        DOOT step (no CBF yet).

        Inputs:
          t: global simulation time (seconds)
          states: list of length self.num_vehicles, only for member vehicles; each must include 'x' (3,)

        Behavior:
          - compute dt = t - last_time
          - update phi via primal-dual iterations
          - build interpolant PHI at current agent positions
          - choose trial move per agent: argmin(||d|| + PHI(x+d)) with ||d|| >= min_displacement_norm
          - convert displacement to velocity v = d/dt
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
        # 5) Transport step via trial moves
        # ------------------------------------------------------------
        V_nom = np.zeros((self.num_vehicles, 3), dtype=float)

        if not self.doot_config.use_random_sampling:
            self.set_v_cmd_batch(V_nom)
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
            V_nom[m, :] = v_disp / dt

        self.set_v_cmd_batch(V_nom)


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


def unit_test_02():
    import copy
    import os

    import numpy as np
    import matplotlib.pyplot as plt
    import cv2

    from scipy.stats import multivariate_normal

    # Initialize the "world"
    num_vehicles = 200
    vehicles = [None] * num_vehicles

    # --- MATLAB-matching parameters ---
    num_targets = 500
    components = 20
    dom_size = 5  # for the same grid construction as MATLAB

    sim_time = 200
    dt = 1
    intervals = int(sim_time / dt)

    # RNG (different each run is OK)
    rng = np.random.default_rng()

    # ============================================================
    # Target distribution µ* (MATLAB: gm = gmdistribution(mean_des, var_des); samples = random(gm,num_targets))
    # ============================================================
    mean_des = 8.0 * (rng.random((components, 2)) - 0.5)  # in [-4,4] for each component center
    var_des = np.array([0.25, 0.25], dtype=float)
    cov_des = np.diag(var_des)

    # MATLAB gmdistribution default: equal component weights
    comp_ids = rng.integers(low=0, high=components, size=num_targets)
    samples_xy = np.empty((num_targets, 2), dtype=float)
    for k in range(components):
        idx = np.where(comp_ids == k)[0]
        if idx.size == 0:
            continue
        samples_xy[idx, :] = rng.multivariate_normal(mean=mean_des[k], cov=cov_des, size=idx.size)

    # targeted_positions expected as 3D points in your coordinator interface
    targeted_positions = np.hstack([samples_xy, np.zeros((num_targets, 1), dtype=float)]).tolist()

    # Optional: density map Z for visualization parity with MATLAB (not used in control)
    x = np.arange(-dom_size, dom_size + dom_size / 100.0, dom_size / 100.0)
    y = np.arange(-dom_size, dom_size + dom_size / 100.0, dom_size / 100.0)
    X, Y = np.meshgrid(x, y)
    grid_xy = np.column_stack([X.ravel(), Y.ravel()])

    # Mixture PDF: average of component PDFs (equal weights)
    Z = np.zeros(grid_xy.shape[0], dtype=float)
    for k in range(components):
        Z += multivariate_normal(mean=mean_des[k], cov=cov_des).pdf(grid_xy)
    Z /= components
    Z = Z.reshape(X.shape)

    # ============================================================
    # Initial agent distribution (MATLAB: gm_init = gmdistribution(mean_init, var_init); pos_init = random(gm_init,N))
    # ============================================================
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

    doot_config = DootConfig(
        num_neighbors=min(10, num_vehicles - 1),  # MATLAB n_neigh = 10
        max_iter_primaldual=10,
        use_random_sampling=True,
        num_trial_move_samples=300,               # MATLAB n_trial_samples = 300
        min_displacement_norm=0.1,                # MATLAB d>=0.1
        planar_std_threshold=0.1,
    )

    coordinator = DootCbfCoordinator(
        vehicles=vehicles,
        velocity_max=1.0,
        targeted_positions=targeted_positions,
        doot_config=doot_config,
    )

    # Output path: same directory as this file
    out_dir = os.path.dirname(os.path.abspath(__file__))
    avi_path = os.path.join(out_dir, "doot_cbf_coordinator_ut02.avi")

    cur_time = 0.0
    cur_x = copy.deepcopy(x0s)

    # --- Matplotlib 2D plot setup ---
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect("equal")
    ax.grid(True)
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    # Show target samples (discrete points) like MATLAB "samples"
    ax.scatter(samples_xy[:, 0], samples_xy[:, 1], s=8, c="k", alpha=0.15, linewidths=0)

    # Initial agent scatter
    XY0 = np.array([cur_x[i]["x"][:2] for i in range(num_vehicles)], dtype=float)

    # Random per-vehicle colors (different each run is OK)
    colors = rng.random((num_vehicles, 3))

    scat = ax.scatter(XY0[:, 0], XY0[:, 1], s=20, c=colors)

    fig.canvas.draw()

    # --- OpenCV VideoWriter ---
    width, height = fig.canvas.get_width_height()
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    video = cv2.VideoWriter(avi_path, fourcc, 10, (width, height))

    for step in range(intervals):
        coordinator.step(cur_time, cur_x)
        v_cmd_fns = coordinator.get_v_cmd_fns()

        for i in range(num_vehicles):
            v = np.asarray(v_cmd_fns[i](cur_time), dtype=float).reshape(3,)
            v[2] = 0.0  # enforce 2D
            cur_x[i]["x"] += v * dt
            cur_x[i]["x"][2] = 0.0

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


if __name__ == "__main__":
    print(f"==== Start DootCbfCoordinator unit test 01 ====")
    unit_test_01()
    print(f"==== Finish DootCbfCoordinator unit test 01 ====")

    print(f"==== Start DootCbfCoordinator unit test 02 ====")
    unit_test_02()
    print(f"==== Finish DootCbfCoordinator unit test 02 ====")
