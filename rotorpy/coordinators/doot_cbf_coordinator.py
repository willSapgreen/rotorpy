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
        self.N = len(self.vehicles)
        if self.N < 1:
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
        self.phi = np.zeros((self.N,), dtype=float)  # dual potential at agent locations
        self.last_time: Optional[float] = None

        # RNG
        self.rng: Optional[np.random.Generator] = None
        if self.doot_config.use_random_sampling:
            self.rng = np.random.default_rng(self._FIXED_SEED)

        # Command storage
        self._v_cmd = np.zeros((self.N, 3), dtype=float)
        self._v_cmd_fns = [self._make_v_cmd_fn(i) for i in range(self.N)]

    def _make_v_cmd_fn(self, i: int) -> Callable[[float], np.ndarray]:
        def v_cmd_fn(t: float) -> np.ndarray:
            return self._v_cmd[i]
        return v_cmd_fn

    def get_v_cmd_fns(self) -> List[Callable[[float], np.ndarray]]:
        return self._v_cmd_fns

    def set_v_cmd_batch(self, V: np.ndarray) -> None:
        V = np.asarray(V, dtype=float)
        if V.shape != (self.N, 3):
            raise ValueError(f"V must have shape {(self.N, 3)}, got {V.shape}")

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
          states: list of length self.N, only for member vehicles; each must include 'x' (3,)

        Behavior:
          - compute dt = t - last_time
          - update phi via primal-dual iterations
          - build interpolant PHI at current agent positions
          - choose trial move per agent: argmin(||d|| + PHI(x+d)) with ||d|| >= min_displacement_norm
          - convert displacement to velocity v = d/dt
          - store v in _v_cmd
        """
        if len(states) != self.N:
            raise ValueError(f"states must have length N={self.N}, got {len(states)}")

        # Initialize time on first call
        if self.last_time is None:
            self.last_time = float(t)
            self.set_v_cmd_batch(np.zeros((self.N, 3), dtype=float))
            return

        dt = float(t) - float(self.last_time)
        self.last_time = float(t)

        if (not np.isfinite(dt)) or dt <= 0.0:
            self.set_v_cmd_batch(np.zeros((self.N, 3), dtype=float))
            return

        # Extract positions X (N,3)
        X = np.zeros((self.N, 3), dtype=float)
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
            Xp = X[:, axes_2d]                           # (N,2)
            Tp = self.targeted_positions[:, axes_2d]     # (M,2)
        else:
            axes_2d = None
            dropped_axis = None
            Xp = X                                       # (N,3)
            Tp = self.targeted_positions                 # (M,3)

        # ------------------------------------------------------------
        # 1) count: assign each target sample to nearest agent
        # ------------------------------------------------------------
        M = int(self.targeted_positions.shape[0])
        diff_ta = Tp[:, None, :] - Xp[None, :, :]        # (M,N,dim)
        d2_ta = np.sum(diff_ta * diff_ta, axis=2)        # (M,N)
        nearest_agent = np.argmin(d2_ta, axis=1)         # (M,)
        count = np.bincount(nearest_agent, minlength=self.N).astype(float) / float(M)

        # ------------------------------------------------------------
        # 2) Neighbor graph + Laplacian from agent kNN (excluding self)
        # ------------------------------------------------------------
        k = int(self.doot_config.num_neighbors)
        if k >= self.N:
            # Excluding self, max meaningful neighbors is N-1
            k = self.N - 1

        diff_aa = Xp[:, None, :] - Xp[None, :, :]        # (N,N,dim)
        d2_aa = np.sum(diff_aa * diff_aa, axis=2)        # (N,N)
        np.fill_diagonal(d2_aa, np.inf)

        # kth index safety when N is small
        kth = min(k - 1, self.N - 2) if self.N >= 2 else 0
        nn_idx = np.argpartition(d2_aa, kth=kth, axis=1)[:, :k]  # (N,k)

        A = np.zeros((self.N, self.N), dtype=float)
        rows = np.repeat(np.arange(self.N), k)
        cols = nn_idx.reshape(-1)
        A[rows, cols] = 1.0

        W = 0.5 * (A + A.T)
        D = np.diag(np.sum(W, axis=1))
        L = D - W

        # ------------------------------------------------------------
        # 3) Primal–dual inner iterations updating phi
        # ------------------------------------------------------------
        one_over_N = 1.0 / float(self.N)
        gain = 1.0 / float(k + 1)

        phi = self.phi
        for _ in range(int(self.doot_config.max_iter_primaldual)):
            phi = phi - gain * (L @ phi) + one_over_N * np.ones((self.N,), dtype=float) - count
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
        V_nom = np.zeros((self.N, 3), dtype=float)

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
            self.set_v_cmd_batch(np.zeros((self.N, 3), dtype=float))
            return

        for m in range(self.N):
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
