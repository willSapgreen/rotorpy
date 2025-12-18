import numpy as np
from typing import Callable, List, Optional, Dict, Any


class DootCbfCoordinator:
    """
    Coordinator that stores per-drone commanded velocities and exposes
    per-drone callables v_cmd_fn(t) -> R^3.

    This is intentionally algorithm-agnostic: you can plug in DOOT, CBF, both,
    or any other policy in `step(...)`.
    """
    def __init__(self, num_drones: int, v_max: Optional[float] = None):
        self.N = int(num_drones)
        self.v_max = v_max  # optional speed cap (m/s)

        # latest commanded velocity for each drone, world frame, m/s
        self._v_cmd = np.zeros((self.N, 3), dtype=float)

        # pre-build per-drone callables
        self._v_cmd_fns = [self._make_v_cmd_fn(i) for i in range(self.N)]

    def _make_v_cmd_fn(self, i: int) -> Callable[[float], np.ndarray]:
        # Note: closure captures `self` and index `i`.
        def v_cmd_fn(t: float) -> np.ndarray:
            return self._v_cmd[i]
        return v_cmd_fn

    def get_v_cmd_fns(self) -> List[Callable[[float], np.ndarray]]:
        return self._v_cmd_fns

    def set_v_cmd(self, i: int, v: np.ndarray) -> None:
        """Directly set v_cmd for one drone (after DOOT+CBF, for example)."""
        v = np.asarray(v, dtype=float).reshape(3,)
        if self.v_max is not None:
            s = np.linalg.norm(v)
            if s > self.v_max and s > 0:
                v = v * (self.v_max / s)
        self._v_cmd[i, :] = v

    def set_v_cmd_batch(self, V: np.ndarray) -> None:
        """Directly set all v_cmd, shape (N,3)."""
        V = np.asarray(V, dtype=float)
        assert V.shape == (self.N, 3)
        if self.v_max is not None:
            speeds = np.linalg.norm(V, axis=1)
            scale = np.ones((self.N,), dtype=float)
            mask = speeds > self.v_max
            scale[mask] = self.v_max / speeds[mask]
            V = V * scale[:, None]
        self._v_cmd[:, :] = V

    def step(self, t: float, states: List[Dict[str, Any]]) -> None:
        """
        Example placeholder: compute v_cmd from states.
        Replace this with real DOOT+CBF.

        states: list of per-drone dicts, e.g. state['x'], state['v'], ...
        """
        # Example policy: command +X for all drones (for demonstration only)
        V = np.zeros((self.N, 3), dtype=float)
        V[:, 0] = 1.0
        self.set_v_cmd_batch(V)
