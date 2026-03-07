from __future__ import annotations

from typing import Optional, Dict, Union
import numpy as np
import torch


class VelocityReference:
    """
    Velocity-driven reference generator (2nd-order command filter on velocity).

    Public interface:
      - __init__(init_pos, init_yaw=0.0, init_time=0.0,
                 yaw_mode="velocity_heading", yaw_speed_eps=1e-3)
      - update(t) -> flat_output dict with keys:
          x, x_dot, x_ddot, x_dddot, x_ddddot, yaw, yaw_dot, yaw_ddot

    2nd-order filter (per axis):
      v_dot = a
      a_dot = -2*zeta*omega*a - omega^2*(v - v_cmd)

    Discretized with explicit Euler using dt between update calls.

    Tunable attributes (set after construction if desired):
      self._omega (rad/s), self._zeta
      self._v_max (m/s), self._a_max (m/s^2), self._j_max (m/s^3)
      self._yaw_rate_max (rad/s)
    """

    def __init__(
        self,
        init_pos: np.ndarray,          # shape (3,)
        init_yaw: float = 0.0,
        init_time: float = 0.0,
        yaw_mode: str = "velocity_heading",
        yaw_speed_eps: float = 1e-3,
    ) -> None:

        # Initialize the states
        self._pos: np.ndarray = init_pos.astype(float).copy()
        self._yaw: float = float(init_yaw)
        self._yaw_mode: str = yaw_mode
        self._yaw_speed_eps: float = float(yaw_speed_eps)
        self._last_t: float = float(init_time)

        self._next_vel_cmd: np.ndarray = np.zeros(3, dtype=float)
        self._has_vel_cmd: bool = False

        # Set up the default config
        # ToDo: make them configurable
        self._omega: float = 6.0   # rad/s
        self._zeta: float = 1.0    # critically damped
        self._v_max: Optional[float] = None         # m/s cap on ||v||
        self._a_max: Optional[float] = None         # m/s^2 cap on ||a||
        self._j_max: Optional[float] = None         # m/s^3 cap on ||a_dot||
        self._yaw_rate_max: Optional[float] = None  # rad/s cap on |yaw_dot|

        # Set filter states
        self._initialized: bool = False
        self._v: np.ndarray = np.zeros(3, dtype=float)  # filtered velocity v_d
        self._a: np.ndarray = np.zeros(3, dtype=float)  # filtered acceleration a_d


    def set_vel_cmd(self, v_cmd: np.ndarray) -> None:
        self._next_vel_cmd = v_cmd.copy()
        self._has_vel_cmd = True


    def update(self, t: float) -> Dict[str, np.ndarray | float]:

        if not self._has_vel_cmd:
            raise RuntimeError("Next velocity command is invalid")

        # Calculate delta time
        t = float(t)
        if t < self._last_t:
            raise RuntimeError(f"Time went backwards: t={t} < last_t={self._last_t}")
        dt: float = t - self._last_t

        v_cmd = self._clip_norm(self._next_vel_cmd, self._v_max)

        # Initialize filter state on first call
        if not self._initialized:
            self._v = v_cmd.copy()
            self._a = np.zeros(3, dtype=float)
            self._initialized = True

        # 2nd-order filter update (explicit Euler)
        if dt > 0.0:
            w = self._omega
            z = self._zeta

            # a_dot (jerk)
            a_dot = -2*z*w*self._a - (w*w)*(self._v - v_cmd)

            # Optional jerk cap
            a_dot = self._clip_norm(a_dot, self._j_max)

            # Integrate accel
            a_new = self._a + dt*a_dot
            a_new = self._clip_norm(a_new, self._a_max)

            # Integrate velocity
            v_new = self._v + dt*a_new
            v_new = self._clip_norm(v_new, self._v_max)

            # Integrate position reference
            self._pos = self._pos + dt*v_new

            # Commit filter states
            self._a = a_new
            self._v = v_new

        # Yaw handling (based on filtered velocity)
        yaw = self._yaw
        yaw_dot: float = 0.0

        if self._yaw_mode == "velocity_heading":
            vx, vy = self._v[0], self._v[1]
            s2 = vx*vx + vy*vy

            if s2 > self._yaw_speed_eps**2 and dt > 0.0:
                yaw_new = float(np.arctan2(vy, vx))

                # Continuous yaw update
                dyaw = self._wrap_pi(yaw_new - self._yaw)
                yaw = self._yaw + dyaw
                yaw_dot = dyaw / dt

                if self._yaw_rate_max is not None:
                    yaw_dot = float(np.clip(yaw_dot, -self._yaw_rate_max, self._yaw_rate_max))

        self._last_t = t
        self._yaw = yaw

        return {
            'x': self._pos.copy(),
            'x_dot': self._v.copy(),
            'x_ddot': self._a.copy(),
            'x_dddot': np.zeros(3),
            'x_ddddot': np.zeros(3),
            'yaw': yaw,
            'yaw_dot': yaw_dot,
            'yaw_ddot': 0.0,
        }


    @staticmethod
    def _wrap_pi(angle: float) -> float:
        return (angle + np.pi) % (2*np.pi) - np.pi


    @staticmethod
    def _clip_norm(vec: np.ndarray, max_norm: Optional[float]) -> np.ndarray:
        if max_norm is None:
            return vec
        n = float(np.linalg.norm(vec))
        if n < 1e-12:
            return vec
        if n > max_norm:
            return vec * (max_norm / n)
        return vec


class BatchedVelocityReference:
    """
    Batched velocity-driven reference generator (2nd-order command filter on velocity).

    Same 2nd-order filter:
      v_dot = a
      a_dot = -2*zeta*omega*a - omega^2*(v - v_cmd)

    Explicit Euler integration per time step.
    """

    def __init__(
        self,
        init_pos: torch.Tensor,   # shape (N,3)
        init_yaw: float = 0.0,
        init_time: float = 0.0,
        yaw_mode: str = "velocity_heading",
        yaw_speed_eps: float = 1e-3,
    ) -> None:

        self.N: int = init_pos.shape[0]
        self.device = init_pos.device

        # Initialize the states
        self._pos: torch.Tensor = init_pos.clone()
        self._yaw: torch.Tensor = torch.full((self.N,), init_yaw, device=self.device, dtype=torch.float64)
        self._last_t: torch.Tensor = torch.full((self.N,), init_time, device=self.device, dtype=torch.float64)

        self._next_vel_cmd: torch.Tensor = torch.zeros((self.N,3), device=self.device, dtype=torch.float64)
        self._has_vel_cmd: bool = False

        # Default config
        self._omega: float = 6.0
        self._zeta: float = 1.0
        self._v_max: Optional[float] = None
        self._a_max: Optional[float] = None
        self._j_max: Optional[float] = None
        self._yaw_rate_max: Optional[float] = None

        # Filter states
        self._initialized: bool = False
        self._v: torch.Tensor = torch.zeros_like(self._pos)
        self._a: torch.Tensor = torch.zeros_like(self._pos)

        self._yaw_mode = yaw_mode
        self._yaw_speed_eps = yaw_speed_eps

    def set_vel_cmd(self, v_cmd: torch.Tensor) -> None:
        self._next_vel_cmd = v_cmd.clone()
        self._has_vel_cmd = True


    def update(self, t) -> Dict[str, torch.Tensor]:
        if not self._has_vel_cmd:
            raise RuntimeError("Next velocity command is invalid")

        # Accept float / numpy / torch, normalize to torch (N,) on self.device
        if isinstance(t, (float, int, np.floating, np.integer)):
            t_vec = torch.full((self.N,), float(t), device=self.device, dtype=torch.float64)

        elif isinstance(t, np.ndarray):
            t_np = np.asarray(t, dtype=float).reshape(-1)
            if t_np.size == 1:
                t_vec = torch.full((self.N,), float(t_np.item()), device=self.device, dtype=torch.float64)
            else:
                if t_np.size != self.N:
                    raise ValueError(f"Expected numpy time array of shape ({self.N},), got ({t_np.size},)")
                t_vec = torch.as_tensor(t_np, device=self.device, dtype=torch.float64)

        elif torch.is_tensor(t):
            # allow scalar tensor or (N,) tensor
            if t.numel() == 1:
                t_vec = torch.full((self.N,), float(t.item()), device=self.device, dtype=torch.float64)
            else:
                t_vec = t.to(device=self.device, dtype=torch.float64).reshape(self.N,)

        else:
            raise TypeError(f"Unsupported time type: {type(t)}")

        if torch.any(t_vec < self._last_t):
            raise RuntimeError("Time went backwards")

        dt = t_vec - self._last_t
        v_cmd = self._clip_norm_batch(self._next_vel_cmd, self._v_max)

        if not self._initialized:
            self._v = v_cmd.clone()
            self._initialized = True

        dt_mask = dt > 0

        if torch.any(dt_mask):
            w, z = self._omega, self._zeta

            a_dot = -2*z*w*self._a - (w*w)*(self._v - v_cmd)
            a_dot = self._clip_norm_batch(a_dot, self._j_max)

            a_new = self._a + dt.unsqueeze(1)*a_dot
            a_new = self._clip_norm_batch(a_new, self._a_max)

            v_new = self._v + dt.unsqueeze(1)*a_new
            v_new = self._clip_norm_batch(v_new, self._v_max)

            self._pos = torch.where(dt_mask.unsqueeze(1), self._pos + dt.unsqueeze(1)*v_new, self._pos)
            self._a = torch.where(dt_mask.unsqueeze(1), a_new, self._a)
            self._v = torch.where(dt_mask.unsqueeze(1), v_new, self._v)

        # -------------------------
        # Yaw handling (batched, based on filtered velocity)
        # -------------------------
        yaw = self._yaw.clone()
        yaw_dot = torch.zeros((self.N,), device=self.device, dtype=torch.float64)

        if self._yaw_mode == "velocity_heading":
            vx = self._v[:, 0]
            vy = self._v[:, 1]
            s2 = vx * vx + vy * vy

            # only update yaw where speed is sufficient AND dt > 0
            active = (s2 > (self._yaw_speed_eps ** 2)) & (dt > 0)

            if torch.any(active):
                yaw_new = torch.atan2(vy, vx)
                dyaw = self._wrap_pi_batch(yaw_new - self._yaw)
                yaw = torch.where(active, self._yaw + dyaw, yaw)

                # yaw rate
                yaw_dot_active = dyaw / dt.clamp(min=1e-12)
                yaw_dot = torch.where(active, yaw_dot_active, yaw_dot)

                if self._yaw_rate_max is not None:
                    yaw_dot = torch.clamp(yaw_dot, -float(self._yaw_rate_max), float(self._yaw_rate_max))

        # Commit yaw
        self._yaw = yaw

        # Update time
        self._last_t = t_vec

        zeros3 = torch.zeros((self.N, 3), device=self.device, dtype=torch.float64)
        zeros1 = torch.zeros((self.N,), device=self.device, dtype=torch.float64)
        return {
            "x": self._pos.clone(),
            "x_dot": self._v.clone(),
            "x_ddot": self._a.clone(),
            "x_dddot": zeros3,     # jerk (not modeled in this ref; set 0 like non-batched)
            "x_ddddot": zeros3,    # snap (set 0)
            "yaw": self._yaw.clone(),
            "yaw_dot": yaw_dot.clone(),
            "yaw_ddot": zeros1,    # yaw accel (set 0)
        }

    @staticmethod
    def _clip_norm_batch(vec: torch.Tensor, max_norm: Optional[float]) -> torch.Tensor:
        if max_norm is None:
            return vec
        n = torch.linalg.norm(vec, dim=1).clamp(min=1e-12)
        scale = torch.clamp(max_norm / n, max=1.0)
        return vec * scale.unsqueeze(1)

    @staticmethod
    def _wrap_pi_batch(angle: torch.Tensor) -> torch.Tensor:
        # Map to (-pi, pi]
        return (angle + torch.pi) % (2.0 * torch.pi) - torch.pi
