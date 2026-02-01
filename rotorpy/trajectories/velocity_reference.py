import numpy as np


class VelocityReference(object):
    """
    Velocity-driven reference generator (2nd-order command filter on velocity).

    Public interface:
      - __init__(v_cmd_fn, init_pos, init_yaw=0.0, yaw_mode="velocity_heading", yaw_speed_eps=1e-3)
      - update(t) -> flat_output dict with keys:
          x, x_dot, x_ddot, x_dddot, x_ddddot, yaw, yaw_dot, yaw_ddot

    2nd-order filter (per axis):
      v_dot = a
      a_dot = -2*zeta*omega*a - omega^2*(v - v_cmd)

    Discretized with explicit Euler using dt between update calls.

    Tunable attributes (set after construction if desired):
      self.omega (rad/s), self.zeta
      self.v_max (m/s), self.a_max (m/s^2), self.j_max (m/s^3)
      self.yaw_rate_max (rad/s)
    """

    def __init__(self, v_cmd_fn, init_pos, init_yaw=0.0,
                 yaw_mode="velocity_heading",
                 yaw_speed_eps=1e-3):
        self.v_cmd_fn = v_cmd_fn

        self.x_ref = np.array(init_pos, dtype=float).copy()

        self.yaw_mode = yaw_mode
        self.yaw_speed_eps = float(yaw_speed_eps)

        self._last_t = None

        # Defaults (adjust as needed)
        self.omega = 6.0   # rad/s
        self.zeta = 1.0    # critically damped

        self.v_max = None         # m/s cap on ||v||
        self.a_max = None         # m/s^2 cap on ||a||
        self.j_max = None         # m/s^3 cap on ||a_dot||
        self.yaw_rate_max = None  # rad/s cap on |yaw_dot|

        # Filter states
        self._initialized = False
        self._v = np.zeros(3, dtype=float)  # filtered velocity v_d
        self._a = np.zeros(3, dtype=float)  # filtered acceleration a_d

        # Yaw state (continuous)
        self._yaw = float(init_yaw)

    @staticmethod
    def _wrap_pi(angle):
        return (angle + np.pi) % (2.0 * np.pi) - np.pi

    @staticmethod
    def _clip_norm(vec, max_norm):
        if max_norm is None:
            return vec
        n = float(np.linalg.norm(vec))
        if n < 1e-12:
            return vec
        if n > max_norm:
            return vec * (max_norm / n)
        return vec

    def update(self, t):
        t = float(t)
        if self._last_t is None:
            dt = 0.0
        else:
            dt = max(0.0, t - self._last_t)

        # Raw command from DOOT/CBF
        v_cmd = np.array(self.v_cmd_fn(t), dtype=float).reshape(3)
        v_cmd = self._clip_norm(v_cmd, self.v_max)

        # Initialize filter state on first call
        if not self._initialized:
            self._v = v_cmd.copy()
            self._a = np.zeros(3, dtype=float)
            self._initialized = True

        # 2nd-order filter update (explicit Euler)
        if dt > 0.0:
            w = float(self.omega)
            z = float(self.zeta)

            # a_dot (jerk)
            a_dot = -2.0 * z * w * self._a - (w * w) * (self._v - v_cmd)

            # Optional jerk cap
            a_dot = self._clip_norm(a_dot, self.j_max)

            # Integrate accel
            a_new = self._a + dt * a_dot
            a_new = self._clip_norm(a_new, self.a_max)

            # Integrate velocity
            v_new = self._v + dt * a_new
            v_new = self._clip_norm(v_new, self.v_max)

            # Integrate position reference
            self.x_ref = self.x_ref + dt * v_new

            # Commit filter states
            self._a = a_new
            self._v = v_new

        # Yaw handling (based on filtered velocity)
        yaw = self._yaw
        yaw_dot = 0.0

        if self.yaw_mode == "constant":
            yaw = self._yaw
            yaw_dot = 0.0

        elif self.yaw_mode == "velocity_heading":
            vx, vy = float(self._v[0]), float(self._v[1])
            s2 = vx * vx + vy * vy

            if s2 > self.yaw_speed_eps ** 2 and dt > 0.0:
                yaw_new = float(np.arctan2(vy, vx))

                # Continuous yaw update
                dyaw = self._wrap_pi(yaw_new - self._yaw)
                yaw = self._yaw + dyaw
                yaw_dot = dyaw / dt

                if self.yaw_rate_max is not None:
                    yaw_dot = float(np.clip(yaw_dot, -self.yaw_rate_max, self.yaw_rate_max))
            else:
                yaw = self._yaw
                yaw_dot = 0.0

        else:
            raise ValueError(f"Unknown yaw_mode: {self.yaw_mode}")

        # Commit time + yaw
        self._last_t = t
        self._yaw = float(yaw)

        flat_output = {
            'x': self.x_ref.copy(),
            'x_dot': self._v.copy(),
            'x_ddot': self._a.copy(),
            'x_dddot': np.zeros(3),
            'x_ddddot': np.zeros(3),
            'yaw': float(yaw),
            'yaw_dot': float(yaw_dot),
            'yaw_ddot': 0.0,
        }
        return flat_output
