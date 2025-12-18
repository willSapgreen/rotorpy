import numpy as np

class VelocityReference(object):
    """
    Velocity-driven reference generator with optional yaw from velocity heading.
    """

    def __init__(self, v_cmd_fn, x0, yaw0=0.0,
                 yaw_mode="velocity_heading",
                 yaw_speed_eps=1e-3):
        """
        v_cmd_fn(t) -> (3,) commanded velocity in world frame, m/s

        yaw_mode:
          - "constant": keep yaw = yaw0, yaw_dot = 0
          - "velocity_heading": yaw aligns with horizontal velocity direction
        yaw_speed_eps: threshold on horizontal speed below which we hold yaw
        """
        self.v_cmd_fn = v_cmd_fn
        self.x_ref = np.array(x0, dtype=float).reshape(3,)
        self.yaw_mode = yaw_mode
        self.yaw_speed_eps = float(yaw_speed_eps)

        self._last_t = None
        self._last_v = None
        self._yaw = float(yaw0)

    def _wrap_pi(self, a):
        return (a + np.pi) % (2*np.pi) - np.pi

    def update(self, t):
        if self._last_t is None:
            self._last_t = t

        dt = t - self._last_t
        self._last_t = t

        v = np.array(self.v_cmd_fn(t), dtype=float).reshape(3,)

        # Integrate x_ref so x and x_dot are kinematically consistent
        if dt > 0.0:
            self.x_ref = self.x_ref + dt * v

        # Yaw / yaw_dot
        yaw = self._yaw
        yaw_dot = 0.0

        if self.yaw_mode == "velocity_heading":
            vx, vy = v[0], v[1]
            s2 = vx*vx + vy*vy

            if s2 > self.yaw_speed_eps**2:
                yaw_new = np.arctan2(vy, vx)

                # Estimate v_dot for yaw_dot (finite difference)
                if self._last_v is not None and dt > 0.0:
                    vdot = (v - self._last_v) / dt
                    vdx, vdy = vdot[0], vdot[1]
                    yaw_dot = (vx * vdy - vy * vdx) / s2

                # Update stored yaw with wrapped continuity
                yaw = yaw_new
                self._yaw = yaw
            else:
                # horizontal speed too small: keep last yaw, yaw_dot = 0
                yaw = self._yaw
                yaw_dot = 0.0

        elif self.yaw_mode == "constant":
            yaw = self._yaw
            yaw_dot = 0.0
        else:
            raise ValueError(f"Unknown yaw_mode: {self.yaw_mode}")

        self._last_v = v.copy()

        flat_output = {
            'x': self.x_ref,
            'x_dot': v,
            'x_ddot': np.zeros(3),
            'x_dddot': np.zeros(3),
            'x_ddddot': np.zeros(3),
            'yaw': float(yaw),
            'yaw_dot': float(yaw_dot),
            'yaw_ddot': 0.0,
        }
        return flat_output
