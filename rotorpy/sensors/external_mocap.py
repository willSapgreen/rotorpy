import numpy as np
from scipy.spatial.transform import Rotation
import torch
import copy
from typing import Dict, List, Optional, Union


def hat_map(s):
        """
        Given vector s in R^3, return associate skew symmetric matrix S in R^3x3
        """
        return np.array([[    0, -s[2],  s[1]],
                         [ s[2],     0, -s[0]],
                         [-s[1],  s[0],     0]])

########### Original NumPy version of MotionCapture (existing) ###########

class MotionCapture():
    """
    The external motion capture is able to provide pose and twist measurements of the vehicle.
    Given the current ground truth state of the vehicle, it will output noisy measurements of the
    pose and twist. Artifacts can be introduced.
    """
    def __init__(self, sampling_rate,
                 mocap_params={'pos_noise_density': 0.0005*np.ones((3,)),  # noise density for position
                                                    'vel_noise_density': 0.005*np.ones((3,)),          # noise density for velocity
                                                    'att_noise_density': 0.0005*np.ones((3,)),          # noise density for attitude
                                                    'rate_noise_density': 0.0005*np.ones((3,)),         # noise density for body rates
                                                    'vel_artifact_max': 5,                              # maximum magnitude of the artifact in velocity (m/s)
                                                    'vel_artifact_prob': 0.001,                         # probability that an artifact will occur for a given velocity measurement
                                                    'rate_artifact_max': 1,                             # maximum magnitude of the artifact in body rates (rad/s)
                                                    'rate_artifact_prob': 0.0002                        # probability that an artifact will occur for a given rate measurement
                                                    },
                with_artifacts=False):
        """
        Parameters:
            sampling_rate, Hz, the rate at which this sensor is being sampled. Used for computing the noise.
            mocap_params, a dict with keys
                pos_noise_density, position noise density, [m/sqrt(Hz)]
                vel_noise_density, velocity noise density, [m/s / sqrt(Hz)]
                att_noise_density, attitude noise density, [rad / sqrt(Hz)]
                rate_noise_density, attitude rate noise density, [rad/s /sqrt(Hz)]
                vel_artifact_prob, probability that a spike will occur for a given velocity measurement
                vel_artifact_max, the maximum magnitude of the artifact spike. [m/s]
                rate_artifact_prob, probability that a spike will occur for a given body rate measurement
                rate_artifact_max, the maximum magnitude of hte artifact spike. [rad/s]
        """

        self.rate_scale = np.sqrt(sampling_rate/2)

        # Noise densities
        self.pos_density = mocap_params['pos_noise_density']
        self.vel_density = mocap_params['vel_noise_density']
        self.att_density = mocap_params['att_noise_density']
        self.rate_density = mocap_params['rate_noise_density']

        # Artifacts
        self.vel_artifact_prob = mocap_params['vel_artifact_prob']
        self.vel_artifact_max = mocap_params['vel_artifact_max']
        self.rate_artifact_prob = mocap_params['rate_artifact_prob']
        self.rate_artifact_max = mocap_params['rate_artifact_max']

        self.initialized = True
        self.with_artifacts = with_artifacts


    def measurement(self, state, with_noise=False, with_artifacts=False):
        """
        Computes and returns the sensor measurement at a time step.
        Inputs:
            state, a dict describing the state with keys
                    x, position, m, shape=(3,)
                    v, linear velocity, m/s, shape=(3,)
                    q, quaternion [i,j,k,w], shape=(4,)
                    w, angular velocity (in LOCAL frame!), rad/s, shape=(3,)
            with_noise, a boolean to indicate if noise is added
            with_artifacts, a boolean to indicate if artifacts are added.
                    Artifacts are added to the velocity and angular rates, and are due
                    to the numerical differentiation scheme used by motion capture systems.
                    They will appear as random spikes in the data.
        Outputs:
            observation, a dictionary with keys
                    x_m, noisy position measurement, m, shape=(3,)
                    v_m, noisy linear velocity, m/s, shape=(3,)
                    q_m, noisy quaternion [i,j,k,w], shape=(4,)
                    w_m, noisy angular velocity (in LOCAL frame!), rad/s, shape=(3,)
        """
        x_measured = copy.deepcopy(state['x']).astype(float)
        v_measured = copy.deepcopy(state['v']).astype(float)
        q_measured = Rotation.from_quat(copy.deepcopy(state['q']))
        w_measured = copy.deepcopy(state['w']).astype(float)

        if with_noise:
            # Add noise to the measurements based on the provided measurement noise.
            x_measured += self.rate_scale * np.random.normal(scale=np.abs(self.pos_density))
            v_measured += self.rate_scale * np.random.normal(scale=np.abs(self.vel_density))
            w_measured += self.rate_scale * np.random.normal(scale=np.abs(self.rate_density))

            # Noise has to be treated differently with quaternions...
            # Following https://www.iri.upc.edu/people/jsola/JoanSola/objectes/notes/kinematics.pdf  pg 43
            # First, let's produce a perturbation vector in R3
            delta_phi = self.rate_scale*np.random.normal(scale=np.abs(self.att_density))

            # Now convert that to a rotation matrix
            delta_rotation = Rotation.from_matrix(np.eye(3) + hat_map(delta_phi))

            # Now apply that rotation to the quaternion
            q_measured = (q_measured * delta_rotation).as_quat()
        else:
            q_measured = q_measured.as_quat()

        if with_artifacts:
            # If including artifacts, first roll the dice on whether or not a spike should occur for each measurement:
            vel_spike_bool = np.random.choice([0,1], p=[1-self.vel_artifact_prob, self.vel_artifact_prob])
            rate_spike_bool = np.random.choice([0,1], p=[1-self.rate_artifact_prob, self.rate_artifact_prob])

            # Choose the axis that the spike will occur on
            vel_axis = np.random.choice([0,1,2])
            rate_axis = np.random.choice([0,1,2])

            # Choose the sign of the spike
            vel_sign = np.random.choice([-1,1])
            rate_sign = np.random.choice([-1,1])

            if vel_spike_bool:
                v_measured[vel_axis] += vel_sign*np.random.uniform(low=0, high=self.vel_artifact_max)
            if rate_spike_bool:
                w_measured[rate_axis] += rate_sign*np.random.uniform(low=0, high=self.rate_artifact_max)

        return {'x': x_measured, 'q': q_measured, 'v': v_measured, 'w': w_measured}


def quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Hamilton product, quats in [i, j, k, w]. Shapes (...,4)."""
    x1, y1, z1, w1 = q1.unbind(dim=-1)
    x2, y2, z2, w2 = q2.unbind(dim=-1)

    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    return torch.stack((x, y, z, w), dim=-1)


def quat_normalize(q: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return q / q.norm(dim=-1, keepdim=True).clamp_min(eps)


def small_angle_delta_quat(delta_phi: torch.Tensor) -> torch.Tensor:
    """
    Small-angle rotation vector -> quaternion, [i,j,k,w].
    delta_q ≈ [0.5*delta_phi, 1], normalized.
    """
    dq = torch.cat([0.5 * delta_phi, torch.ones_like(delta_phi[..., :1])], dim=-1)
    return quat_normalize(dq)


########### Batched version of MotionCapture using PyTorch ###########

class BatchedMotionCapture:
    """
    Per-drone batched motion capture.

    Assumptions for measurement():
        state['x']: (N, 3)
        state['v']: (N, 3)
        state['q']: (N, 4) in [i,j,k,w]
        state['w']: (N, 3)

    Constructor expects:
        mocap_params: List[Dict[str, torch.Tensor]] length N
        Each dict has keys:
            'pos_noise_density': (3,) or (1,3) or (N,3) but per-drone item should be (3,) or (1,3)
            'vel_noise_density': (3,) or (1,3)
            'att_noise_density': (3,) or (1,3)
            'rate_noise_density': (3,) or (1,3)
            'vel_artifact_max': (1,) or scalar tensor
            'vel_artifact_prob': (1,) or scalar tensor
            'rate_artifact_max': (1,) or scalar tensor
            'rate_artifact_prob': (1,) or scalar tensor

        with_artifacts: list[bool] length N OR torch.BoolTensor shape (N,) or (N,1)
    """

    REQUIRED_KEYS = (
        "pos_noise_density",
        "vel_noise_density",
        "att_noise_density",
        "rate_noise_density",
        "vel_artifact_max",
        "vel_artifact_prob",
        "rate_artifact_max",
        "rate_artifact_prob",
    )

    def __init__(
        self,
        num_drones: int,
        sampling_rate: float,
        mocap_params: List[Dict[str, torch.Tensor]],
        with_artifacts: Union[List[bool], torch.Tensor],
        device: str = "cpu",
        dtype: torch.dtype = torch.float64,
    ):
        self.device = torch.device(device)
        self.dtype = dtype
        self.num_drones = int(num_drones)
        self.sampling_rate = float(sampling_rate)

        if not isinstance(mocap_params, list) or len(mocap_params) != self.num_drones:
            raise ValueError(f"mocap_params must be a list of length num_drones={self.num_drones}")

        # sqrt(sampling_rate/2)
        self.rate_scale = (self.sampling_rate / 2.0) ** 0.5

        # --- validate keys and tensor types (caller responsibility) ---
        for i, mp in enumerate(mocap_params):
            if not isinstance(mp, dict):
                raise TypeError(f"mocap_params[{i}] must be a dict")
            missing = [k for k in self.REQUIRED_KEYS if k not in mp]
            if missing:
                raise KeyError(f"mocap_params[{i}] missing keys: {missing}")
            for k in self.REQUIRED_KEYS:
                if not isinstance(mp[k], torch.Tensor):
                    raise TypeError(f"mocap_params[{i}]['{k}'] must be a torch.Tensor (caller responsibility)")

        # --- helper: normalize shapes for stacking ---
        def _as_row3(t: torch.Tensor) -> torch.Tensor:
            # Accept (3,) or (1,3). Return (1,3).
            if t.dim() == 1 and t.numel() == 3:
                return t.view(1, 3)
            if t.dim() == 2 and t.shape == (1, 3):
                return t
            raise ValueError(f"Expected tensor shape (3,) or (1,3); got {tuple(t.shape)}")

        def _as_scalar1(t: torch.Tensor) -> torch.Tensor:
            # Accept scalar () or (1,) or (1,1). Return (1,1).
            if t.dim() == 0:
                return t.view(1, 1)
            if t.dim() == 1 and t.numel() == 1:
                return t.view(1, 1)
            if t.dim() == 2 and t.shape == (1, 1):
                return t
            raise ValueError(f"Expected scalar/(1,)/(1,1); got {tuple(t.shape)}")

        # --- stack per-drone params into batched tensors ---
        pos_list = []
        vel_list = []
        att_list = []
        rate_list = []

        vel_p_list = []
        vel_m_list = []
        rate_p_list = []
        rate_m_list = []

        for mp in mocap_params:
            pos_list.append(_as_row3(mp["pos_noise_density"]))
            vel_list.append(_as_row3(mp["vel_noise_density"]))
            att_list.append(_as_row3(mp["att_noise_density"]))
            rate_list.append(_as_row3(mp["rate_noise_density"]))

            vel_p_list.append(_as_scalar1(mp["vel_artifact_prob"]))
            vel_m_list.append(_as_scalar1(mp["vel_artifact_max"]))
            rate_p_list.append(_as_scalar1(mp["rate_artifact_prob"]))
            rate_m_list.append(_as_scalar1(mp["rate_artifact_max"]))

        # Noise densities: (N,3)
        self.pos_density = torch.cat(pos_list, dim=0).to(self.device, self.dtype).abs()
        self.vel_density = torch.cat(vel_list, dim=0).to(self.device, self.dtype).abs()
        self.att_density = torch.cat(att_list, dim=0).to(self.device, self.dtype).abs()
        self.rate_density = torch.cat(rate_list, dim=0).to(self.device, self.dtype).abs()

        # Artifact params: (N,1)
        self.vel_artifact_prob = torch.cat(vel_p_list, dim=0).to(self.device, self.dtype)
        self.vel_artifact_max = torch.cat(vel_m_list, dim=0).to(self.device, self.dtype)
        self.rate_artifact_prob = torch.cat(rate_p_list, dim=0).to(self.device, self.dtype)
        self.rate_artifact_max = torch.cat(rate_m_list, dim=0).to(self.device, self.dtype)

        # with_artifacts mask: (N,1) bool
        if isinstance(with_artifacts, list):
            if len(with_artifacts) != self.num_drones:
                raise ValueError(f"with_artifacts list must have length {self.num_drones}")
            mask = torch.tensor(with_artifacts, device=self.device, dtype=torch.bool)
        elif isinstance(with_artifacts, torch.Tensor):
            mask = with_artifacts.to(self.device, torch.bool)
        else:
            raise TypeError("with_artifacts must be a list[bool] or torch.Tensor")

        if mask.dim() == 1 and mask.shape[0] == self.num_drones:
            mask = mask.view(self.num_drones, 1)
        elif mask.dim() == 2 and mask.shape == (self.num_drones, 1):
            pass
        else:
            raise ValueError(f"with_artifacts must be shape (N,) or (N,1); got {tuple(mask.shape)}")

        self.with_artifacts_mask = mask
        self.initialized = True

    @torch.no_grad()
    def measurement(
        self,
        state: Dict[str, torch.Tensor],
        with_noise: Union[bool, torch.Tensor] = False,       # now supports per-drone mask
        with_artifacts: Union[bool, torch.Tensor] = False,   # now supports per-drone mask
    ) -> Dict[str, torch.Tensor]:
        x = state["x"].to(self.device, self.dtype).clone()
        v = state["v"].to(self.device, self.dtype).clone()
        q = state["q"].to(self.device, self.dtype).clone()
        w = state["w"].to(self.device, self.dtype).clone()

        q = quat_normalize(q)

        N = self.num_drones
        if x.shape[0] != N:
            raise ValueError(f"state batch size {x.shape[0]} != num_drones {N}")

        # ---- canonicalize masks to (N,1) bool tensors ----
        def _mask_ni(flag: Union[bool, torch.Tensor]) -> torch.Tensor:
            if isinstance(flag, bool):
                return torch.full((N, 1), flag, device=self.device, dtype=torch.bool)
            if not isinstance(flag, torch.Tensor):
                raise TypeError("with_noise/with_artifacts must be bool or torch.BoolTensor")
            m = flag.to(self.device, torch.bool)
            if m.dim() == 0:
                return torch.full((N, 1), bool(m.item()), device=self.device, dtype=torch.bool)
            if m.dim() == 1 and m.shape[0] == N:
                return m.view(N, 1)
            if m.dim() == 2 and m.shape == (N, 1):
                return m
            raise ValueError(f"mask must be shape (), (N,), or (N,1); got {tuple(m.shape)}")

        noise_mask = _mask_ni(with_noise)
        artifacts_mask_call = _mask_ni(with_artifacts)

        # ---- per-drone noise ----
        if noise_mask.any():
            # Generate noise for all, then apply mask
            rn = torch.randn((N, 3), device=self.device, dtype=self.dtype)

            x = x + (self.rate_scale * rn * self.pos_density) * noise_mask.to(self.dtype)
            v = v + (self.rate_scale * torch.randn((N, 3), device=self.device, dtype=self.dtype) * self.vel_density) * noise_mask.to(self.dtype)
            w = w + (self.rate_scale * torch.randn((N, 3), device=self.device, dtype=self.dtype) * self.rate_density) * noise_mask.to(self.dtype)

            delta_phi = (self.rate_scale * torch.randn((N, 3), device=self.device, dtype=self.dtype) * self.att_density) * noise_mask.to(self.dtype)
            dq = small_angle_delta_quat(delta_phi)

            q = quat_mul(q, dq)
            q = quat_normalize(q)

        # ---- per-drone artifacts ----
        # Gate by both: per-drone init mask AND per-call mask
        effective_artifacts = self.with_artifacts_mask & artifacts_mask_call

        if effective_artifacts.any():
            vel_spike = (torch.rand((N, 1), device=self.device, dtype=self.dtype) < self.vel_artifact_prob) & effective_artifacts
            rate_spike = (torch.rand((N, 1), device=self.device, dtype=self.dtype) < self.rate_artifact_prob) & effective_artifacts

            vel_axis = torch.randint(0, 3, (N,), device=self.device)
            rate_axis = torch.randint(0, 3, (N,), device=self.device)

            def _signed_uniform(max_val_ni: torch.Tensor) -> torch.Tensor:
                sign = torch.where(torch.rand((N, 1), device=self.device) < 0.5, -1.0, 1.0).to(self.dtype)
                mag = torch.rand((N, 1), device=self.device, dtype=self.dtype) * max_val_ni
                return sign * mag  # (N,1)

            if vel_spike.any():
                dv = torch.zeros((N, 3), device=self.device, dtype=self.dtype)
                dv.scatter_add_(1, vel_axis.view(N, 1), _signed_uniform(self.vel_artifact_max))
                v = v + dv * vel_spike.to(self.dtype)

            if rate_spike.any():
                dw = torch.zeros((N, 3), device=self.device, dtype=self.dtype)
                dw.scatter_add_(1, rate_axis.view(N, 1), _signed_uniform(self.rate_artifact_max))
                w = w + dw * rate_spike.to(self.dtype)

        return {"x": x, "q": q, "v": v, "w": w}


########### Unit test for MotionCapture and BatchedMotionCapture. Plots measurements to verify noise/artifact characteristics visually. #############
if __name__ == "__main__":

    import matplotlib.pyplot as plt

    def merge_dicts(dicts_in):
        """
        Concatenates contents of a list of N state dicts into a single dict by
        prepending a new dimension of size N. This is more convenient for plotting
        and analysis. Requires dicts to have consistent keys and have values that
        are numpy arrays.
        """
        dict_out = {}
        for k in dicts_in[0].keys():
            dict_out[k] = []
            for d in dicts_in:
                dict_out[k].append(d[k])
            dict_out[k] = np.array(dict_out[k])
        return dict_out

    # -------------------------
    # Unit test 1: NumPy MotionCapture (existing)
    # -------------------------
    sim_rate = 1 / 500
    mocap_params = {
        "pos_noise_density": 0.0005 * np.ones((3,)),
        "vel_noise_density": 0.005 * np.ones((3,)),
        "att_noise_density": 0.0005 * np.ones((3,)),
        "rate_noise_density": 0.0005 * np.ones((3,)),
        "vel_artifact_max": 5,
        "vel_artifact_prob": 0.001,
        "rate_artifact_max": 1,
        "rate_artifact_prob": 0.0002,
    }
    sensor = MotionCapture(sampling_rate=sim_rate, mocap_params=mocap_params, with_artifacts=True)

    measurements = []

    state = {"x": np.zeros((3,)), "v": np.zeros((3,)), "q": np.array([0, 0, 0, 1]), "w": np.zeros((3,))}
    for i in range(1000):
        state = {
            "x": np.array(
                [
                    np.sin(2 * np.pi * i / 1000),
                    np.sin(2 * np.pi * i / 1000 - np.pi / 2),
                    np.sin(2 * np.pi * i / 1000 - np.pi / 5),
                ]
            ),
            "v": np.array(
                [
                    np.sin(2 * np.pi * i / 1000),
                    np.sin(2 * np.pi * i / 1000 - np.pi / 2),
                    np.sin(2 * np.pi * i / 1000 - np.pi / 5),
                ]
            ),
            "q": np.array([0, 0, 0, 1]),
            "w": np.array(
                [
                    np.sin(2 * np.pi * i / 1000),
                    np.sin(2 * np.pi * i / 1000 - np.pi / 2),
                    np.sin(2 * np.pi * i / 1000 - np.pi / 5),
                ]
            ),
        }
        current = sensor.measurement(state, with_noise=True, with_artifacts=True)
        measurements.append(current)

    measurements = merge_dicts(measurements)

    x_m = measurements["x"]
    v_m = measurements["v"]
    q_m = measurements["q"]
    w_m = measurements["w"]

    q_norm = np.linalg.norm(q_m, axis=1)

    (fig, axes) = plt.subplots(nrows=4, ncols=1, sharex=True, num="Measurements (NumPy MotionCapture)")
    fig.set_figwidth(9)
    fig.set_figheight(9)

    axe = axes[0]
    axe.plot(x_m[:, 0], "r", markersize=2)
    axe.plot(x_m[:, 1], "g", markersize=2)
    axe.plot(x_m[:, 2], "b", markersize=2)
    axe.set_ylim(bottom=-1.5, top=1.5)

    axe = axes[1]
    axe.plot(v_m[:, 0], "r", markersize=2)
    axe.plot(v_m[:, 1], "g", markersize=2)
    axe.plot(v_m[:, 2], "b", markersize=2)
    axe.set_ylim(bottom=-1.5, top=1.5)

    axe = axes[2]
    axe.plot(q_m[:, 0], "r", markersize=2)
    axe.plot(q_m[:, 1], "g", markersize=2)
    axe.plot(q_m[:, 2], "b", markersize=2)
    axe.plot(q_m[:, 3], "m", markersize=2)
    axe.plot(q_norm, "k", markersize=2)

    axe = axes[3]
    axe.plot(w_m[:, 0], "r", markersize=2)
    axe.plot(w_m[:, 1], "g", markersize=2)
    axe.plot(w_m[:, 2], "b", markersize=2)
    axe.set_ylim(bottom=-1.5, top=1.5)

    # -------------------------
    # Unit test 2: PyTorch BatchedMotionCapture
    # -------------------------
    num_drones = 6
    device = "cuda"  # change to "cuda" if you want
    dtype = torch.float64

    # Per-drone mocap params list (caller provides torch.Tensor values)
    mocap_params_list = []
    base_vel_artifact_prob = 0.001
    for i in range(num_drones):
        mp_i = {
            "pos_noise_density": (0.0005 * torch.ones((3,), dtype=dtype)),
            "vel_noise_density": (0.005 * torch.ones((3,), dtype=dtype)),
            "att_noise_density": (0.0005 * torch.ones((3,), dtype=dtype)),
            "rate_noise_density": (0.0005 * torch.ones((3,), dtype=dtype)),
            "vel_artifact_max": torch.tensor(5.0, dtype=dtype),
            "vel_artifact_prob": torch.tensor(base_vel_artifact_prob * (1.0 + 0.1 * i), dtype=dtype),
            "rate_artifact_max": torch.tensor(1.0, dtype=dtype),
            "rate_artifact_prob": torch.tensor(0.0002, dtype=dtype),
        }
        mocap_params_list.append(mp_i)

    # Per-drone init-time artifact enable mask (persistent capability)
    # Example: enable artifacts capability only for even-index drones
    with_artifacts_init = [(i % 2) == 0 for i in range(num_drones)]

    sensor_t = BatchedMotionCapture(
        num_drones=num_drones,
        sampling_rate=sim_rate,
        mocap_params=mocap_params_list,
        with_artifacts=with_artifacts_init,
        device=device,
        dtype=dtype,
    )

    # Collect history (T, N, dim)
    x_hist = []
    v_hist = []
    q_hist = []
    w_hist = []

    T = 1000

    # Give each drone a different phase offset so traces aren't identical
    phase = torch.linspace(0.0, 2.0 * np.pi, steps=num_drones, device=device, dtype=dtype)
    phase2 = phase + (np.pi / 2)
    phase3 = phase + (np.pi / 5)

    for i in range(T):
        t = torch.tensor(2.0 * np.pi * i / T, device=device, dtype=dtype)

        # Shape (N,)
        s1 = torch.sin(t + phase)
        s2 = torch.sin(t + phase2)
        s3 = torch.sin(t + phase3)

        # Shape (N,3)
        x_true = torch.stack([s1, s2, s3], dim=1)
        v_true = torch.stack([s1, s2, s3], dim=1)

        # Identity quaternion per drone (N,4) [i,j,k,w]
        q_true = torch.zeros((num_drones, 4), device=device, dtype=dtype)
        q_true[:, 3] = 1.0

        w_true = torch.stack([s1, s2, s3], dim=1)

        state_t = {"x": x_true, "v": v_true, "q": q_true, "w": w_true}

        # Per-step per-drone noise/artifact masks (torch.BoolTensor).
        # Example pattern:
        #   - noise ON for all drones for first half, OFF for second half
        #   - artifacts toggled every 100 steps for odd drones, but also gated by with_artifacts_init
        noise_mask = torch.full((num_drones,), (i < (T // 2)), device=device, dtype=torch.bool)

        artifacts_mask = torch.zeros((num_drones,), device=device, dtype=torch.bool)
        toggle_on = ((i // 100) % 2) == 0
        for d in range(num_drones):
            if (d % 2) == 1:
                artifacts_mask[d] = toggle_on

        meas_t = sensor_t.measurement(state_t, with_noise=noise_mask, with_artifacts=artifacts_mask)

        x_hist.append(meas_t["x"].cpu())
        v_hist.append(meas_t["v"].cpu())
        q_hist.append(meas_t["q"].cpu())
        w_hist.append(meas_t["w"].cpu())

    x_hist = torch.stack(x_hist, dim=0).numpy()  # (T,N,3)
    v_hist = torch.stack(v_hist, dim=0).numpy()  # (T,N,3)
    q_hist = torch.stack(q_hist, dim=0).numpy()  # (T,N,4)
    w_hist = torch.stack(w_hist, dim=0).numpy()  # (T,N,3)
    q_norm_hist = np.linalg.norm(q_hist, axis=2)  # (T,N)

    # -------------------------
    # Plot ONE randomly selected drone, same fields as NumPy test
    # -------------------------
    rng_plot = np.random.default_rng(10)
    d = int(rng_plot.integers(low=0, high=num_drones))

    (fig2, axes2) = plt.subplots(nrows=4, ncols=1, sharex=True, num="Measurements (PyTorch BatchedMotionCapture)")
    fig2.set_figwidth(9)
    fig2.set_figheight(9)

    # Position (x,y,z)
    axe = axes2[0]
    axe.plot(x_hist[:, d, 0], "r", markersize=2)
    axe.plot(x_hist[:, d, 1], "g", markersize=2)
    axe.plot(x_hist[:, d, 2], "b", markersize=2)
    axe.set_ylim(bottom=-1.5, top=1.5)
    axe.set_title(f"Drone {d} | init_artifacts={with_artifacts_init[d]}")

    # Velocity (vx,vy,vz)
    axe = axes2[1]
    axe.plot(v_hist[:, d, 0], "r", markersize=2)
    axe.plot(v_hist[:, d, 1], "g", markersize=2)
    axe.plot(v_hist[:, d, 2], "b", markersize=2)
    axe.set_ylim(bottom=-1.5, top=1.5)

    # Quaternion (i,j,k,w) + norm
    axe = axes2[2]
    axe.plot(q_hist[:, d, 0], "r", markersize=2)
    axe.plot(q_hist[:, d, 1], "g", markersize=2)
    axe.plot(q_hist[:, d, 2], "b", markersize=2)
    axe.plot(q_hist[:, d, 3], "m", markersize=2)
    axe.plot(q_norm_hist[:, d], "k", markersize=2)

    # Body rates (wx,wy,wz)
    axe = axes2[3]
    axe.plot(w_hist[:, d, 0], "r", markersize=2)
    axe.plot(w_hist[:, d, 1], "g", markersize=2)
    axe.plot(w_hist[:, d, 2], "b", markersize=2)
    axe.set_ylim(bottom=-1.5, top=1.5)

    plt.show()
