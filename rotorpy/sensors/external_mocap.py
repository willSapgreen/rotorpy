import numpy as np
from scipy.spatial.transform import Rotation
import copy
import torch
from typing import Dict, Optional

def hat_map(s):
        """
        Given vector s in R^3, return associate skew symmetric matrix S in R^3x3
        """
        return np.array([[    0, -s[2],  s[1]],
                         [ s[2],     0, -s[0]],
                         [-s[1],  s[0],     0]])

class MotionCapture():
    """
    The external motion capture is able to provide pose and twist measurements of the vehicle.
    Given the current ground truth state of the vehicle, it will output noisy measurements of the
    pose and twist. Artifacts can be introduced
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


class BatchedMotionCapture:
    """
    Assumption for measurement():
      state['x']: (num_drones, 3)
      state['v']: (num_drones, 3)
      state['q']: (num_drones, 4) in [i,j,k,w]
      state['w']: (num_drones, 3)
    """

    def __init__(
        self,
        num_drones: int,
        sampling_rate: float,
        mocap_params: Optional[Dict] = None,
        with_artifacts: bool = False,
        device: str = "cpu",
        dtype: torch.dtype = torch.float64,
    ):
        self.device = torch.device(device)
        self.dtype = dtype
        self.num_drones = int(num_drones)
        self.sampling_rate = float(sampling_rate)

        # Matches your original: sqrt(sampling_rate/2)
        self.rate_scale = (self.sampling_rate / 2.0) ** 0.5

        if mocap_params is None:
            mocap_params = {
                "pos_noise_density": 0.0005 * torch.ones((3,)),
                "vel_noise_density": 0.005 * torch.ones((3,)),
                "att_noise_density": 0.0005 * torch.ones((3,)),
                "rate_noise_density": 0.0005 * torch.ones((3,)),
                "vel_artifact_max": 5.0,
                "vel_artifact_prob": 0.001,
                "rate_artifact_max": 1.0,
                "rate_artifact_prob": 0.0002,
            }

        # Noise densities (3,)
        self.pos_density = mocap_params["pos_noise_density"].to(self.device, self.dtype).abs()
        self.vel_density = mocap_params["vel_noise_density"].to(self.device, self.dtype).abs()
        self.att_density = mocap_params["att_noise_density"].to(self.device, self.dtype).abs()
        self.rate_density = mocap_params["rate_noise_density"].to(self.device, self.dtype).abs()

        # Artifacts
        self.vel_artifact_prob = float(mocap_params["vel_artifact_prob"])
        self.vel_artifact_max = float(mocap_params["vel_artifact_max"])
        self.rate_artifact_prob = float(mocap_params["rate_artifact_prob"])
        self.rate_artifact_max = float(mocap_params["rate_artifact_max"])

        self.initialized = True
        self.with_artifacts = bool(with_artifacts)

    @torch.no_grad()
    def measurement(
        self,
        state: Dict[str, torch.Tensor],
        with_noise: bool = False,
        with_artifacts: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Returns dict of torch tensors: {'x','q','v','w'}, each batched over num_drones.
        """
        # Assume always full batch: N == num_drones
        x = state["x"].to(self.device, self.dtype).clone()
        v = state["v"].to(self.device, self.dtype).clone()
        q = state["q"].to(self.device, self.dtype).clone()
        w = state["w"].to(self.device, self.dtype).clone()

        # (Optional but nice) keep quats normalized
        q = quat_normalize(q)

        num_drones = self.num_drones  # equals x.shape[0] by assumption

        if with_noise:
            # Gaussian noise with density * sqrt(fs/2)
            x = x + self.rate_scale * torch.randn((num_drones, 3), device=self.device, dtype=self.dtype) * self.pos_density
            v = v + self.rate_scale * torch.randn((num_drones, 3), device=self.device, dtype=self.dtype) * self.vel_density
            w = w + self.rate_scale * torch.randn((num_drones, 3), device=self.device, dtype=self.dtype) * self.rate_density

            # Attitude noise: small-angle perturbation
            delta_phi = self.rate_scale * torch.randn((num_drones, 3), device=self.device, dtype=self.dtype) * self.att_density
            dq = small_angle_delta_quat(delta_phi)

            # Apply perturbation (same “small angle” spirit as your original)
            q = quat_mul(q, dq)   # if you need dq ⊗ q instead, swap order here
            q = quat_normalize(q)

        if with_artifacts:
            # Decide per drone if a spike happens
            vel_spike = (torch.rand((num_drones,), device=self.device) < self.vel_artifact_prob)
            rate_spike = (torch.rand((num_drones,), device=self.device) < self.rate_artifact_prob)

            # Random axis per drone
            vel_axis = torch.randint(0, 3, (num_drones,), device=self.device)
            rate_axis = torch.randint(0, 3, (num_drones,), device=self.device)

            # Random sign per drone
            vel_sign = torch.where(torch.rand((num_drones,), device=self.device) < 0.5, -1.0, 1.0).to(self.dtype)
            rate_sign = torch.where(torch.rand((num_drones,), device=self.device) < 0.5, -1.0, 1.0).to(self.dtype)

            # Random magnitude per drone
            vel_mag = torch.rand((num_drones,), device=self.device, dtype=self.dtype) * self.vel_artifact_max
            rate_mag = torch.rand((num_drones,), device=self.device, dtype=self.dtype) * self.rate_artifact_max

            # Apply spikes via scatter-add (vectorized)
            if vel_spike.any():
                dv = torch.zeros((num_drones, 3), device=self.device, dtype=self.dtype)
                dv.scatter_add_(1, vel_axis.view(num_drones, 1), (vel_sign * vel_mag).view(num_drones, 1))
                v = v + dv * vel_spike.to(self.dtype).view(num_drones, 1)

            if rate_spike.any():
                dw = torch.zeros((num_drones, 3), device=self.device, dtype=self.dtype)
                dw.scatter_add_(1, rate_axis.view(num_drones, 1), (rate_sign * rate_mag).view(num_drones, 1))
                w = w + dw * rate_spike.to(self.dtype).view(num_drones, 1)

        return {"x": x, "q": q, "v": v, "w": w}


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
    # Unit test 2: PyTorch BatchedMotionCapture (NEW)
    # -------------------------
    num_drones = 6
    device = "cpu"  # change to "cuda" if you want

    mocap_params_torch = {
        "pos_noise_density": 0.0005 * torch.ones((3,)),
        "vel_noise_density": 0.005 * torch.ones((3,)),
        "att_noise_density": 0.0005 * torch.ones((3,)),
        "rate_noise_density": 0.0005 * torch.ones((3,)),
        "vel_artifact_max": 5.0,
        "vel_artifact_prob": 0.001,
        "rate_artifact_max": 1.0,
        "rate_artifact_prob": 0.0002,
    }

    sensor_t = BatchedMotionCapture(
        num_drones=num_drones,
        sampling_rate=sim_rate,
        mocap_params=mocap_params_torch,
        with_artifacts=True,
        device=device,
        dtype=torch.float64,
    )

    # Collect history (T, N, dim)
    x_hist = []
    v_hist = []
    q_hist = []
    w_hist = []

    T = 1000

    # Give each drone a different phase offset so traces aren't identical
    phase = torch.linspace(0.0, 2.0 * np.pi, steps=num_drones, device=device, dtype=torch.float64)
    phase2 = phase + (np.pi / 2)
    phase3 = phase + (np.pi / 5)

    for i in range(T):
        t = torch.tensor(2.0 * np.pi * i / T, device=device, dtype=torch.float64)

        # Shape (N,)
        s1 = torch.sin(t + phase)
        s2 = torch.sin(t + phase2)
        s3 = torch.sin(t + phase3)

        # Shape (N,3)
        x_true = torch.stack([s1, s2, s3], dim=1)
        v_true = torch.stack([s1, s2, s3], dim=1)

        # Identity quaternion per drone (N,4) [i,j,k,w]
        q_true = torch.zeros((num_drones, 4), device=device, dtype=torch.float64)
        q_true[:, 3] = 1.0

        w_true = torch.stack([s1, s2, s3], dim=1)

        state_t = {"x": x_true, "v": v_true, "q": q_true, "w": w_true}

        meas_t = sensor_t.measurement(state_t, with_noise=True, with_artifacts=True)

        x_hist.append(meas_t["x"].cpu())
        v_hist.append(meas_t["v"].cpu())
        q_hist.append(meas_t["q"].cpu())
        w_hist.append(meas_t["w"].cpu())

    x_hist = torch.stack(x_hist, dim=0).numpy()  # (T,N,3)
    v_hist = torch.stack(v_hist, dim=0).numpy()  # (T,N,3)
    q_hist = torch.stack(q_hist, dim=0).numpy()  # (T,N,4)
    w_hist = torch.stack(w_hist, dim=0).numpy()  # (T,N,3)
    q_norm_hist = np.linalg.norm(q_hist, axis=2)  # (T,N)

    # Plot ONE drone's measurements for easy visual comparison (drone 0)
    d = 0
    (fig2, axes2) = plt.subplots(nrows=4, ncols=1, sharex=True, num="Measurements (PyTorch BatchedMotionCapture)")
    fig2.set_figwidth(9)
    fig2.set_figheight(9)

    axe = axes2[0]
    axe.plot(x_hist[:, d, 0], "r", markersize=2)
    axe.plot(x_hist[:, d, 1], "g", markersize=2)
    axe.plot(x_hist[:, d, 2], "b", markersize=2)
    axe.set_ylim(bottom=-1.5, top=1.5)
    axe.set_title(f"Drone {d} (of {num_drones})")

    axe = axes2[1]
    axe.plot(v_hist[:, d, 0], "r", markersize=2)
    axe.plot(v_hist[:, d, 1], "g", markersize=2)
    axe.plot(v_hist[:, d, 2], "b", markersize=2)
    axe.set_ylim(bottom=-1.5, top=1.5)

    axe = axes2[2]
    axe.plot(q_hist[:, d, 0], "r", markersize=2)
    axe.plot(q_hist[:, d, 1], "g", markersize=2)
    axe.plot(q_hist[:, d, 2], "b", markersize=2)
    axe.plot(q_hist[:, d, 3], "m", markersize=2)
    axe.plot(q_norm_hist[:, d], "k", markersize=2)

    axe = axes2[3]
    axe.plot(w_hist[:, d, 0], "r", markersize=2)
    axe.plot(w_hist[:, d, 1], "g", markersize=2)
    axe.plot(w_hist[:, d, 2], "b", markersize=2)
    axe.set_ylim(bottom=-1.5, top=1.5)

    plt.show()
