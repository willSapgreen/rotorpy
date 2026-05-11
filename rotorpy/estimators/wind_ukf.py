import numpy as np
from scipy.spatial.transform import Rotation
import copy

try:
    from filterpy.kalman import UnscentedKalmanFilter
    from filterpy.kalman import MerweScaledSigmaPoints
except ImportError:
    pass

"""
The Wind UKF uses the same model as the EKF found in wind_ekf.py, but instead applies the Unscented Kalman Filter. The benefit
of this approach is the accuracy of the UKF is 3rd order (compared to EKF's 1st order), and Jacobians do not need to be computed.
"""
class WindUKF:
    """
    WindUKF
        Given approximate dynamics near level flight, the wind EKF will produce an estimate of the local wind vector acting on the body.
        It requires knowledge of the effective drag coefficient on each axis, which would be determined either from real flight or computed in simulation, and the mass of the vehicle.
        The inputs to the filter are the mass normalized collective thrust and the body rates on each axis.
        Measurements of body velocity, Euler angles, and acceleration are provided to the vehicle.
        The filter estimates the Euler angles, body velocities, and wind velocities.

        State space:
            xhat = [psi, theta, phi, u, v, w, windx, windy, windz]
        Measurement space:
            u = [T/m, p, q, r]
    """

    def __init__(self, quad_params,
                       Q=np.diag(np.concatenate([0.05*np.ones(3),0.07*np.ones(3),0.01*np.ones(3)])),
                       R=np.diag(np.concatenate([0.0005*np.ones(3),0.0010*np.ones(3),np.sqrt(100/2)*(0.38**2)*np.ones(3)])),
                       xhat0=np.array([0,0,0, 0.1,0.05,0.02, 1.5,1.5,1.5]),
                       P0=1*np.eye(9),
                       dt=1/100,
                       alpha=0.1,
                       beta=2.0,
                       kappa=-1):
        """
        Inputs:
            quad_params, dict with keys specified in quadrotor_params.py
            Q, the process noise covariance
            R, the measurement noise covariance
            x0, the initial filter state
            P0, the initial state covariance
            dt, the time between predictions
        """

        self.mass            = quad_params['mass'] # kg

        # Frame parameters
        self.c_Dx            = quad_params['c_Dx']  # drag coeff, N/(m/s)**2
        self.c_Dy            = quad_params['c_Dy']  # drag coeff, N/(m/s)**2
        self.c_Dz            = quad_params['c_Dz']  # drag coeff, N/(m/s)**2

        self.g = 9.81 # m/s^2

        # Filter parameters
        self.xhat = xhat0
        self.P = P0

        self.dt = dt

        self.N = self.xhat.shape[0]

        self.points = MerweScaledSigmaPoints(self.N, alpha=alpha, beta=beta, kappa=kappa)

        self.uk = np.array([self.g, 0, 0, 0])

        self.filter = UnscentedKalmanFilter(dim_x=self.N, dim_z=self.N, dt=dt, fx=self.f, hx=self.h, points=self.points)
        self.filter.x = xhat0
        self.filter.P = P0
        self.filter.R = R
        self.filter.Q = Q


    def step(self, ground_truth_state, controller_command, imu_measurement, mocap_measurement):
        """
        The step command will update the filter based on the following.
        Inputs:
            ground_truth_state, the ground truth state is mainly there if it's necessary to compute certain portions of the state, e.g., actual thrust produced by the rotors.
            controller_command, the controller command taken, this has to be converted to the appropriate control vector u depending on the filter model.
            imu_measurement, contains measurements from an inertial measurement unit. These measurements are noisy, potentially biased, and potentially off-axis. The measurement
                        is specific acceleration, i.e., total force minus gravity.
            mocap_measurement, provides noisy measurements of pose and twist.

        Outputs:
            A dictionary with the following keys:
                filter_state, containing the current filter estimate.
                covariance, containing the current covariance matrix.

        The ground truth state is supplied in case you would like to have "knowns" in your filter, or otherwise manipulate the state to create a custom measurement of your own desires.
        IMU measurements are commonly used in filter setups, so we already supply these measurements as an input into the system.
        Motion capture measurements are useful if you want noisy measurements of the pose and twist of the vehicle.
        """

        # Extract the appropriate u vector based on the controller commands.
        self.uk = self.construct_control_vector(ground_truth_state, controller_command)

        # Construct the measurement vector yk
        orientation = Rotation.from_quat(copy.deepcopy(mocap_measurement['q']))
        euler_angles = orientation.as_euler('zyx', degrees=False)  # Get Euler angles from current orientation
        inertial_speed = mocap_measurement['v']
        body_speed = (orientation.as_matrix()).T@inertial_speed
        zk = np.array([euler_angles[0],                 # phi
                       euler_angles[1],                 # theta
                       euler_angles[2],                 # psi
                       body_speed[0],                   # vx
                       body_speed[1],                   # vy
                       body_speed[2],                   # vz
                       imu_measurement['accel'][0],     # body x acceleration
                       imu_measurement['accel'][1],     # body y acceleration
                       imu_measurement['accel'][2]      # body z acceleration
                       ])

        self.filter.predict()
        self.filter.update(zk)

        return {'filter_state': self.filter.x, 'covariance': self.filter.P}

    def f(self, xk, dt):
        """
        Process model
        """

        va = np.sqrt((xk[3]-xk[6])**2 + (xk[4]-xk[7])**2 + (xk[5]-xk[8])**2)  # Compute the norm of the airspeed vector

        # The process model is integrated using forward Euler. Below assumes Euler angles are given in order of [phi, theta, psi] (XYZ)
        # xdot = np.array([self.uk[1] + xk[0]*xk[1]*self.uk[2] + xk[2]*self.uk[3],
        #                 self.uk[2] - xk[0]*self.uk[3],
        #                 xk[0]*self.uk[2] + self.uk[3],
        #                 -self.c_Dx/self.mass*(xk[3]-xk[6])*va + self.g*xk[1] + xk[4]*self.uk[3] - xk[5]*self.uk[1],
        #                 -self.c_Dy/self.mass*(xk[4]-xk[7])*va - self.g*xk[0] + xk[5]*self.uk[1] - xk[3]*self.uk[3],
        #                 self.uk[0] - self.c_Dz/self.mass*(xk[5]-xk[8])*va - self.g + xk[3]*self.uk[2] - xk[4]*self.uk[1],
        #                 0,
        #                 0,
        #                 0])

        # The process model, below assumes Euler angles are given in the order of [psi, theta, phi] (ZYX)
        xdot = np.array([xk[2]*self.uk[2] + self.uk[3],
                        self.uk[2] - xk[2]*self.uk[3],
                        self.uk[1] + xk[2]*xk[1]*self.uk[2] + xk[0]*self.uk[3],
                        -self.c_Dx/self.mass*(xk[3]-xk[6])*va + self.g*xk[1] + xk[4]*self.uk[3] - xk[5]*self.uk[1],
                        -self.c_Dy/self.mass*(xk[4]-xk[7])*va - self.g*xk[2] + xk[5]*self.uk[1] - xk[3]*self.uk[3],
                        self.uk[0] - self.c_Dz/self.mass*(xk[5]-xk[8])*va - self.g + xk[3]*self.uk[2] - xk[4]*self.uk[1],
                        0,
                        0,
                        0])

        xkp1 = xk + xdot*dt

        return xkp1

    def h(self, xk):
        """
        Measurement model
        """

        h = np.zeros(xk.shape)

        va = np.sqrt((xk[3]-xk[6])**2 + (xk[4]-xk[7])**2 + (xk[5]-xk[8])**2)  # Compute the norm of the airspeed vector

        h[0:3] = np.hstack((np.eye(3), np.zeros((3,6))))@(xk)
        h[3:6] = np.hstack((np.zeros((3,3)), np.eye(3), np.zeros((3,3))))@(xk)

        h[6:] = np.array([-self.c_Dx/self.mass*(xk[3]-xk[6])*va,
                          -self.c_Dy/self.mass*(xk[4]-xk[7])*va,
                          self.uk[0]-self.c_Dz/self.mass*(xk[5]-xk[8])*va])

        return h

    def construct_control_vector(self, ground_truth_state, controller_command):
        """
        Constructs control vector
        """
        uk = np.array([controller_command['cmd_thrust']/self.mass,    # Compute mass normalized thrust from the command thrust, note that this is not the actual thrust...
                       ground_truth_state['w'][0],                      # Body rate in x axis
                       ground_truth_state['w'][1],                      # Body rate in y axis
                       ground_truth_state['w'][2]]                      # Body rate in z axis
                       )

        return uk


import torch

class BatchedWindUKF:
    def __init__(self, num_drones, quad_params, dt=1/100, device='cpu'):
        """
        Inputs:
            num_drones: Number of drones in the batch.
            quad_params: Dict of tensors.
                - mass, c_Dx, c_Dy, c_Dz: Shape (num_drones, 1)
                - alpha, beta, kappa: Shape (num_drones, 1)
                - xhat0: Initial state estimate [psi, theta, phi, u, v, w, wx, wy, wz], Shape (num_drones, 9)
                - P0: Initial state covariance, Shape (num_drones, 9, 9)
                - Q: Process noise covariance, Shape (num_drones, 9, 9)
                - R: Measurement noise covariance, Shape (num_drones, 9, 9)
            dt: The time between predictions.
            device: 'cpu' or 'cuda'.
        """
        self.num_drones = num_drones
        self.device = device
        self.dt = dt
        self.n = 9  # State dimension
        self.num_sigmas = 2 * self.n + 1
        self.g = 9.81

        # Physical parameters (B, 1)
        self.mass = quad_params['mass'].to(device).double().view(num_drones, 1)
        self.kx = quad_params['c_Dx'].to(device).double().view(num_drones, 1) / self.mass
        self.ky = quad_params['c_Dy'].to(device).double().view(num_drones, 1) / self.mass
        self.kz = quad_params['c_Dz'].to(device).double().view(num_drones, 1) / self.mass

        # Filter state and unique covariances per drone
        self.xhat = quad_params['xhat0'].to(device).double().view(num_drones, 9)
        self.P = quad_params['P0'].to(device).double().view(num_drones, 9, 9)
        self.Q = quad_params['Q'].to(device).double().view(num_drones, 9, 9)
        self.R = quad_params['R'].to(device).double().view(num_drones, 9, 9)

        # UKF Scaling Parameters (B, 1)
        alpha = quad_params.get('alpha', torch.full((num_drones, 1), 0.1, device=device)).double()
        beta = quad_params.get('beta', torch.full((num_drones, 1), 2.0, device=device)).double()
        kappa = quad_params.get('kappa', torch.full((num_drones, 1), -1.0, device=device)).double()

        # Compute batched weights
        lambd = alpha**2 * (self.n + kappa) - self.n  # (B, 1)

        # Mean weights (B, num_sigmas)
        self.w_m = torch.full((num_drones, self.num_sigmas), 0.0, device=device, dtype=torch.double)
        self.w_m[:, 0:1] = lambd / (self.n + lambd)
        self.w_m[:, 1:] = 1.0 / (2.0 * (self.n + lambd))

        # Covariance weights (B, num_sigmas)
        self.w_c = self.w_m.clone()
        self.w_c[:, 0:1] = self.w_m[:, 0:1] + (1.0 - alpha**2 + beta)

        # Scaling factor for sigma point generation (B, 1)
        self.gamma = torch.sqrt(self.n + lambd)

    def step(self, ground_truth_state, controller_command, imu_measurement, mocap_measurement):
        """
        One-step propagate and update.
        """
        # 1. Prepare control vector uk: [T/m, p, q, r]
        uk = torch.zeros((self.num_drones, 4), device=self.device, dtype=torch.double)
        uk[:, 0] = (controller_command['cmd_thrust'].view(-1) / self.mass.view(-1))
        uk[:, 1:4] = ground_truth_state['w'].view(self.num_drones, 3)

        # 2. Prepare measurement vector yk: [psi, theta, phi, u, v, w, ax, ay, az]
        yk = torch.zeros((self.num_drones, 9), device=self.device, dtype=torch.double)
        q_mocap = mocap_measurement['q'].view(self.num_drones, 4)

        # State order matches WindUKF.f(): [psi, theta, phi]
        yk[:, 0:3] = self._quat_to_euler_zyx(q_mocap)
        yk[:, 3:6] = self._rotate_to_body(mocap_measurement['v'].view(self.num_drones, 3), q_mocap)
        yk[:, 6:9] = imu_measurement['accel'].view(self.num_drones, 3)

        # 3. Filter steps
        self.predict(uk)
        self.update(yk, uk)

        return {'filter_state': self.xhat, 'covariance': self.P}

    def generate_sigma_points(self, x, P):
        """Generates (B, 2n+1, n) sigma points using batched Cholesky."""
        B = x.shape[0]
        # Ensure positive definiteness for Cholesky
        eps = 1e-9 * torch.eye(self.n, device=self.device, dtype=torch.double)
        L = torch.linalg.cholesky(P + eps) # (B, n, n)

        sigmas = torch.zeros((B, self.num_sigmas, self.n), device=self.device, dtype=torch.double)
        sigmas[:, 0, :] = x

        # Scaling matrix: gamma * L
        scaled_L = self.gamma.unsqueeze(-1) * L # (B, n, n)

        for i in range(self.n):
            sigmas[:, i + 1, :] = x + scaled_L[:, :, i]
            sigmas[:, i + 1 + self.n, :] = x - scaled_L[:, :, i]
        return sigmas

    def predict(self, uk):
        sigmas = self.generate_sigma_points(self.xhat, self.P)

        # Propagate sigmas through dynamics
        sigmas_flat = sigmas.view(-1, self.n)
        uk_expanded = uk.repeat_interleave(self.num_sigmas, dim=0)

        sigmas_f = self.compute_dynamics(sigmas_flat, uk_expanded)
        sigmas_f = sigmas_f.view(self.num_drones, self.num_sigmas, self.n)

        # Predict Mean: x = sum(w_m * sigmas_f)
        self.xhat = torch.einsum('bs,bsn->bn', self.w_m, sigmas_f)

        # Predict Covariance: P = sum(w_c * (f - x)(f - x)^T) + Q
        y = sigmas_f - self.xhat.unsqueeze(1)
        self.P = torch.einsum('bs,bsn,bsm->bnm', self.w_c, y, y) + self.Q

    def update(self, zk, uk):
        sigmas = self.generate_sigma_points(self.xhat, self.P)

        # Sigmas through measurement model
        sigmas_flat = sigmas.view(-1, self.n)
        uk_expanded = uk.repeat_interleave(self.num_sigmas, dim=0)

        sigmas_h = self.compute_measurement(sigmas_flat, uk_expanded)
        sigmas_h = sigmas_h.view(self.num_drones, self.num_sigmas, self.n)

        # Measurement mean and covariances
        z_mean = torch.einsum('bs,bsn->bn', self.w_m, sigmas_h)
        dz = sigmas_h - z_mean.unsqueeze(1)
        dx = sigmas - self.xhat.unsqueeze(1)

        S = torch.einsum('bs,bsn,bsm->bnm', self.w_c, dz, dz) + self.R # (B, n, n)
        Pxz = torch.einsum('bs,bsn,bsm->bnm', self.w_c, dx, dz)       # (B, n, n)

        # Kalman Gain and Posteriori
        K = torch.bmm(Pxz, torch.inverse(S))
        self.xhat = self.xhat + torch.bmm(K, (zk - z_mean).unsqueeze(-1)).squeeze(-1)
        self.P = self.P - torch.bmm(K, torch.bmm(S, K.transpose(1, 2)))

    def compute_dynamics(self, xk, uk):
        """Matches WindUKF.f() ZYX order logic."""
        vax, vay, vaz = xk[:, 3]-xk[:, 6], xk[:, 4]-xk[:, 7], xk[:, 5]-xk[:, 8]
        va = torch.sqrt(vax**2 + vay**2 + vaz**2 + 1e-6)

        kx = self.kx.repeat_interleave(self.num_sigmas, 0).view(-1)
        ky = self.ky.repeat_interleave(self.num_sigmas, 0).view(-1)
        kz = self.kz.repeat_interleave(self.num_sigmas, 0).view(-1)

        xdot = torch.zeros_like(xk)
        # ZYX Order: psi, theta, phi
        xdot[:, 0] = xk[:, 2]*uk[:, 2] + uk[:, 3]
        xdot[:, 1] = uk[:, 2] - xk[:, 2]*uk[:, 3]
        xdot[:, 2] = uk[:, 1] + xk[:, 2]*xk[:, 1]*uk[:, 2] + xk[:, 0]*uk[:, 3]

        # Body Velocities u, v, w
        xdot[:, 3] = -kx * vax * va + self.g * xk[:, 1] + xk[:, 4] * uk[:, 3] - xk[:, 5] * uk[:, 1]
        xdot[:, 4] = -ky * vay * va - self.g * xk[:, 2] + xk[:, 5] * uk[:, 1] - xk[:, 3] * uk[:, 3]
        xdot[:, 5] = uk[:, 0] - kz * vaz * va - self.g + xk[:, 3] * uk[:, 2] - xk[:, 4] * uk[:, 1]

        return xk + xdot * self.dt

    def compute_measurement(self, xk, uk):
        """Matches WindUKF.h() logic."""
        vax, vay, vaz = xk[:, 3]-xk[:, 6], xk[:, 4]-xk[:, 7], xk[:, 5]-xk[:, 8]
        va = torch.sqrt(vax**2 + vay**2 + vaz**2 + 1e-6)

        kx = self.kx.repeat_interleave(self.num_sigmas, 0).view(-1)
        ky = self.ky.repeat_interleave(self.num_sigmas, 0).view(-1)
        kz = self.kz.repeat_interleave(self.num_sigmas, 0).view(-1)

        h = torch.zeros_like(xk)
        h[:, 0:6] = xk[:, 0:6]
        h[:, 6] = -kx * vax * va
        h[:, 7] = -ky * vay * va
        h[:, 8] = uk[:, 0] - kz * vaz * va
        return h

    def _quat_to_euler_zyx(self, q):
        # q: [x, y, z, w], returns [psi (yaw), theta (pitch), phi (roll)]
        x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        psi = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))
        theta = torch.asin(torch.clamp(2 * (w * y - z * x), -1.0, 1.0))
        phi = torch.atan2(2 * (w * x + y * z), 1 - 2 * (x**2 + y**2))
        return torch.stack([psi, theta, phi], dim=1)

    def _rotate_to_body(self, v_world, q):
        # v_body = R(q).T * v_world. Uses conjugate of q: [x, y, z, w]
        x, y, z, w = q[:, 0:1], q[:, 1:2], q[:, 2:3], q[:, 3:4]
        q_inv_vec = -torch.cat([x, y, z], dim=1)
        qv = torch.cross(q_inv_vec, v_world, dim=1)
        return v_world + 2.0 * torch.cross(q_inv_vec, qv + w * v_world, dim=1)
