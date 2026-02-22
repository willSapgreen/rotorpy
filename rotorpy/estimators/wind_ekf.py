import numpy as np
from scipy.spatial.transform import Rotation
import copy


class WindEKF:
    """
    Wind EKF:
        Given approximate dynamics near level flight, the wind EKF will produce an estimate of the local wind vector acting on the body.
        It requires knowledge of the effective drag coefficient on each axis, which would be determined either from real flight or computed in simulation, and the mass of the vehicle.
        The inputs to the filter are the mass normalized collective thrust and the body rates on each axis.
        Measurements of body velocity, Euler angles, and acceleration are provided to the vehicle.
        The filter estimates the Euler angles, body velocities, and wind velocities.

        State space:
            xhat = [phi, theta, psi, u, v, w, windx, windy, windz]
        Input space:
            u = [T/m, p, q, r]
    """

    def __init__(self, quad_params, Q=np.diag(np.concatenate([0.5*np.ones(3),0.7*np.ones(3),0.1*np.ones(3)])),
                                    R=np.diag(np.concatenate([0.0005*np.ones(3),0.0010*np.ones(3),np.sqrt(100/2)*(0.38**2)*np.ones(3)])),
                                    xhat0=np.array([0,0,0, 0.1,0.05,0.02, 1.5,1.5,1.5]),
                                    P0=1*np.eye(9),
                                    dt=1/100):
        """
        Inputs:
            quad_params, dict with keys specified in quadrotor_params.py
            Q, the process noise covariance
            R, the measurement noise covariance
            x0, the initial filter state
            P0, the initial state covariance
            dt, the time between predictions
        """
        # Quadrotor physical parameters.
        # Inertial parameters
        self.mass            = quad_params['mass'] # kg

        # Frame parameters
        self.c_Dx            = quad_params['c_Dx']  # drag coeff, N/(m/s)**2
        self.c_Dy            = quad_params['c_Dy']  # drag coeff, N/(m/s)**2
        self.c_Dz            = quad_params['c_Dz']  # drag coeff, N/(m/s)**2

        self.g = 9.81 # m/s^2

        # Filter parameters
        self.Q = Q
        self.R = R
        self.xhat = xhat0
        self.P = P0

        self.innovation = np.zeros((9,))

        self.dt = dt

        # Initialize the Jacobians at starting position and assuming hover thrust.
        self.computeJacobians(self.xhat, np.array([self.g, 0, 0, 0]))

    def step(self, ground_truth_state, controller_command, imu_measurement, mocap_measurement):
        """
        This will perform both a propagate and update step in one for the sake of readability in other parts of the code.
        """
        self.propagate(ground_truth_state, controller_command)
        self.update(ground_truth_state, controller_command, imu_measurement, mocap_measurement)

        return self.pack_results()

    def propagate(self, ground_truth_state, controller_command):
        """
        Propagate occurs whenever an action u is taken.
        Inputs:
            ground_truth_state, the ground truth state is mainly there if it's necessary to compute certain portions of the state, e.g., actual thrust produced by the rotors.
            controller_command, the controller command taken, this has to be converted to the appropriate control vector u depending on the filter model.

        Outputs:
            xhat, the current state estimate after propagation
            P, the current covariance matrix after propagation
        """

        # Extract the appropriate u vector based on the controller commands.
        uk = self.construct_control_vector(ground_truth_state, controller_command)

        # First propagate the dynamics using the process model
        self.xhat = self.process_model(self.xhat, uk)

        # Update the covariance matrix using the linearized version of the dynamics
        self.computeJacobians(self.xhat, uk)
        self.P = self.A@self.P@(self.A.T) + self.Q

        return self.pack_results

    def update(self, ground_truth_state, controller_command, imu_measurement, mocap_measurement):
        """
        Update the estimate based on new sensor measurments.
        Inputs:
            ground_truth_state, the ground truth state is mainly there if it's necessary to compute certain portions of the state, e.g., actual thrust produced by the rotors.
            controller_command, the controller command taken, this has to be converted to the appropriate control vector u depending on the filter model.
            imu_measurement, contains measurements from an inertial measurement unit. These measurements are noisy, potentially biased, and potentially off-axis. The measurement
                        is specific acceleration, i.e., total force minus gravity.
            mocap_measurement, provides noisy measurements of pose and twist.

        Outputs:
            xhat, the current state estimate after measurement update
            P, the current covariance matrix after measurement update
        """

        # Extract the appropriate u vector based on the controller commands.
        uk = self.construct_control_vector(ground_truth_state, controller_command)

        # Construct the measurement vector yk
        orientation = Rotation.from_quat(copy.deepcopy(mocap_measurement['q']))
        euler_angles = orientation.as_euler('zyx', degrees=False)  # Get Euler angles from current orientation
        inertial_speed = mocap_measurement['v']
        body_speed = (orientation.as_matrix()).T@inertial_speed
        yk = np.array([euler_angles[0],                 # phi
                       euler_angles[1],                 # theta
                       euler_angles[2],                 # psi
                       body_speed[0],                   # vx
                       body_speed[1],                   # vy
                       body_speed[2],                   # vz
                       imu_measurement['accel'][0],     # body x acceleration
                       imu_measurement['accel'][1],     # body y acceleration
                       imu_measurement['accel'][2]      # body z acceleration
                       ])

        # First linearize the measurement model.
        self.computeJacobians(self.xhat, uk)

        # Now compute the Kalman gain
        K = self.P@(self.C.T)@np.linalg.inv(self.C@self.P@(self.C.T) + self.R)

        # Next compute the posteriori distribution
        self.innovation = yk - self.measurement_model(self.xhat, uk)
        self.xhat = self.xhat + K@self.innovation
        self.P = (np.eye(self.xhat.shape[0]) - K@self.C)@self.P

        return self.pack_results()

    def process_model(self, xk, uk):
        """
        Process model
        """

        va = np.sqrt((xk[3]-xk[6])**2 + (xk[4]-xk[7])**2 + (xk[5]-xk[8])**2)  # Compute the norm of the airspeed vector

        # The process model is integrated using forward Euler.
        xdot = np.array([uk[1] + xk[0]*xk[1]*uk[2] + xk[2]*uk[3],
                        uk[2] - xk[0]*uk[3],
                        xk[0]*uk[2] + uk[3],
                        -self.c_Dx/self.mass*(xk[3]-xk[6])*va + self.g*xk[1] + xk[4]*uk[3] - xk[5]*uk[1],
                        -self.c_Dy/self.mass*(xk[4]-xk[7])*va - self.g*xk[0] + xk[5]*uk[1] - xk[3]*uk[3],
                        uk[0] - self.c_Dz/self.mass*(xk[5]-xk[8])*va - self.g + xk[3]*uk[2] - xk[4]*uk[1],
                        0,
                        0,
                        0])

        xkp1 = xk + xdot*self.dt

        return xkp1

    def measurement_model(self, xk, uk):
        """
        Measurement model
        """

        h = np.zeros(xk.shape)

        va = np.sqrt((xk[3]-xk[6])**2 + (xk[4]-xk[7])**2 + (xk[5]-xk[8])**2)  # Compute the norm of the airspeed vector

        h[0:3] = np.hstack((np.eye(3), np.zeros((3,6))))@(xk)
        h[3:6] = np.hstack((np.zeros((3,3)), np.eye(3), np.zeros((3,3))))@(xk)

        h[6:] = np.array([-self.c_Dx/self.mass*(xk[3]-xk[6])*va,
                          -self.c_Dy/self.mass*(xk[4]-xk[7])*va,
                          uk[0]-self.c_Dz/self.mass*(xk[5]-xk[8])*va])

        return h

    def computeJacobians(self, x, u):
        """
        Compute the Jacobians of the process and measurement model at the operating points x and u.
        """

        va = np.sqrt((x[3]-x[6])**2 + (x[4]-x[7])**2 + (x[5]-x[8])**2)  # Compute the norm of the airspeed vector

        # Partial derivatives of va for chain rule
        dvadu = (x[3]-x[6])/va
        dvadv = (x[4]-x[7])/va
        dvadw = (x[5]-x[8])/va
        dvadwx = -dvadu
        dvadwy = -dvadv
        dvadwz = -dvadw

        kx = self.c_Dx/self.mass
        ky = self.c_Dy/self.mass
        kz = self.c_Dz/self.mass

        vax = x[3] - x[6]
        vay = x[4] - x[7]
        vaz = x[5] - x[8]

        self.A = np.array([[x[1]*u[2], x[0]*u[2] + u[3], 0, 0, 0, 0, 0, 0, 0],
                           [-u[3], 0, 0, 0, 0, 0, 0, 0, 0],
                           [ u[2], 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, self.g, 0, -kx*(dvadu*vax + va), u[3] - kx*(dvadv*vax), -u[1] - kx*(dvadw*vax), -kx*(dvadwx*vax-va), -kx*(dvadwy*vax), -kx*(dvadwz*vax)],
                           [-self.g, 0, 0, -ky*(dvadu*vay)-u[3], -ky*(dvadv*vay + va), u[1]-ky*(dvadw*vay), -ky*(dvadwx*vay), -ky*(dvadv*vay - va), -ky*(dvadw*vay)],
                           [0, 0, 0, u[2] - kz*(dvadu*vaz), -u[1] - kz*(dvadv*vaz), -kz*(dvadw*vaz + va), -kz*(dvadwx*vaz), -kz*(dvadwy*vaz), -kz*(dvadwz*vaz - va)],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0]])

        self.C = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 1, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 1, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 1, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 1, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 1, 0, 0, 0],
                           [0, 0, 0, -kx*(dvadu*vax + va),  -kx*(dvadv*vax), -kx*(dvadw*vax), -kx*(dvadwx*vax-va), -kx*(dvadwy*vax), -kx*(dvadwz*vax)],
                           [0, 0, 0, -ky*(dvadu*vay), -ky*(dvadv*vay + va), -ky*(dvadw*vay), -ky*(dvadwx*vay), -ky*(dvadv*vay - va), -ky*(dvadw*vay)],
                           [0, 0, 0, -kz*(dvadu*vaz), -kz*(dvadv*vaz), -kz*(dvadw*vaz + va), -kz*(dvadwx*vaz), -kz*(dvadwy*vaz), -kz*(dvadwz*vaz - va)]])

        return

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

    def pack_results(self):
        # return {'euler_est': self.xhat[0:3], 'v_est': self.xhat[3:6], 'wind_est': self.xhat[6:9],
        #          'covariance': self.P, 'innovation': self.innovation}
        return {'filter_state': self.xhat, 'covariance': self.P}


import torch

class BatchedWindEKF:
    def __init__(self, num_drones, quad_params, dt=1/100, device='cpu'):
        """
        Inputs:
            num_drones: Number of drones in the batch.
            quad_params: Dict of tensors.
                - mass, c_Dx, c_Dy, c_Dz: Shape (num_drones, 1)
                - xhat0: Initial state estimate, Shape (num_drones, 9)
                - P0: Initial state covariance, Shape (num_drones, 9, 9)
                - Q: Process noise covariance, Shape (num_drones, 9, 9)
                - R: Measurement noise covariance, Shape (num_drones, 9, 9)
            dt: The time between predictions.
            device: 'cpu' or 'cuda'.
        """
        self.num_drones = num_drones
        self.device = device
        self.dt = dt
        self.g = 9.81

        # Physical parameters: ensure they are (B, 1) for consistent broadcasting
        self.mass = quad_params['mass'].to(device).double().view(num_drones, 1)
        self.kx = (quad_params['c_Dx'].to(device).double().view(num_drones, 1) / self.mass)
        self.ky = (quad_params['c_Dy'].to(device).double().view(num_drones, 1) / self.mass)
        self.kz = (quad_params['c_Dz'].to(device).double().view(num_drones, 1) / self.mass)

        # Filter state and unique covariances per drone
        self.xhat = quad_params['xhat0'].to(device).double().view(num_drones, 9)
        self.P = quad_params['P0'].to(device).double().view(num_drones, 9, 9)
        self.Q = quad_params['Q'].to(device).double().view(num_drones, 9, 9)
        self.R = quad_params['R'].to(device).double().view(num_drones, 9, 9)

        self.innovation = torch.zeros((num_drones, 9), device=device, dtype=torch.double)

    def step(self, ground_truth_state, controller_command, imu_measurement, mocap_measurement):
        """
        Batched step function to propagate and update the EKF.
        """
        # 1. Prepare control vector uk: [T/m, p, q, r]
        uk = torch.zeros((self.num_drones, 4), device=self.device, dtype=torch.double)

        # Robust thrust handling to fix broadcasting/expansion errors
        thrust = controller_command['cmd_thrust'].view(-1)
        mass = self.mass.view(-1)
        uk[:, 0] = thrust / mass # (B,)

        # Body rates: ensure shape (B, 3)
        uk[:, 1:4] = ground_truth_state['w'].view(self.num_drones, 3)

        # 2. Prepare measurement vector yk: [phi, theta, psi, u, v, w, ax, ay, az]
        yk = torch.zeros((self.num_drones, 9), device=self.device, dtype=torch.double)

        # Convert Quaternions [x, y, z, w] to Euler [yaw, pitch, roll] (matching zyx order)
        q_mocap = mocap_measurement['q'].view(self.num_drones, 4)
        yk[:, 0:3] = self._quat_to_euler_zyx(q_mocap)

        # Rotate inertial velocity to body frame: v_body = R(q).T * v_world
        v_world = mocap_measurement['v'].view(self.num_drones, 3)
        yk[:, 3:6] = self._rotate_to_body(v_world, q_mocap)

        # IMU specific acceleration
        yk[:, 6:9] = imu_measurement['accel'].view(self.num_drones, 3)

        # 3. Filter Propagation and Update
        self.predict(uk)
        self.update(yk, uk)

        return {'filter_state': self.xhat, 'covariance': self.P}

    def predict(self, uk):
        Ad, _ = self.compute_jacobians(self.xhat, uk)
        self.xhat = self.xhat + (self.compute_xdot(self.xhat, uk) * self.dt)
        self.P = torch.bmm(torch.bmm(Ad, self.P), Ad.transpose(1, 2)) + self.Q

    def update(self, yk, uk):
        _, C = self.compute_jacobians(self.xhat, uk)
        self.innovation = yk - self.measurement_model(self.xhat, uk)

        # S = C*P*C^T + R
        S = torch.bmm(torch.bmm(C, self.P), C.transpose(1, 2)) + self.R
        # K = P*C^T * inv(S)
        K = torch.bmm(torch.bmm(self.P, C.transpose(1, 2)), torch.inverse(S))

        # x = x + K*innovation
        self.xhat = self.xhat + torch.bmm(K, self.innovation.unsqueeze(-1)).squeeze(-1)
        # P = (I - K*C)*P
        eye = torch.eye(9, device=self.device, dtype=torch.double).repeat(self.num_drones, 1, 1)
        self.P = torch.bmm(eye - torch.bmm(K, C), self.P)

    def compute_xdot(self, x, u):
        vax, vay, vaz = x[:, 3]-x[:, 6], x[:, 4]-x[:, 7], x[:, 5]-x[:, 8]
        va = torch.sqrt(vax**2 + vay**2 + vaz**2 + 1e-6)
        kx, ky, kz = self.kx.view(-1), self.ky.view(-1), self.kz.view(-1)

        xdot = torch.zeros_like(x)
        # Euler angle kinematics (Small angle approximation matching original code)
        xdot[:, 0] = u[:, 1] + x[:, 0] * x[:, 1] * u[:, 2] + x[:, 2] * u[:, 3]
        xdot[:, 1] = u[:, 2] - x[:, 0] * u[:, 3]
        xdot[:, 2] = x[:, 0] * u[:, 2] + u[:, 3]

        # Body frame accelerations
        xdot[:, 3] = -kx * vax * va + self.g * x[:, 1] + x[:, 4] * u[:, 3] - x[:, 5] * u[:, 1]
        xdot[:, 4] = -ky * vay * va - self.g * x[:, 0] + x[:, 5] * u[:, 1] - x[:, 3] * u[:, 3]
        xdot[:, 5] = u[:, 0] - kz * vaz * va - self.g + x[:, 3] * u[:, 2] - x[:, 4] * u[:, 1]
        return xdot

    def measurement_model(self, x, u):
        h = torch.zeros_like(x)
        vax, vay, vaz = x[:, 3]-x[:, 6], x[:, 4]-x[:, 7], x[:, 5]-x[:, 8]
        va = torch.sqrt(vax**2 + vay**2 + vaz**2 + 1e-6)
        kx, ky, kz = self.kx.view(-1), self.ky.view(-1), self.kz.view(-1)

        h[:, 0:6] = x[:, 0:6]
        h[:, 6] = -kx * vax * va
        h[:, 7] = -ky * vay * va
        h[:, 8] = u[:, 0] - kz * vaz * va
        return h

    def compute_jacobians(self, x, u):
        B = self.num_drones
        vax, vay, vaz = x[:, 3]-x[:, 6], x[:, 4]-x[:, 7], x[:, 5]-x[:, 8]
        va = torch.sqrt(vax**2 + vay**2 + vaz**2 + 1e-6)
        dvadu, dvadv, dvadw = vax/va, vay/va, vaz/va
        kx, ky, kz = self.kx.view(-1), self.ky.view(-1), self.kz.view(-1)

        A = torch.zeros((B, 9, 9), device=self.device, dtype=torch.double)
        A[:, 0, 0], A[:, 0, 1] = x[:, 1] * u[:, 2], x[:, 0] * u[:, 2] + u[:, 3]
        A[:, 1, 0], A[:, 2, 0] = -u[:, 3], u[:, 2]

        A[:, 3, 1] = self.g
        A[:, 3, 3] = -kx * (dvadu * vax + va)
        A[:, 3, 4] = u[:, 3] - kx * dvadv * vax
        A[:, 3, 5] = -u[:, 1] - kx * dvadw * vax
        A[:, 3, 6], A[:, 3, 7], A[:, 3, 8] = kx * (dvadu * vax + va), kx * dvadv * vax, kx * dvadw * vax

        A[:, 4, 0] = -self.g
        A[:, 4, 3], A[:, 4, 4], A[:, 4, 5] = -ky * dvadu * vay - u[:, 3], -ky * (dvadv * vay + va), u[:, 1] - ky * dvadw * vay
        A[:, 4, 6], A[:, 4, 7], A[:, 4, 8] = ky * dvadu * vay, ky * (dvadv * vay + va), ky * dvadw * vay

        A[:, 5, 3], A[:, 5, 4], A[:, 5, 5] = u[:, 2] - kz * dvadu * vaz, -u[:, 1] - kz * dvadv * vaz, -kz * (dvadw * vaz + va)
        A[:, 5, 6], A[:, 5, 7], A[:, 5, 8] = kz * dvadu * vaz, kz * dvadv * vaz, kz * (dvadw * vaz + va)

        Ad = torch.eye(9, device=self.device, dtype=torch.double).repeat(B, 1, 1) + A * self.dt
        C = torch.zeros((B, 9, 9), device=self.device, dtype=torch.double)
        C[:, 0:6, 0:6] = torch.eye(6, device=self.device, dtype=torch.double)
        C[:, 6:9, 3:9] = A[:, 3:6, 3:9] / self.dt
        return Ad, C

    def _quat_to_euler_zyx(self, q):
        # q: (B, 4) in [x, y, z, w]
        x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        # Yaw (z), Pitch (y), Roll (x)
        yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))
        pitch = torch.asin(torch.clamp(2 * (w * y - z * x), -1.0, 1.0))
        roll = torch.atan2(2 * (w * x + y * z), 1 - 2 * (x**2 + y**2))
        return torch.stack([yaw, pitch, roll], dim=1)

    def _rotate_to_body(self, v_world, q):
        # Rotates v_world to body frame using conjugate of q: [x, y, z, w]
        x, y, z, w = q[:, 0:1], q[:, 1:2], q[:, 2:3], q[:, 3:4]
        q_inv_vec = -torch.cat([x, y, z], dim=1)
        qv = torch.cross(q_inv_vec, v_world, dim=1)
        return v_world + 2.0 * torch.cross(q_inv_vec, qv + w * v_world, dim=1)
