import numpy as np
import torch
import time
import matplotlib.pyplot as plt

from rotorpy.trajectories.minsnap import MinSnap, BatchedMinSnap
from rotorpy.vehicles.multirotor import Multirotor, BatchedMultirotorParams, BatchedMultirotor
from rotorpy.controllers.quadrotor_control import BatchedSE3Control, SE3Control
from rotorpy.vehicles.crazyflie_params import quad_params as cf_quad_params
from rotorpy.vehicles.hummingbird_params import quad_params as hb_quad_params
from rotorpy.utils.trajgen_utils import generate_random_minsnap_traj
from rotorpy.world import World
from rotorpy.wind.default_winds import NoWind, BatchedNoWind
from rotorpy.simulate import simulate, simulate_batch
from rotorpy.sensors.imu import Imu, BatchedImu
from rotorpy.sensors.external_mocap import MotionCapture, BatchedMotionCapture
from rotorpy.estimators.wind_ekf import WindEKF, BatchedWindEKF
from rotorpy.estimators.wind_ukf import WindUKF, BatchedWindUKF

# Helpe func
def convert_params_to_batched(all_params, device='cpu'):
    num_drones = len(all_params)

    # 1. Extract physical parameters into (N, 1) tensors
    # We use a list comprehension to pull the values from your list of dicts
    mass_list = [[d['mass']] for d in all_params]
    cdx_list  = [[d['c_Dx']] for d in all_params]
    cdy_list  = [[d['c_Dy']] for d in all_params]
    cdz_list  = [[d['c_Dz']] for d in all_params]

    # 2. Setup the output dictionary
    quad_params = {
        'mass': torch.tensor(mass_list, device=device).double(),
        'c_Dx': torch.tensor(cdx_list, device=device).double(),
        'c_Dy': torch.tensor(cdy_list, device=device).double(),
        'c_Dz': torch.tensor(cdz_list, device=device).double(),
    }

    # 3. Add Filter Defaults (These are not in all_quad_params)
    # Initial state (N, 9) - Starting level with near-zero velocity
    quad_params['xhat0'] = torch.zeros((num_drones, 9), device=device).double()
    quad_params['xhat0'][:, 3:6] = 0.01 # Prevent singularity in Jacobian

    # Initial Covariance (N, 9, 9)
    quad_params['P0'] = torch.eye(9, device=device).double().repeat(num_drones, 1, 1)

    # Process Noise (N, 9, 9)
    quad_params['Q'] = torch.eye(9, device=device).double().repeat(num_drones, 1, 1) * 0.1

    # Measurement Noise (N, 9, 9)
    quad_params['R'] = torch.eye(9, device=device).double().repeat(num_drones, 1, 1) * 0.01

    return quad_params

# SETTINGS
# Based on if you want to use GPU or CPU for simulation. CPU is faster for smaller batch sizes.
USE_CPU = False  # Toggle this for CPU/GPU testing

def main():
    # This prevents pytorch from spawning multiple threads. Commenting this line out can improve performance
    # on CPU for larger batch sizes. But can also cause issues if you want to do your own multiprocessing,
    # e.g. with the multiprocessing python module.
    torch.set_num_threads(1)

    print(f"==== Batched version ====")

    if USE_CPU:
        device = torch.device("cpu")
    else:
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{torch.cuda.current_device()}")  # typically cuda:0
        else:
            device = torch.device("cpu")


    print(f"Using device: {device}")

    # How many drones to simulate in parallel. Up to a limit, increasing this value increases the efficiency (average FPS)
    # of the simulation. How many you can simulate in parallel efficiently will depend on your machine.
    num_drones = 2000
    dtype = torch.float64

    #### Initial Drone States ####
    # We create initial states for each drone in the batch.
    # In this case, we're setting the initial state of every drone to be basically 0.
    # Note that we're going to be working in torch (as opposed to numpy in standard RotorPy)
    init_rotor_speed = 1788.53
    x0 = {
        'x': torch.zeros(num_drones, 3, device=device, dtype=dtype),
        'v': torch.zeros(num_drones, 3, device=device, dtype=dtype),
        'q': torch.tensor([0, 0, 0, 1], device=device, dtype=dtype).repeat(num_drones, 1),
        'w': torch.zeros(num_drones, 3, device=device, dtype=dtype),
        'wind': torch.zeros(num_drones, 3, device=device, dtype=dtype),
        'rotor_speeds': torch.tensor([init_rotor_speed]*4, device=device, dtype=dtype).repeat(num_drones, 1)
    }

    # Forces the tensors to the exact device object instance.
    for key in x0:
        if isinstance(x0[key], torch.Tensor):
            x0[key] = x0[key].to(device)

    #### Generate Trajectories ####
    # when interfacing with some of the standard rotorpy hardware, you'll have to convert from torch -> numpy.
    world = World({"bounds": {"extents": [-10, 10, -10, 10, -10, 10]}, "blocks": []})
    num_waypoints = 4
    v_avg_des = 2.0
    positions = x0['x'].cpu().numpy()
    trajectories = []

    # Generate the same trajectories each time.
    np.random.seed(10)
    num_done = 0
    while num_done < num_drones:
        traj = generate_random_minsnap_traj(world, num_waypoints, v_avg_des, min_distance=1.0, max_distance=2.0,
                                            start_position=positions[num_done])
        if traj is not None:
            trajectories.append(traj)
            num_done += 1

    # Set to 0 if you want sim results to be more deterministic (default value is 100)
    cf_quad_params["motor_noise_std"] = 0
    hb_quad_params["motor_noise_std"] = 0

    # control_abstraction = "cmd_motor_speeds"  # the default abstraction
    control_abstraction = "cmd_ctatt"

    # We'll simulate half crazyflies, half hummingbirds
    all_quad_params = [cf_quad_params]*(num_drones//2) + [hb_quad_params]*(num_drones//2)

    # Optional: specify feedback gains for each drone in the batch. (can be different for each drone)
    kp_pos = torch.tensor([6.5, 6.5, 15], device=device, dtype=dtype).repeat(num_drones, 1)
    kd_pos = torch.tensor([4.0, 4.0, 9], device=device, dtype=dtype).repeat(num_drones, 1)
    kp_att = torch.tensor([544.0], device=device, dtype=dtype).repeat(num_drones, 1)
    kd_att = torch.tensor([46.64], device=device, dtype=dtype).repeat(num_drones, 1)

    # Collate all the individual MinSnap objects into a single BatchedMinSnap object, which allows us to compute
    # reference commands for all trajectories at the same time.
    batched_trajs = BatchedMinSnap(trajectories, device=device)

    # Define this object which contains dynamics params for each of the drones.
    # If the batch size is large, this can save memory by sharing the dynamics params across the controller and
    # multirotor object.
    batch_params = BatchedMultirotorParams(all_quad_params, num_drones, device)

    # Define a batched controller object which lets us compute control inputs for all drones in the batch at the
    # same time. Note that currently, all drones in the batch must share the same quad_params.
    controller = BatchedSE3Control(batch_params, num_drones, device=device,
                                   kp_pos=kp_pos, kd_pos=kd_pos,
                                   kp_att=kp_att, kd_att=kd_att)

    # Define a batched multirotor, which simulates all drones in the batch simultaneously.
    # Choose 'dopri5' to mimic scipy's default solve_ivp behavior with an adaptive step size, or 'rk4'
    # for a fixed step-size integrator, which is lower-fidelity but much faster.
    vehicle = BatchedMultirotor(batch_params, num_drones, x0, device=device, integrator='dopri5', control_abstraction=control_abstraction)

    # Optional: define when each drone in the batch should terminate.
    dt = 0.01
    t_fs = np.array([trajectory.t_keyframes[-1] for trajectory in trajectories])

    # Define a wind profile -- for batched drones, only NoWind and ConstantWind are supported rn.
    wind_profile = BatchedNoWind(num_drones)

    # Define a BatchedIMU object, which simulates noisy IMU measurements
    batched_imu = BatchedImu(num_drones, device=device)

    # Define a BatchedMotionCapture object
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
    with_artifacts_list = [(i % 2) == 0 for i in range(num_drones)]
    batched_mocap = BatchedMotionCapture(num_drones,
        sampling_rate=int(1/dt),
        mocap_params=mocap_params_list,
        with_artifacts=with_artifacts_list,
        device=device)

    # Define a BatchedWindEKF object
    batched_quad_params = convert_params_to_batched(all_quad_params, device=device)
    batched_wind_ekf = BatchedWindUKF(num_drones, batched_quad_params, device=device)

    # Call the simulate_batch function, which will simulate all drones using the vectorized dynamics.
    sim_fn_start_time = time.time()
    results = simulate_batch(world,
                             x0,
                             vehicle,
                             controller,
                             batched_trajs,
                             wind_profile,
                             batched_imu,
                             batched_mocap,
                             batched_wind_ekf,
                             t_final=t_fs,
                             t_step=dt,
                             safety_margin=0.25,
                             use_mocap=True,
                             print_fps=False)
    sim_fn_end_time = time.time()
    print(f"Time to simulate {num_drones} batched: {sim_fn_end_time - sim_fn_start_time:.4f}s")

    # A dict containing arrays of shape (N, num_drones, ...), where N is the number of timesteps it took for the
    # last drone to terminate. Has the same keys as the state dict returned by the standard simulate() function.
    simulate_fn_states = results[1]

    # Contains the timesteps at which each drone terminated.
    simulate_fn_done_times = results[-1]
    print(f"FPS of batched simulation: {np.sum(simulate_fn_done_times)/(sim_fn_end_time - sim_fn_start_time):.2f}")

    # Contains the exit statuses for each drone.
    exit_statuses = results[-2]

    #### Sequential Simulation ####
    print(f"==== Sequential Version ====")
    # For comparison, we'll also simulate a standard Multirotor.
    x0_single = {'x': np.array([0, 0, 0]),
                 'v': np.zeros(3, ),
                 'q': np.array([0, 0, 0, 1]),  # [i,j,k,w]
                 'w': np.zeros(3, ),
                 'wind': np.array([0, 0, 0]),  # Since wind is handled elsewhere, this value is overwritten
                 'rotor_speeds': np.array([1788.53, 1788.53, 1788.53, 1788.53])}

    all_seq_states = []

    # Initialize MotionCapture object
    mocap_params = {'pos_noise_density': 0.0005*np.ones((3,)),  # noise density for position
                    'vel_noise_density': 0.0010*np.ones((3,)),          # noise density for velocity
                    'att_noise_density': 0.0005*np.ones((3,)),          # noise density for attitude
                    'rate_noise_density': 0.0005*np.ones((3,)),         # noise density for body rates
                    'vel_artifact_max': 5,                              # maximum magnitude of the artifact in velocity (m/s)
                    'vel_artifact_prob': 0.001,                         # probability that an artifact will occur for a given velocity measurement
                    'rate_artifact_max': 1,                             # maximum magnitude of the artifact in body rates (rad/s)
                    'rate_artifact_prob': 0.0002                        # probability that an artifact will occur for a given rate measurement
                    }
    mocap = MotionCapture(sampling_rate=int(1/dt), mocap_params=mocap_params, with_artifacts=False)

    series_start_time = time.time()
    total_time = 0
    total_frames = 0

    for d in range(num_drones):
        controller_single = SE3Control(all_quad_params[d])
        vehicle_single = Multirotor(all_quad_params[d], initial_state=x0_single, control_abstraction=control_abstraction)
        wind_ekf = WindEKF(all_quad_params[d])
        start_time = time.time()
        single_result = simulate(world,
                                 x0_single,
                                 vehicle_single,
                                 controller_single,
                                 trajectories[d],
                                 NoWind(),
                                 Imu(sampling_rate=int(1/dt)),
                                 mocap,
                                 wind_ekf,
                                 trajectories[d].t_keyframes[-1],
                                 dt,
                                 0.25,
                                 use_mocap=False,
                                 print_fps=False)

        all_seq_states.append(single_result[1])
        total_frames += len(single_result[0])
        total_time += time.time() - start_time

    print(f"Time to simulate {num_drones} sequentially: {time.time() - series_start_time:.4f}s")
    print(f"Average FPS of sequential simulation: {total_frames/total_time:.2f}")

    ### Comparison Plots ###
    num_to_plot = 3
    fig, ax = plt.subplots(num_to_plot, 3, figsize=(12, 8))
    fig2, ax2 = plt.subplots(num_to_plot, 3, figsize=(12, 8))

    which_sim = np.random.choice(num_drones, num_to_plot, replace=False)

    def get_data(state_dict, key, drone_idx, end_time, dim):
        """Safely extracts and converts data regardless of device/type."""
        data = state_dict[key][:end_time, int(drone_idx), dim]
        if hasattr(data, 'detach'): # Check if it's a torch tensor
            return data.detach().cpu().numpy()
        return data # Already numpy

    # Plot Positions & Body Rates
    for j, sim_idx in enumerate(which_sim):
        ts = np.arange(trajectories[sim_idx].t_keyframes[-1] + dt, step=dt)
        end_t = simulate_fn_done_times[sim_idx]

        for dimension in range(3):
            # Position Plot (fig 1)
            ax[j][dimension].plot(ts, [trajectories[sim_idx].update(t)['x'][dimension] for t in ts], 'k--', alpha=0.5, label='ref')
            ax[j][dimension].plot(ts, all_seq_states[int(sim_idx)]['x'][:,dimension], label='seq')
            ax[j][dimension].plot(ts, get_data(simulate_fn_states, 'x', sim_idx, end_t, dimension), label='batch')

            # Body Rate Plot (fig 2)
            ax2[j][dimension].plot(ts, all_seq_states[int(sim_idx)]['w'][:,dimension], label='seq')
            ax2[j][dimension].plot(ts, get_data(simulate_fn_states, 'w', sim_idx, end_t, dimension), label='batch')

            if j == 0:
                ax[j][dimension].legend()
                ax2[j][dimension].legend()

    fig.suptitle("Position Comparison")
    fig2.suptitle("Body Rate Comparison")
    plt.show()

if __name__ == "__main__":
    main()
