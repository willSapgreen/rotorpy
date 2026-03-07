"""
basic_usage_swarm.py
Test script for DOOT coordinator with RotorPy swarm simulation.
"""


import os
import argparse
import numpy as np
import torch

from rotorpy.environments import EnvironmentBatch
from rotorpy.world import World

from rotorpy.vehicles.multirotor import BatchedMultirotorParams, BatchedMultirotor
from rotorpy.vehicles.crazyflie_params import quad_params
from rotorpy.controllers.quadrotor_control import BatchedSE3Control
from rotorpy.trajectories.velocity_reference import BatchedVelocityReference

from rotorpy.wind.default_winds import BatchedNoWind
from rotorpy.sensors.imu import BatchedImu
from rotorpy.sensors.external_mocap import BatchedMotionCapture
from rotorpy.estimators.wind_ekf import BatchedWindEKF

from rotorpy.coordinators.doot_cbf_coordinator import (
    BatchedDootCbfCoordinator,
    DootConfig,
)


def get_world_extents(world_obj) -> np.ndarray:
    """
    Return extents as np.array([xmin, xmax, ymin, ymax, zmin, zmax], dtype=float).

    RotorPy variants seen:
    - world.world is a dict containing {"bounds": {"extents": [...]}}
    - (older) world.bounds is a dict with key "extents"
    - (older) world.bounds is an object with attribute "extents"
    """
    # Preferred (your version): world.world["bounds"]["extents"]
    if hasattr(world_obj, "world") and isinstance(world_obj.world, dict):
        w = world_obj.world
        if "bounds" in w and isinstance(w["bounds"], dict) and "extents" in w["bounds"]:
            return np.asarray(w["bounds"]["extents"], dtype=float)

    # Fallbacks
    b = getattr(world_obj, "bounds", None)
    if b is not None:
        if isinstance(b, dict) and "extents" in b:
            return np.asarray(b["extents"], dtype=float)
        if hasattr(b, "extents"):
            return np.asarray(getattr(b, "extents"), dtype=float)

    raise RuntimeError(
        "Cannot find world extents. Expected world.world['bounds']['extents'] "
        "or world.bounds['extents'] or world.bounds.extents."
    )


def rejection_sample_xy(
    rng: np.random.Generator,
    num: int,
    mean_xy: np.ndarray,
    std_xy: np.ndarray,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    batch_size: int = 2048,
) -> np.ndarray:
    """
    Rejection-sample 2D Gaussian points until `num` samples fall inside the box bounds.
    Returns (num, 2).
    """
    out = np.empty((0, 2), dtype=float)
    while out.shape[0] < num:
        batch = rng.normal(loc=mean_xy, scale=std_xy, size=(batch_size, 2))
        ok = (
            (batch[:, 0] >= x_min)
            & (batch[:, 0] <= x_max)
            & (batch[:, 1] >= y_min)
            & (batch[:, 1] <= y_max)
        )
        if np.any(ok):
            out = np.vstack([out, batch[ok]])
    return out[:num, :]


def sample_gmm_targets_xy_in_bounds(
    rng: np.random.Generator,
    num_samples: int,
    num_gmm_components: int,
    mean_des: np.ndarray,        # (num_gmm_components, 2)
    std_des: np.ndarray,         # (2,)
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    batch_size: int = 4096,
) -> np.ndarray:
    """
    Draw `num_samples` samples from an equal-weight 2D Gaussian mixture:
        k ~ Uniform{0..num_gmm_components-1}
        x ~ N(mean_des[k], diag(std_des^2))
    Rejection-sample so that returned samples lie inside [x_min, x_max] x [y_min, y_max].
    Returns (num_samples, 2).
    """
    out = np.empty((0, 2), dtype=float)
    while out.shape[0] < num_samples:
        # Choose num_gmm_components uniformly (equal weights)
        comp_ids = rng.integers(low=0, high=num_gmm_components, size=batch_size)

        # Vectorized Gaussian draw with diagonal covariance:
        # sample = mean_des[comp_id] + std_des * N(0,1)
        noise = rng.standard_normal((batch_size, 2)) * std_des.reshape(1, 2)
        batch = mean_des[comp_ids, :] + noise

        ok = (
            (batch[:, 0] >= x_min)
            & (batch[:, 0] <= x_max)
            & (batch[:, 1] >= y_min)
            & (batch[:, 1] <= y_max)
        )
        if np.any(ok):
            out = np.vstack([out, batch[ok]])

    return out[:num_samples, :]


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


def write_basic_usage_swarm_2d_video(
    results,
    vis_output_name,
    x_min, x_max, y_min, y_max,
    num_vehicles,
    samples_xy,
    rng,
):
    """
    Batched results format:
      results["state"]["x"] : (T, B, 3) numpy
    """
    import os
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt

    # ===================== Output path =====================
    out_dir = os.path.dirname(os.path.abspath(__file__))
    avi_path = os.path.join(out_dir, vis_output_name)

    # ===================== Visualization bounds =====================
    vis_pad = 0.5
    vis_x_min, vis_x_max = x_min - vis_pad, x_max + vis_pad
    vis_y_min, vis_y_max = y_min - vis_pad, y_max + vis_pad

    # ===================== Extract trajectories =====================
    X = np.asarray(results["state"]["x"], dtype=float)  # (T,B,3)
    T, B, _ = X.shape
    if num_vehicles != B:
        # Keep behavior deterministic and explicit
        raise ValueError(f"num_vehicles={num_vehicles} but results has B={B} vehicles.")

    traj_xy = X[:, :, :2]  # (T,B,2)
    num_frames = T

    print(f"[basic_usage_swarm] batched traj: T={T}, B={B}")

    # Pre-allocate offsets with NaNs (NaN points are not rendered by Matplotlib scatter)
    offsets = np.full((B, 2), np.nan, dtype=float)

    # ===================== Matplotlib setup =====================
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect("equal")
    ax.set_xlim(vis_x_min, vis_x_max)
    ax.set_ylim(vis_y_min, vis_y_max)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True)

    # Target samples
    if samples_xy is not None and len(samples_xy) > 0:
        ax.scatter(
            samples_xy[:, 0],
            samples_xy[:, 1],
            s=60,  # sample radius
            c="k",
            alpha=0.45,
            marker="^",
            linewidths=0,
        )

    # Initial vehicle positions (k=0)
    XY0 = traj_xy[0, :, :]  # (B,2)

    colors = rng.random((B, 3))
    scat = ax.scatter(XY0[:, 0], XY0[:, 1], s=20, c=colors)

    fig.canvas.draw()

    # ===================== OpenCV writer =====================
    width, height = fig.canvas.get_width_height()
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    video = cv2.VideoWriter(avi_path, fourcc, 60, (width, height))

    # ===================== Frame loop =====================
    for k in range(num_frames):
        offsets[:, :] = traj_xy[k, :, :]  # (B,2)
        scat.set_offsets(offsets)
        ax.set_title(f"frame {k}/{num_frames-1}")

        fig.canvas.draw()

        buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
        buf = buf.reshape(height, width, 4)
        frame = buf[:, :, [3, 2, 1]]  # ARGB → BGR
        video.write(frame)

    video.release()
    plt.close(fig)

    print(f"[basic_usage_swarm] Video written to: {avi_path}")


def write_basic_usage_swarm_3d_video(
    results,
    vis_output_name_3d,
    x_min, x_max, y_min, y_max, z_min, z_max,
    num_vehicles,
    samples_xy,
    rng,
    elev: float = 20.0,
    azim: float = -60.0,
    fps: int = 60,
):
    """
    Batched results format:
      results["state"]["x"] : (T, B, 3) numpy
    """
    import os
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D)

    # ===================== Output path =====================
    out_dir = os.path.dirname(os.path.abspath(__file__))
    avi_path = os.path.join(out_dir, vis_output_name_3d)

    # ===================== Visualization bounds =====================
    vis_pad = 0.5
    vis_x_min, vis_x_max = x_min - vis_pad, x_max + vis_pad
    vis_y_min, vis_y_max = y_min - vis_pad, y_max + vis_pad
    vis_z_min, vis_z_max = z_min - vis_pad, z_max + vis_pad

    if abs(vis_z_max - vis_z_min) < 1e-6:
        vis_z_min -= 1.0
        vis_z_max += 1.0

    # ===================== Extract trajectories =====================
    X = np.asarray(results["state"]["x"], dtype=float)  # (T,B,3)
    T, B, _ = X.shape
    if num_vehicles != B:
        raise ValueError(f"num_vehicles={num_vehicles} but results has B={B} vehicles.")

    traj_xyz = X[:, :, :3]  # (T,B,3)
    num_frames = T

    print(f"[basic_usage_swarm] 3D batched traj: T={T}, B={B}")

    # ===================== Matplotlib setup =====================
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")

    ax.set_xlim(vis_x_min, vis_x_max)
    ax.set_ylim(vis_y_min, vis_y_max)
    ax.set_zlim(vis_z_min, vis_z_max)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    ax.view_init(elev=elev, azim=azim)

    try:
        ax.set_box_aspect((vis_x_max - vis_x_min, vis_y_max - vis_y_min, vis_z_max - vis_z_min))
    except Exception:
        pass

    # Target samples at z=0
    if samples_xy is not None and len(samples_xy) > 0:
        ax.scatter(
            samples_xy[:, 0],
            samples_xy[:, 1],
            np.zeros((samples_xy.shape[0],), dtype=float),
            s=60,
            c="k",
            alpha=0.45,
            marker="^",
            linewidths=0,
        )

    # Initial positions (k=0)
    X0 = traj_xyz[0, :, :]  # (B,3)

    colors = rng.random((B, 3))
    scat = ax.scatter(X0[:, 0], X0[:, 1], X0[:, 2], s=20, c=colors, depthshade=True)

    fig.canvas.draw()

    # ===================== OpenCV writer =====================
    width, height = fig.canvas.get_width_height()
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    video = cv2.VideoWriter(avi_path, fourcc, fps, (width, height))

    # ===================== Frame loop =====================
    for k in range(num_frames):
        xs = traj_xyz[k, :, 0]
        ys = traj_xyz[k, :, 1]
        zs = traj_xyz[k, :, 2]

        scat._offsets3d = (xs, ys, zs)
        ax.set_title(f"3D frame {k}/{num_frames-1} | elev={elev:.1f}, azim={azim:.1f}")

        fig.canvas.draw()

        buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
        buf = buf.reshape(height, width, 4)
        frame = buf[:, :, [3, 2, 1]]  # ARGB → BGR
        video.write(frame)

    video.release()
    plt.close(fig)

    print(f"[basic_usage_swarm] 3D Video written to: {avi_path}")


def save_basic_usage_swarm_csv(results, csv_path):
    """
    Batched results format:
      results["state"]["x"] : (T, B, 3) numpy

    Output columns:
      vehicle, step, x, y, z
    """
    import numpy as np
    import csv

    X = np.asarray(results["state"]["x"], dtype=float)  # (T,B,3)
    T, B, _ = X.shape

    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["vehicle", "step", "x", "y", "z"])

        for i in range(B):
            Xi = X[:, i, :]  # (T,3)
            for step in range(T):
                w.writerow([i, step, Xi[step, 0], Xi[step, 1], Xi[step, 2]])


def parse_args():
    parser = argparse.ArgumentParser(
        description="DOOT swarm simulation (2D) with GMM targets"
    )

    parser.add_argument("--num-vehicles", type=int, default=50,
                        help="Number of vehicles in the swarm")

    parser.add_argument("--num-samples", type=int, default=125,
                        help="Number of target samples")

    parser.add_argument("--num-gmm-components", type=int, default=8,
                        help="Number of GMM components")

    parser.add_argument("--t-final", type=float, default=10.0,
                        help="Simulation final time (seconds)")

    parser.add_argument("--output-name", type=str, default="basic_usage_swarm",
                        help="Base output name (used for .csv and .avi)")

    parser.add_argument("--elev", type=float, default=20.0,
                        help="3D camera elevation angle (degrees)")

    parser.add_argument("--azim", type=float, default=-60.0,
                        help="3D camera azimuth angle (degrees)")

    parser.add_argument("--no-3d-video", action="store_true",
                        help="Disable 3D AVI output")

    parser.add_argument("--use-cpu", action="store_true",
                        help="Force CPU even if CUDA is available")

    return parser.parse_args()


def run_world(args):
    import time

    print("Start configuration")
    t0 = time.perf_counter()

    # ------------------------------------------------------------
    # Select the device, CPU or GPU
    # ------------------------------------------------------------
    if args.use_cpu:
        device = torch.device("cpu")
    else:
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{torch.cuda.current_device()}")
        else:
            device = torch.device("cpu")

    print(f"Using device: {device}")
    dtype = torch.float64

    # ------------------------------------------------------------
    # Retrieve the common configuration
    # ------------------------------------------------------------
    num_vehicles = args.num_vehicles
    num_samples = args.num_samples
    num_gmm_components = args.num_gmm_components
    t_final = args.t_final
    output_name = args.output_name

    world = World.from_file(
        os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                "..",
                "rotorpy",
                "worlds",
                "wide_open.json",
            )
        )
    )

    # Set up the sampling rate
    dt = 0.01
    sampling_rate = int(1/dt)

    # ------------------------------------------------------------
    # Set up visualization bounds
    # ------------------------------------------------------------
    extents = get_world_extents(world)
    x_min, x_max, y_min, y_max, z_min, z_max = extents.tolist()

    vis_pad = 0.5
    vis_x_min, vis_x_max = x_min - vis_pad, x_max + vis_pad
    vis_y_min, vis_y_max = y_min - vis_pad, y_max + vis_pad

    # ------------------------------------------------------------
    # Fix the random seed
    # ------------------------------------------------------------
    seed = 0  # edit this integer to change the deterministic run
    rng = np.random.default_rng(seed)

    # ------------------------------------------------------------
    # Construct the target positions
    # ------------------------------------------------------------
    # Component centers in [-4, 4] for each axis
    mean_des = 8.0 * (rng.random((num_gmm_components, 2)) - 0.5)

    # Component covariance diag([0.25, 0.25]) => std = [0.5, 0.5]
    var_des = np.array([0.25, 0.25], dtype=float)
    std_des = np.sqrt(var_des)

    # Generate the target position via Gaussian mixture model
    samples_xy = sample_gmm_targets_xy_in_bounds(
        rng=rng,
        num_samples=num_samples,
        num_gmm_components=num_gmm_components,
        mean_des=mean_des,
        std_des=std_des,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
    )

    # Initialize the targeted_positions (z=0)
    targets = torch.zeros((num_samples, 3), device=device, dtype=dtype)
    targets[:, 0] = torch.as_tensor(samples_xy[:, 0], device=device, dtype=dtype)
    targets[:, 1] = torch.as_tensor(samples_xy[:, 1], device=device, dtype=dtype)
    targeted_positions_tensor = targets

    # Initialize the coordinator(s)
    # Neighbors: ceil(5% of N), force odd, cap at N-1
    k_raw = int(np.ceil(0.05 * num_vehicles))
    if (k_raw % 2) == 0:
        k_raw += 1
    num_neighbors = min(k_raw, num_vehicles - 1)

    doot_config = DootConfig(
        num_neighbors=num_neighbors,
        max_iter_primaldual=10,
        use_random_sampling=True,
        num_trial_move_samples=300,
        mean_trial_move=[0.0, 0.0, 0.0],
        var_trial_move=[0.05, 0.05, 1e-6],
        min_displacement_norm=0.1,
        planar_std_threshold=0.1,
    )

    # ------------------------------------------------------------
    # Initialize the coordinator
    # ------------------------------------------------------------
    # Since BatchedDootCbfCoordinator only uses the number of vehicles,
    # just pass a dummy list
    idxs = list(range(num_vehicles))
    vehicles_list = [None] * num_vehicles
    coordinator = BatchedDootCbfCoordinator(
        vehicles=vehicles_list,
        velocity_max=1.0,
        targeted_positions=targeted_positions_tensor,
        doot_config=doot_config,
        idxs=idxs
    )

    # ------------------------------------------------------------
    # Construct the vehicle initial positions
    # ------------------------------------------------------------
    # Agent init (ut02-style): N([5,5], diag([2,6])) with z=0, rejection-sampled inside world extents
    mean_init_xy = np.array([5.0, 5.0], dtype=float)
    std_init_xy = np.sqrt(np.array([2.0, 6.0], dtype=float))

    pos_init_xy = rejection_sample_xy(
        rng=rng,
        num=num_vehicles,
        mean_xy=mean_init_xy,
        std_xy=std_init_xy,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
    )

    # Build RotorPy initial-state dicts (z fixed at 0)
    init_rotor_speed = 1788.53

    # positions (num_vehicles,3) torch
    x_init = torch.zeros((num_vehicles, 3), device=device, dtype=dtype)
    x_init[:, 0] = torch.as_tensor(pos_init_xy[:, 0], device=device, dtype=dtype)
    x_init[:, 1] = torch.as_tensor(pos_init_xy[:, 1], device=device, dtype=dtype)
    x_init[:, 2] = 0.0

    init_positions = {
        "x": x_init,
        "v": torch.zeros((num_vehicles, 3), device=device, dtype=dtype),
        "q": torch.tensor([0.0, 0.0, 0.0, 1.0], device=device, dtype=dtype).repeat(num_vehicles, 1),
        "w": torch.zeros((num_vehicles, 3), device=device, dtype=dtype),
        "wind": torch.zeros((num_vehicles, 3), device=device, dtype=dtype),
        "rotor_speeds": torch.full((num_vehicles, 4), init_rotor_speed, device=device, dtype=dtype),
    }

    # ------------------------------------------------------------
    # Initialize multirotors and controllers
    # ------------------------------------------------------------
    all_quad_params = [quad_params] * num_vehicles
    batch_params = BatchedMultirotorParams(all_quad_params, num_vehicles, device=device)

    batched_vehicle = BatchedMultirotor(
        batch_params,
        num_vehicles,
        init_positions,
        device=device,
        integrator="dopri5",
        control_abstraction="cmd_motor_speeds", # ToDo: study and experiment other control settings
    )

    # Optional: specify feedback gains for each drone in the batch. (can be different for each drone)
    kp_pos = torch.tensor([6.5, 6.5, 15], device=device, dtype=dtype).repeat(num_vehicles, 1)
    kd_pos = torch.tensor([4.0, 4.0, 9], device=device, dtype=dtype).repeat(num_vehicles, 1)
    kp_att = torch.tensor([544.0], device=device, dtype=dtype).repeat(num_vehicles, 1)
    kd_att = torch.tensor([46.64], device=device, dtype=dtype).repeat(num_vehicles, 1)
    batched_controller = BatchedSE3Control(
        batch_params,
        num_vehicles,
        device=device,
        kp_pos=kp_pos, kd_pos=kd_pos,
        kp_att=kp_att, kd_att=kd_att)


    # ------------------------------------------------------------
    # Initialize trajectories
    # ------------------------------------------------------------
    batched_trajectory = BatchedVelocityReference(
        init_pos=init_positions["x"],     # (num_vehicles,3) torch
        init_yaw=0.0,
        init_time=0.0,
        yaw_mode="velocity_heading",
        yaw_speed_eps=1e-3,
    )

    # IMPORTANT: seed initial vel cmd so update() won’t throw
    batched_trajectory.set_vel_cmd(torch.zeros((num_vehicles, 3), device=device, dtype=dtype))

    # ------------------------------------------------------------
    # Initialize the wind profile
    # ------------------------------------------------------------
    batched_wind_profile = BatchedNoWind(num_vehicles)

    # ------------------------------------------------------------
    # Initialize the IMU
    # ------------------------------------------------------------
    batched_imu = BatchedImu(num_vehicles, device=device)

    # ------------------------------------------------------------
    # Initialize the motion capture
    # ------------------------------------------------------------
    mocap_params_list = []
    base_vel_artifact_prob = 0.001
    for i in range(num_vehicles):
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
    with_artifacts_list = [(i % 2) == 0 for i in range(num_vehicles)]
    batched_mocap = BatchedMotionCapture(num_vehicles,
        sampling_rate=sampling_rate,
        mocap_params=mocap_params_list,
        with_artifacts=with_artifacts_list,
        device=device)

    # ------------------------------------------------------------
    # Initialize EKF wind
    # ------------------------------------------------------------
    batched_quad_params = convert_params_to_batched(all_quad_params, device=device)
    batched_wind_ekf = BatchedWindEKF(num_vehicles, batched_quad_params, device=device)

    # ------------------------------------------------------------
    # Initialize the environment
    # ------------------------------------------------------------
    sim_instance = EnvironmentBatch(
        vehicles=batched_vehicle,
        controllers=batched_controller,
        trajectories=batched_trajectory,
        coordinators=[coordinator],
        imus=batched_imu,
        mocaps=batched_mocap,
        estimators=batched_wind_ekf,
        world=world,
        wind_profile=batched_wind_profile,
        sim_rate=sampling_rate,
        safety_margin=0.25,
    )

    # Set initial state AFTER environment is created
    sim_instance.set_init(init_positions)


    # ------------------------------------------------------------
    # Run simulation
    # ------------------------------------------------------------
    print("Start simulation")
    t0 = time.perf_counter()

    results = sim_instance.run(
        t_final=t_final,
        use_mocap=False,
        terminate=False,
        animate_bool=False,
        animate_wind=False,
        verbose=True,
        fname=None,
    )

    t1 = time.perf_counter()
    print(f"Finish simulation in {t1 - t0:.3f} seconds")

    # ------------------------------------------------------------
    # Save results and generate visualization
    # ------------------------------------------------------------
    print("Start visualization output")
    t0 = time.perf_counter()

    # results['state'] is a list of dicts; each dict value may be list-like
    for k, v in results["state"].items():
        if isinstance(v, list):
            results["state"][k] = np.asarray(v)

    # 2D video
    vis_output_name = output_name + ".avi"
    write_basic_usage_swarm_2d_video(
        results, vis_output_name,
        x_min, x_max, y_min, y_max,
        num_vehicles, samples_xy, rng
    )

    # 3D video
    if not args.no_3d_video:
        vis_output_name_3d = output_name + "_3d.avi"
        write_basic_usage_swarm_3d_video(
            results, vis_output_name_3d,
            x_min, x_max, y_min, y_max, z_min, z_max,
            num_vehicles, samples_xy, rng,
            elev=args.elev, azim=args.azim,
            fps=60,
        )

    t1 = time.perf_counter()
    print(f"Finish visualization output in {t1 - t0:.3f} seconds")


    # ------------------------------------------------------------
    # Save & postprocess
    # ------------------------------------------------------------
    print("Start CSV output")
    t0 = time.perf_counter()

    out_dir = os.path.dirname(os.path.abspath(__file__))
    csv_file_name = output_name + ".csv"
    csv_path = os.path.join(out_dir, csv_file_name)
    save_basic_usage_swarm_csv(results, csv_path)
    print(f"[basic_usage_swarm] CSV written to: {csv_path}")

    t1 = time.perf_counter()
    print(f"Finish CSV output in {t1 - t0:.3f} seconds")


if __name__ == "__main__":
    import time

    t0 = time.perf_counter()

    args = parse_args()

    print(f"==== Start basic usage swarm ====")

    run_world(args)

    print(f"==== Finish basic usage swarm ====")

    t1 = time.perf_counter()

    print(f"Takes {t1 - t0:.3f} seconds")
