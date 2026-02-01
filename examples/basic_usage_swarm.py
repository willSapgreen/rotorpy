"""
basic_usage_swarm.py
Test script for DOOT coordinator with RotorPy swarm simulation.
"""

# ===================== Imports =====================

import os
import numpy as np
import argparse

from rotorpy.environments import EnvironmentSwarm
from rotorpy.world import World

from rotorpy.vehicles.multirotor import Multirotor
from rotorpy.vehicles.crazyflie_params import quad_params

from rotorpy.controllers.quadrotor_control import SE3Control
from rotorpy.trajectories.velocity_reference import VelocityReference

from rotorpy.wind.default_winds import SinusoidWind

# Coordinator
from rotorpy.coordinators.doot_cbf_coordinator import (
    DootCbfCoordinator,
    DootConfig,
)

# ===================== Helpers =====================

def _get_world_extents(world_obj) -> np.ndarray:
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


def _rejection_sample_xy(
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


def _sample_gmm_targets_xy_in_bounds(
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


def _write_basic_usage_swarm_2d_video(results, vis_output_name,
                                      x_min, x_max, y_min, y_max,
                                      num_vehicles, samples_xy, rng):
    """
    Generate a 2D AVI video visualizing:
      - analytic GMM target PDF
      - target samples (faint points)
      - vehicle trajectories frame-by-frame

    Output:
      basic_usage_swarm_2d.avi (same directory as this script)

    Assumes the following variables exist in scope:
      - mean_des, var_des, num_gmm_components
      - samples_xy
      - num_vehicles
      - rng
      - x_min, x_max, y_min, y_max
    """

    import os
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import multivariate_normal

    # ===================== Output path =====================

    out_dir = os.path.dirname(os.path.abspath(__file__))
    avi_path = os.path.join(out_dir, vis_output_name)

    # ===================== Visualization bounds =====================

    vis_pad = 0.5
    vis_x_min, vis_x_max = x_min - vis_pad, x_max + vis_pad
    vis_y_min, vis_y_max = y_min - vis_pad, y_max + vis_pad

    # ===================== Extract trajectories =====================

    traj_xy = []
    traj_len = []
    for i in range(num_vehicles):
        xy = np.asarray(results["state"][i]["x"], dtype=float)[:, :2]
        traj_xy.append(xy)
        traj_len.append(xy.shape[0])

    num_frames = int(max(traj_len))
    print(f"[basic_usage_swarm] traj lengths: min={min(traj_len)}, max={max(traj_len)}")

    # Pre-allocate offsets with NaNs (NaN points are not rendered by Matplotlib scatter)
    offsets = np.full((num_vehicles, 2), np.nan, dtype=float)


    # ===================== Matplotlib setup =====================

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect("equal")
    ax.set_xlim(vis_x_min, vis_x_max)
    ax.set_ylim(vis_y_min, vis_y_max)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True)

    # Target samples
    ax.scatter(
        samples_xy[:, 0],
        samples_xy[:, 1],
        s=60, # sample radius
        c="k",
        alpha=0.45,
        marker="^",
        linewidths=0,
    )

    # Initial vehicle positions
    XY0 = np.stack(
        [np.asarray(results["state"][i]["x"])[0, :2] for i in range(num_vehicles)],
        axis=0,
    )

    colors = rng.random((num_vehicles, 3))
    scat = ax.scatter(XY0[:, 0], XY0[:, 1], s=20, c=colors)

    fig.canvas.draw()

    # ===================== OpenCV writer =====================

    width, height = fig.canvas.get_width_height()
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    video = cv2.VideoWriter(avi_path, fourcc, 60, (width, height))

    # ===================== Frame loop =====================

    for k in range(num_frames):
        # fill offsets for vehicles that have this frame
        for i in range(num_vehicles):
            if k < traj_len[i]:
                offsets[i, :] = traj_xy[i][k]
            else:
                offsets[i, :] = np.nan  # hide vehicle i at this frame

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


def _save_basic_usage_swarm_csv(results, csv_path):
    """
    Save per-vehicle trajectories to CSV for your results format:
      results['state'] is list[num_vehicles] of dicts; state['x'] is (Ti,3)

    Output columns:
      vehicle, step, x, y, z
    """
    import numpy as np
    import pandas as pd

    rows = []
    num_vehicles = len(results["state"])

    for i in range(num_vehicles):
        Xi = np.asarray(results["state"][i]["x"], dtype=float)  # (Ti,3)
        Ti = Xi.shape[0]
        for k in range(Ti):
            rows.append((i, k, Xi[k, 0], Xi[k, 1], Xi[k, 2]))

    df = pd.DataFrame(rows, columns=["vehicle", "step", "x", "y", "z"])
    df.to_csv(csv_path, index=False)
    return df


def _save_basic_usage_swarm_csv(results, csv_path):
    """
    Save per-vehicle trajectories to CSV for this results format:
      results['state'] is list[num_vehicles] of dicts
      results['state'][i]['x'] is (Ti, 3)

    Output columns:
      vehicle, step, x, y, z
    """
    import numpy as np
    import csv

    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["vehicle", "step", "x", "y", "z"])

        for i in range(len(results["state"])):
            Xi = np.asarray(results["state"][i]["x"], dtype=float)  # (Ti,3)
            for step in range(Xi.shape[0]):
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

    return parser.parse_args()


def run_world(args):
    import time

    print("Start configuration")
    t0 = time.perf_counter()

    # ===================== Param config =====================
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

    extents = _get_world_extents(world)
    x_min, x_max, y_min, y_max, z_min, z_max = extents.tolist()

    # Visualization-only bounds (no plotting changes yet; kept for future use)
    vis_pad = 0.5
    vis_x_min, vis_x_max = x_min - vis_pad, x_max + vis_pad
    vis_y_min, vis_y_max = y_min - vis_pad, y_max + vis_pad

    # ===================== Experiment knobs =====================

    seed = 0  # edit this integer to change the deterministic run
    rng = np.random.default_rng(seed)

    # ===================== Swarm setup =====================

    vehicles = [Multirotor(quad_params) for _ in range(num_vehicles)]
    controllers = [SE3Control(quad_params) for _ in range(num_vehicles)]

    # ===================== Target distribution µ* (discrete samples) =====================

    # Component centers in [-4, 4] for each axis
    mean_des = 8.0 * (rng.random((num_gmm_components, 2)) - 0.5)

    # Component covariance diag([0.25, 0.25]) => std = [0.5, 0.5]
    var_des = np.array([0.25, 0.25], dtype=float)
    std_des = np.sqrt(var_des)

    # Samples inside world extents (rejection-sampled)
    samples_xy = _sample_gmm_targets_xy_in_bounds(
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

    # targeted_positions are the samples (z=0)
    targeted_positions = np.hstack([samples_xy, np.zeros((num_samples, 1), dtype=float)]).tolist()

    # ===================== DOOT config =====================

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
        var_trial_move=[0.05, 0.05, 0.0],
        min_displacement_norm=0.1,
        planar_std_threshold=0.1,
    )

    coordinator = DootCbfCoordinator(
        vehicles=vehicles,
        velocity_max=1.0,
        targeted_positions=targeted_positions,
        doot_config=doot_config,
    )

    t1 = time.perf_counter()
    print(f"Finish configuration in {t1 - t0:.3f} seconds")

    # ===================== Initial states (MUST come before trajectories) =====================

    print("Start initialization")
    t0 = time.perf_counter()

    # Agent init (ut02-style): N([5,5], diag([2,6])) with z=0, rejection-sampled inside world extents
    mean_init_xy = np.array([5.0, 5.0], dtype=float)
    std_init_xy = np.sqrt(np.array([2.0, 6.0], dtype=float))

    pos_init_xy = _rejection_sample_xy(
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
    init_poss = []
    for i in range(num_vehicles):
        p = np.array([pos_init_xy[i, 0], pos_init_xy[i, 1], 0.0], dtype=float)

        # (Optional) if z=0 is outside z-bounds, fail early (should not happen with your world)
        if not (z_min <= p[2] <= z_max):
            raise RuntimeError(f"z=0 is outside world z-bounds [{z_min}, {z_max}].")

        init_poss.append(
            {
                "x": p,
                "v": np.zeros(3, dtype=float),
                "q": np.array([0.0, 0.0, 0.0, 1.0], dtype=float),
                "w": np.zeros(3, dtype=float),
                "wind": np.zeros(3, dtype=float),
                "rotor_speeds": np.full(4, 1788.53, dtype=float),
            }
        )

    # ===================== Trajectories (now init_poss exists) =====================

    v_cmd_fns = coordinator.get_transport_vel_fns()
    trajectories = [
        VelocityReference(v_cmd_fns[i], init_poss[i]["x"])
        for i in range(num_vehicles)
    ]

    # ===================== Environment =====================

    sim_instance = EnvironmentSwarm(
        vehicles=vehicles,
        controllers=controllers,
        trajectories=trajectories,
        coordinators=[coordinator],
        imus=None,
        mocaps=None,
        estimators=None,
        world=world,
        wind_profile=SinusoidWind(),
        sim_rate=100,
        safety_margin=0.25,
    )

    # Set initial state AFTER environment is created
    sim_instance.set_init(init_poss)

    t1 = time.perf_counter()
    print(f"Finish initialization in {t1 - t0:.3f} seconds")

    # ===================== Run simulation =====================

    print("Start simulation")
    t0 = time.perf_counter()

    results = sim_instance.run(
        t_final=t_final,
        use_mocap=False,
        terminates=False,
        animate_bool=False,
        animate_wind=False,
        verbose=True,
        fname=None,
    )

    t1 = time.perf_counter()
    print(f"Finish simulation in {t1 - t0:.3f} seconds")

    # ===================== Save & postprocess =====================

    print("Start visualization output")
    t0 = time.perf_counter()

    # results['state'] is a list of dicts; each dict value may be list-like
    for i in range(len(results.get("state", []))):
        for k, v in results["state"][i].items():
            if isinstance(v, list):
                results["state"][i][k] = np.asarray(v)

    vis_output_name = output_name + ".avi"
    _write_basic_usage_swarm_2d_video(results, vis_output_name,
                                      x_min, x_max, y_min, y_max,
                                      num_vehicles, samples_xy, rng)
    t1 = time.perf_counter()
    print(f"Finish visualization output in {t1 - t0:.3f} seconds")


    # ===================== Save & postprocess =====================

    print("Start CSV output")
    t0 = time.perf_counter()

    out_dir = os.path.dirname(os.path.abspath(__file__))
    csv_file_name = output_name + ".csv"
    csv_path = os.path.join(out_dir, csv_file_name)
    _save_basic_usage_swarm_csv(results, csv_path)
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
