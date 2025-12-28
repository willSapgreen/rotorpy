"""
basic_usage_swarm.py
Test script for DOOT coordinator with RotorPy swarm simulation.

Changes from prior version (per discussion):
- num_vehicles = 50
- num_targets  = 125 target samples drawn from an 8-component 2D Gaussian mixture (equal weights)
- deterministic RNG seed via a single `seed` variable (edit manually to change)
- target samples are rejection-sampled to lie within world extents
- vehicle initial positions are ut02-style Gaussian (mean=[5,5], var=[2,6]) and rejection-sampled to lie within world extents
- neighbor count: ceil(5% of N), force odd, cap at N-1
- var_trial_move starts at MATLAB small value [0.05, 0.05] (z variance 0)
- bounds are read from the world JSON (no hard-coded bounds in this script)
- target PDF visualization deferred
"""

# ===================== Imports =====================

import os
import numpy as np

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
    num_targets: int,
    components: int,
    mean_des: np.ndarray,        # (components, 2)
    std_des: np.ndarray,         # (2,)
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    batch_size: int = 4096,
) -> np.ndarray:
    """
    Draw `num_targets` samples from an equal-weight 2D Gaussian mixture:
        k ~ Uniform{0..components-1}
        x ~ N(mean_des[k], diag(std_des^2))
    Rejection-sample so that returned samples lie inside [x_min, x_max] x [y_min, y_max].
    Returns (num_targets, 2).
    """
    out = np.empty((0, 2), dtype=float)
    while out.shape[0] < num_targets:
        # Choose components uniformly (equal weights)
        comp_ids = rng.integers(low=0, high=components, size=batch_size)

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

    return out[:num_targets, :]


def _write_basic_usage_swarm_2d_video(results):
    """
    Generate a 2D AVI video visualizing:
      - analytic GMM target PDF (MATLAB-style)
      - target samples (faint points)
      - vehicle trajectories frame-by-frame

    Output:
      basic_usage_swarm_2d.avi (same directory as this script)

    Assumes the following variables exist in scope:
      - mean_des, var_des, components
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
    avi_path = os.path.join(out_dir, "basic_usage_swarm_2d.avi")

    # ===================== Visualization bounds =====================

    vis_pad = 0.5
    vis_x_min, vis_x_max = x_min - vis_pad, x_max + vis_pad
    vis_y_min, vis_y_max = y_min - vis_pad, y_max + vis_pad

    # ===================== PDF grid (unit_test style) =====================

    num_grid = 201
    xg = np.linspace(vis_x_min, vis_x_max, num_grid)
    yg = np.linspace(vis_y_min, vis_y_max, num_grid)
    X, Y = np.meshgrid(xg, yg)
    grid_xy = np.column_stack([X.ravel(), Y.ravel()])

    cov_des = np.diag(var_des)

    Z = np.zeros(grid_xy.shape[0], dtype=float)
    for k in range(components):
        Z += multivariate_normal(mean=mean_des[k], cov=cov_des).pdf(grid_xy)
    Z /= float(components)
    Z = Z.reshape(X.shape)

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

    # Background PDF
    ax.imshow(
        Z,
        extent=[vis_x_min, vis_x_max, vis_y_min, vis_y_max],
        origin="lower",
        cmap="gray_r",
        interpolation="nearest",
    )

    # Target samples
    ax.scatter(
        samples_xy[:, 0],
        samples_xy[:, 1],
        s=8,
        c="k",
        alpha=0.15,
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


# ===================== World =====================

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

t_final = 200.0

# ===================== Swarm setup =====================

num_vehicles = 50
vehicles = [Multirotor(quad_params) for _ in range(num_vehicles)]
controllers = [SE3Control(quad_params) for _ in range(num_vehicles)]

# ===================== Target distribution µ* (discrete samples) =====================

num_targets = 125
components = 8

# Component centers in [-4, 4] for each axis (same construction as MATLAB/ut02)
mean_des = 8.0 * (rng.random((components, 2)) - 0.5)

# Component covariance diag([0.25, 0.25]) => std = [0.5, 0.5]
var_des = np.array([0.25, 0.25], dtype=float)
std_des = np.sqrt(var_des)

# Samples inside world extents (rejection-sampled)
samples_xy = _sample_gmm_targets_xy_in_bounds(
    rng=rng,
    num_targets=num_targets,
    components=components,
    mean_des=mean_des,
    std_des=std_des,
    x_min=x_min,
    x_max=x_max,
    y_min=y_min,
    y_max=y_max,
)

# targeted_positions are the samples (z=0)
targeted_positions = np.hstack([samples_xy, np.zeros((num_targets, 1), dtype=float)]).tolist()

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
    var_trial_move=[0.05, 0.05, 0.0],   # start with MATLAB small value; adjust later if needed
    min_displacement_norm=0.1,
    planar_std_threshold=0.1,
)

coordinator = DootCbfCoordinator(
    vehicles=vehicles,
    velocity_max=1.0,
    targeted_positions=targeted_positions,
    doot_config=doot_config,
)

# ===================== Initial states (MUST come before trajectories) =====================

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
x0s = []
for i in range(num_vehicles):
    p = np.array([pos_init_xy[i, 0], pos_init_xy[i, 1], 0.0], dtype=float)

    # (Optional) if z=0 is outside z-bounds, fail early (should not happen with your world)
    if not (z_min <= p[2] <= z_max):
        raise RuntimeError(f"z=0 is outside world z-bounds [{z_min}, {z_max}].")

    x0s.append(
        {
            "x": p,
            "v": np.zeros(3, dtype=float),
            "q": np.array([0.0, 0.0, 0.0, 1.0], dtype=float),
            "w": np.zeros(3, dtype=float),
            "wind": np.zeros(3, dtype=float),
            "rotor_speeds": np.full(4, 1788.53, dtype=float),
        }
    )

# ===================== Trajectories (now x0s exists) =====================

v_cmd_fns = coordinator.get_v_cmd_fns()
trajectories = [
    VelocityReference(v_cmd_fns[i], x0s[i]["x"])
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
sim_instance.set_init(x0s)

# ===================== Run simulation =====================

results = sim_instance.run(
    t_final=t_final,
    use_mocap=False,
    terminates=False,
    animate_bool=False,
    animate_wind=False,
    verbose=True,
    fname=None,
)


# results['state'] is a list of dicts; each dict value may be list-like
for i in range(len(results.get("state", []))):
    for k, v in results["state"][i].items():
        if isinstance(v, list):
            results["state"][i][k] = np.asarray(v)

_write_basic_usage_swarm_2d_video(results)

# ===================== Save & postprocess =====================

out_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(out_dir, "basic_usage_swarm.csv")
_save_basic_usage_swarm_csv(results, csv_path)
print(f"[basic_usage_swarm] CSV written to: {csv_path}")
