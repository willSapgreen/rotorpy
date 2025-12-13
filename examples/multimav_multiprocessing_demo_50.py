"""
Imports
"""

from rotorpy.vehicles.multirotor import Multirotor
from rotorpy.vehicles.crazyflie_params import quad_params
from rotorpy.controllers.quadrotor_control import SE3Control
from rotorpy.trajectories.hover_traj import HoverTraj
from rotorpy.trajectories.circular_traj import ThreeDCircularTraj
from rotorpy.trajectories.lissajous_traj import TwoDLissajous
from rotorpy.trajectories.speed_traj import ConstantSpeed
from rotorpy.trajectories.minsnap import MinSnap
from rotorpy.world import World
from rotorpy.utils.animate import animate
from rotorpy.simulate import merge_dicts

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
import os
import yaml
import multiprocessing

####################### Helper functions

def run_sim(trajectory, t_offset, t_final=10, t_step=1/100):
    """
    Runs an instance of the simulation environment which creates a vehicle object and tracking controller on
    an individual cpu process using Python's multiprocessing.
    Inputs:
        trajectory: the trajectory object for this mav to track.
        t_offset: time offset (useful for offsetting multiple mavs on the same trajectory).
        t_final: duration of the sim for this object.
        t_step: timestep for the simulation.
    Outputs:
        time: time array.
        states: array of quadrotor states.
        controls: array of quadrotor control variables.
        flats: array of flat outputs describing the trajectory to track.
    """
    mav = Multirotor(quad_params)
    controller = SE3Control(quad_params)

    # Init mav at the first waypoint for the trajectory.
    x0 = {'x': trajectory.update(t_offset)['x'],
        'v': np.zeros(3,),
        'q': np.array([0, 0, 0, 1]), # [i,j,k,w]
        'w': np.zeros(3,),
        'wind': np.array([0,0,0]),  # Since wind is handled elsewhere, this value is overwritten
        'rotor_speeds': np.array([1788.53, 1788.53, 1788.53, 1788.53])}

    time = [0]
    states = [x0]
    flats = [trajectory.update(time[-1] + t_offset)]
    controls = [controller.update(time[-1], states[-1], flats[-1])]

    while True:
        if time[-1] >= t_final:
            break
        time.append(time[-1] + t_step)
        states.append(mav.step(states[-1], controls[-1], t_step))
        flats.append(trajectory.update(time[-1] + t_offset))
        controls.append(controller.update(time[-1], states[-1], flats[-1]))

    time        = np.array(time, dtype=float)
    states      = merge_dicts(states)
    controls    = merge_dicts(controls)
    flats       = merge_dicts(flats)

    return time, states, controls, flats

def worker_fn(cfg):
    """
    Enumerates over the configurations for each process in multiprocessing.
    """
    return run_sim(*cfg)

def find_collisions(all_positions, epsilon=1e-1):
    """
    Checks if any two agents get within epsilon meters of any other agent.
    Inputs:
        all_positions: the position vs time for each agent concatenated into one array.
        epsilon: the distance threshold constituting a collision.
    Outputs:
        collisions: a list of dictionaries where each dict describes the time of a collision, agents involved, and the location.
    """

    N, M, _ = all_positions.shape
    collisions = []

    for t in range(N):
        # Get positions.
        pos_t = all_positions[t]

        dist_sq = np.sum((pos_t[:, np.newaxis, :] - pos_t[np.newaxis, :, :])**2, axis=-1)

        # Set diagonal to a large value to avoid false positives.
        np.fill_diagonal(dist_sq, np.inf)

        close_pairs = np.where(dist_sq < epsilon**2)

        for i, j in zip(*close_pairs):
            if i < j: # avoid duplicate pairs.
                collision_info = {
                    "timestep": t,
                    "agents": (i, j),
                    "location": pos_t[i]
                }
                collisions.append(collision_info)

    return collisions

####################### Start of user code

# World: just a box large enough for all agents
world = World.empty([-3, 3, -3, 3, -3, 3])

dt = 1/100
tf = 10.0

config_list = []

# -----------------------------
# 1) 25 Lissajous trajectories
# -----------------------------
# 5x5 grid of offsets to spread them out
n_liss = 5
offset_span = 1.0      # total +/- span in x,y
height = 2.0

for i in range(n_liss):
    for j in range(n_liss):
        x_offset = -offset_span + 2 * offset_span * i / (n_liss - 1)
        y_offset = -offset_span + 2 * offset_span * j / (n_liss - 1)

        # small time offset so they don't all start in perfect sync
        t_offset = 0.2 * (i * n_liss + j)

        traj = TwoDLissajous(
            A=1.0, B=1.0, a=2, b=1,
            x_offset=x_offset,
            y_offset=y_offset,
            height=height,
        )
        config_list.append((traj, t_offset, tf, dt))

# Sanity check: should have 25 entries so far
assert len(config_list) == 25, f"Expected 25 Lissajous configs, got {len(config_list)}"

# -----------------------------
# 2) 25 MinSnap trajectories
# -----------------------------
# 25 start/end pairs around a circle, flying left->right across the box
n_minsnap = 25
R = 0.8  # radius in y–z plane

for k in range(n_minsnap):
    theta = 2 * np.pi * k / n_minsnap

    # start on left (-2, circle in y–z)
    x0 = np.array([
        -2.0,
        R * np.cos(theta),
        R * np.sin(theta),
    ])

    # end on right (+2, same y–z ring)
    xf = np.array([
        +2.0,
        R * np.cos(theta),
        R * np.sin(theta),
    ])

    traj = MinSnap(
        points=np.row_stack((x0, xf)),
        v_avg=1.0,
        verbose=False,
    )

    t_offset = 0.0  # you can stagger if you want
    config_list.append((traj, t_offset, tf, dt))

# Final sanity check: 25 Lissajous + 25 MinSnap = 50 drones
assert len(config_list) == 50, f"Expected 50 configs, got {len(config_list)}"

# -----------------------------
# Run RotorPy in parallel
# -----------------------------
with multiprocessing.Pool() as pool:
    results = pool.map(worker_fn, config_list)

# Concatenate states for animation
all_pos = []
all_rot = []
all_wind = []
all_time = results[0][0]

for r in results:
    all_pos.append(r[1]['x'])
    all_wind.append(r[1]['wind'])
    all_rot.append(Rotation.from_quat(r[1]['q']).as_matrix())

all_pos  = np.stack(all_pos, axis=1)  # (T, M, 3)
all_wind = np.stack(all_wind, axis=1) # (T, M, 3)
all_rot  = np.stack(all_rot, axis=1)  # (T, M, 3, 3)

# Collision check
collisions = find_collisions(all_pos, epsilon=2e-1)

# Animate all 50 drones
ani = animate(
    all_time,
    all_pos,
    all_rot,
    all_wind,
    animate_wind=False,
    world=world,
    filename=None,
)

# Static 3D plot of trajectories + collision markers
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
colors = plt.cm.tab20(np.linspace(0, 1, all_pos.shape[1]))

for mav in range(all_pos.shape[1]):
    ax.plot(all_pos[:, mav, 0], all_pos[:, mav, 1], all_pos[:, mav, 2],
            color=colors[mav])
    ax.plot([all_pos[-1, mav, 0]],
            [all_pos[-1, mav, 1]],
            [all_pos[-1, mav, 2]],
            '*', markersize=6,
            markerfacecolor=colors[mav],
            markeredgecolor='k')

world.draw(ax)

for event in collisions:
    ax.plot([all_pos[event['timestep'], event['agents'][0], 0]],
            [all_pos[event['timestep'], event['agents'][0], 1]],
            [all_pos[event['timestep'], event['agents'][0], 2]],
            'rx', markersize=8)

ax.set_xlabel("x, m")
ax.set_ylabel("y, m")
ax.set_zlabel("z, m")

plt.show()
