"""
basic_usage_swarm.py
Test script for DOOT coordinator with RotorPy swarm simulation.
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

# ===================== Swarm setup =====================

num_vehicles = 20  # (CHANGED) 20 vehicles

vehicles = [Multirotor(quad_params) for _ in range(num_vehicles)]
controllers = [SE3Control(quad_params) for _ in range(num_vehicles)]

# (CHANGED) only one target position at origin
targeted_positions = [
    [0.0, 0.0, 0.0],
]

doot_config = DootConfig(
    num_neighbors=min(5, num_vehicles - 1),   # keep "10 neighbors" intent, but cap at num_vehicles-1
    max_iter_primaldual=10,
    use_random_sampling=True,
    num_trial_move_samples=300,
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

bounds = np.array([[-5, 5], [-5, 5], [-0.5, 3]], dtype=float)
mean_init = np.array([4.0, 4.0, 0.0], dtype=float)
std_init  = np.sqrt(np.array([1.0, 3.0, 0.0], dtype=float))

rng = np.random.default_rng(0)
x0_positions = np.empty((0, 3), dtype=float)

while x0_positions.shape[0] < num_vehicles:
    batch = rng.normal(loc=mean_init, scale=std_init, size=(num_vehicles, 3))
    batch[:, 2] = mean_init[2]  # enforce z variance = 0 exactly
    ok = np.all((batch >= bounds[:, 0]) & (batch <= bounds[:, 1]), axis=1)
    x0_positions = np.vstack([x0_positions, batch[ok]])

x0_positions = x0_positions[:num_vehicles]

# Build RotorPy initial-state dicts
x0s = []
for i in range(num_vehicles):
    p = x0_positions[i, :].astype(float)
    x0s.append(
        {
            "x": p,
            "v": np.zeros(3, dtype=float),
            "q": np.array([0.0, 0.0, 0.0, 1.0], dtype=float),
            "w": np.zeros(3, dtype=float),
            "wind": np.zeros(3, dtype=float),
            "rotor_speeds": np.array([1788.53, 1788.53, 1788.53, 1788.53], dtype=float),
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
    t_final=10.0,
    use_mocap=False,
    terminates=False,
    animate_bool=True,
    animate_wind=True,
    verbose=True,
    fname=None,
)

# ===================== Save & postprocess =====================

sim_instance.save_to_csv("basic_usage_swarm.csv")

from rotorpy.utils.postprocessing import unpack_sim_data
df = unpack_sim_data(results)
