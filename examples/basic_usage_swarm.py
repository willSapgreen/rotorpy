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

N = 10  # number of vehicles

vehicles = [Multirotor(quad_params) for _ in range(N)]
controllers = [SE3Control(quad_params) for _ in range(N)]

targeted_positions = [
    [-2.0,  0.0, 1.5],
    [ 2.0,  0.0, 1.5],
]

doot_config = DootConfig(
    num_neighbors=5,
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

x0s = []

xs = np.linspace(-1.5, 1.5, 5)
ys = np.linspace(-1.0, 1.0, 2)

idx = 0
for y in ys:
    for x in xs:
        if idx >= N:
            break
        x0s.append(
            {
                "x": np.array([x, y, 1.0], dtype=float),
                "v": np.zeros(3, dtype=float),
                "q": np.array([0.0, 0.0, 0.0, 1.0], dtype=float),
                "w": np.zeros(3, dtype=float),
                "wind": np.zeros(3, dtype=float),
                "rotor_speeds": np.array([1788.53, 1788.53, 1788.53, 1788.53], dtype=float),
            }
        )
        idx += 1

# ===================== Trajectories (now x0s exists) =====================

v_cmd_fns = coordinator.get_v_cmd_fns()
trajectories = [
    VelocityReference(v_cmd_fns[i], x0s[i]["x"])
    for i in range(N)
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
    t_final=3.5,
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
