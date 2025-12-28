import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
import time as clk

from rotorpy.simulate import simulate, simulate_swarm
from rotorpy.utils.plotter import *
from rotorpy.world import World

from rotorpy.utils.postprocessing import unpack_sim_data

import os

@staticmethod
def check_len(name, lst, standard):
    if len(lst) != standard:
        raise ValueError(
            f"EnvironmentSwarm(): {name} length mismatch. "
            f"Expected {standard}, got {len(lst)}"
        )

class EnvironmentSwarm():
    """
    Sandbox represents an instance of the simulation environment containing a unique vehicle,
    controller, trajectory generator, wind profile.
    """
    def __init__(self, vehicles,                 # vehicle object, must be specified.
                       controllers,              # controller object, must be specified.
                       trajectories,              # trajectory object, must be specified.
                       coordinators,
                       imus = None,              # imu sensor object, if none is supplied it will choose a default IMU sensor.
                       mocaps = None,            # mocap sensor object, if none is supplied it will choose a default mocap.
                       estimators    = None,     # estimator object

                       world        = None,     # The world object
                       wind_profile = None,     # wind profile object, if none is supplied it will choose no wind.
                       sim_rate     = 100,      # The update frequency of the simulator in Hz
                       safety_margin = 0.25,    # The radius of the safety region around the robot.
                       ):

        #  Check that all input lists have the same length
        names = [
            "vehicles",
            "controllers",
            "trajectories",
        ]

        lists = [
            vehicles,
            controllers,
            trajectories,
        ]

        sizes = {name: len(lst) for name, lst in zip(names, lists)}
        expected = sizes["vehicles"]

        mismatch = {k: v for k, v in sizes.items() if v != expected}

        if mismatch:
            raise ValueError(
                f"EnvironmentSwarm(): inconsistent list sizes. "
                f"Expected {expected}. Got {mismatch}"
            )

        # Verify the number of coordinators
        if len(coordinators) == 0:
            raise ValueError("EnvironmentSwarm(): coordinators cannot be empty")

        # Set the common variable
        num_vehicles = len(vehicles)

        # Initialize the member variables
        self.vehicles = vehicles
        self.controllers = controllers
        self.trajectories = trajectories
        self.coordinators = coordinators
        self.safety_margin = safety_margin
        self.sim_rate = sim_rate
        self.imus = []
        self.mocaps = []
        self.estimators = []

        if world is None:
            # If no world is specified, assume that it means that the intended world is free space.
            wbound = 3
            self.world = World.empty((-wbound, wbound, -wbound,
                                       wbound, -wbound, wbound))
        else:
            self.world = world

        if wind_profile is None:
            # If wind is not specified, default to no wind.
            from rotorpy.wind.default_winds import NoWind
            self.wind_profile = NoWind()
        else:
            self.wind_profile = wind_profile

        # In the event of no specified IMU, default to 0 bias with white noise with default parameters as specified below.
        if imus is None:
            from rotorpy.sensors.imu import Imu
            for idx in range(num_vehicles):
                self.imus.append(Imu(p_BS = np.zeros(3,), R_BS = np.eye(3), sampling_rate=sim_rate))
        else:
            check_len("imus", imus, num_vehicles)
            self.imus = imus

        if mocaps is None:
            # If no mocap is specified, set a default mocap.
            # Default motion capture properties. Pretty much made up based on qualitative comparison with real data from Vicon.
            mocap_params = {'pos_noise_density': 0.0005*np.ones((3,)),  # noise density for position
                    'vel_noise_density': 0.0010*np.ones((3,)),          # noise density for velocity
                    'att_noise_density': 0.0005*np.ones((3,)),          # noise density for attitude
                    'rate_noise_density': 0.0005*np.ones((3,)),         # noise density for body rates
                    'vel_artifact_max': 5,                              # maximum magnitude of the artifact in velocity (m/s)
                    'vel_artifact_prob': 0.001,                         # probability that an artifact will occur for a given velocity measurement
                    'rate_artifact_max': 1,                             # maximum magnitude of the artifact in body rates (rad/s)
                    'rate_artifact_prob': 0.0002                        # probability that an artifact will occur for a given rate measurement
            }
            from rotorpy.sensors.external_mocap import MotionCapture

            for idx in range(num_vehicles):
                self.mocaps.append(MotionCapture(sampling_rate=sim_rate, mocap_params=mocap_params, with_artifacts=False))
        else:
            check_len("mocaps", mocaps, num_vehicles)
            self.mocaps = mocaps

        if estimators is None:
            # In the likely case where an estimator is not supplied, default to the null state estimator.
            from rotorpy.estimators.nullestimator import NullEstimator
            for idx in range(num_vehicles):
                self.estimators.append(NullEstimator())
        else:
            check_len("estimators", estimators, num_vehicles)
            self.estimators = estimators

        return

    def set_init(self, initial_states):
        if len(self.vehicles) != len(initial_states):
            raise ValueError(
                f"EnvironmentSwarm::set_init(): inconsistent initial_states sizes. "
                f"Expected {len(self.vehicles)}. Got {len(initial_states)}"
            )
        for idx in range(len(initial_states)):
            self.vehicles[idx].initial_state = initial_states[idx]
        self.initial_states = initial_states

    def run(self,
            t_final=10,
            use_mocap=False,
            terminates=False,
            animate_bool=False,
            animate_wind=False,
            verbose=False,
            fname=None):
        """
        Run the swarm simulator and (optionally) animate all vehicles on the same axes.
        """

        self.t_step = 1.0 / self.sim_rate
        self.t_final = t_final
        self.use_mocap = use_mocap

        # terminates must be per-vehicle list for simulate_swarm
        if isinstance(terminates, list):
            self.terminates = terminates
        else:
            self.terminates = [terminates for _ in range(len(self.vehicles))]

        start_time = clk.time()

        (times,
        states,
        controls,
        flats,
        imu_measurements,
        imu_gts,
        mocap_measurements,
        state_estimates,
        exits) = simulate_swarm(
            self.world,
            self.wind_profile,
            self.initial_states,
            self.vehicles,
            self.controllers,
            self.trajectories,
            self.imus,
            self.mocaps,
            self.estimators,
            self.terminates,
            self.coordinators,
            self.t_final,
            self.t_step,
            self.safety_margin,
            self.use_mocap
        )

        if verbose:
            wall = clk.time() - start_time
            last_times = [t[-1] if len(t) else None for t in times]
            print('-------------------RESULTS-----------------------')
            print(f"SIM T_FINAL -- {self.t_final:3.2f} s | WALL TIME -- {wall:3.2f} s")
            print(f"LAST SIM TIMES (per vehicle) -- {last_times}")
            print(f"EXIT STATUSES (per vehicle) -- {[e.value if e is not None else None for e in exits]}")

        # Save raw swarm result
        self.result = dict(
            time=times,
            state=states,
            control=controls,
            flat=flats,
            imu_measurements=imu_measurements,
            imu_gt=imu_gts,
            mocap_measurements=mocap_measurements,
            state_estimate=state_estimates,
            exits=exits,
        )

        # ---- Build arrays for animate() ----
        # Use a common time base. Since each vehicle may terminate early, truncate to the shortest.
        lengths = [len(t) for t in times]
        N = min(lengths)
        if N == 0:
            raise RuntimeError("run(): no samples produced (all time histories empty).")

        # Assume all vehicles share same dt; take vehicle 0's time truncated.
        time_anim = times[0][:N]

        # Stack position/wind/rotation into (N, M, ...)
        M = len(states)
        pos = np.stack([states[i]['x'][:N] for i in range(M)], axis=1)        # (N, M, 3)
        wind = np.stack([states[i]['wind'][:N] for i in range(M)], axis=1)    # (N, M, 3)
        rot = np.stack([Rotation.from_quat(states[i]['q'][:N]).as_matrix()
                        for i in range(M)], axis=1)                           # (N, M, 3, 3)

        # ---- Animate ----
        if fname is not None:
            if fname.endswith(".gif"):
                fname = fname[:-4]
            elif fname.endswith(".mp4"):
                fname = fname[:-4]

        if animate_bool:
            # keep reference to prevent garbage collection
            self.ani = animate(time_anim, pos, rot, wind,
                            animate_wind=animate_wind,
                            world=self.world,
                            filename=fname)
            plt.show()

        return self.result



    def save_to_csv(self, savepath=None):
        """
        Save the simulation data in self.results to a file.
        """

        if savepath is None:
            savepath = "rotorpy_simulation_results.csv"

        if self.result is None:
            print("Error: cannot save if no results have been generated! Aborting save.")
            return
        else:
            if not ".csv" in savepath:
                savepath = savepath + ".csv"
            dataframe = unpack_sim_data(self.result)
            dataframe.to_csv(savepath)


class Environment():
    """
    Sandbox represents an instance of the simulation environment containing a unique vehicle,
    controller, trajectory generator, wind profile.

    """

    def __init__(self, vehicle,                 # vehicle object, must be specified.
                       controller,              # controller object, must be specified.
                       trajectory,              # trajectory object, must be specified.
                       wind_profile = None,     # wind profile object, if none is supplied it will choose no wind.
                       imu = None,              # imu sensor object, if none is supplied it will choose a default IMU sensor.
                       mocap = None,            # mocap sensor object, if none is supplied it will choose a default mocap.
                       world        = None,     # The world object
                       estimator    = None,     # estimator object
                       sim_rate     = 100,      # The update frequency of the simulator in Hz
                       safety_margin = 0.25,    # The radius of the safety region around the robot.
                       ):

        self.sim_rate = sim_rate
        self.vehicle = vehicle
        self.controller = controller
        self.trajectory = trajectory

        self.safety_margin = safety_margin

        if world is None:
            # If no world is specified, assume that it means that the intended world is free space.
            wbound = 3
            self.world = World.empty((-wbound, wbound, -wbound,
                                       wbound, -wbound, wbound))
        else:
            self.world = world

        if wind_profile is None:
            # If wind is not specified, default to no wind.
            from rotorpy.wind.default_winds import NoWind
            self.wind_profile = NoWind()
        else:
            self.wind_profile = wind_profile

        if imu is None:
            # In the event of specified IMU, default to 0 bias with white noise with default parameters as specified below.
            from rotorpy.sensors.imu import Imu
            self.imu = Imu(p_BS = np.zeros(3,),
                           R_BS = np.eye(3),
                           sampling_rate=sim_rate)
        else:
            self.imu = imu

        if mocap is None:
            # If no mocap is specified, set a default mocap.
            # Default motion capture properties. Pretty much made up based on qualitative comparison with real data from Vicon.
            mocap_params = {'pos_noise_density': 0.0005*np.ones((3,)),  # noise density for position
                    'vel_noise_density': 0.0010*np.ones((3,)),          # noise density for velocity
                    'att_noise_density': 0.0005*np.ones((3,)),          # noise density for attitude
                    'rate_noise_density': 0.0005*np.ones((3,)),         # noise density for body rates
                    'vel_artifact_max': 5,                              # maximum magnitude of the artifact in velocity (m/s)
                    'vel_artifact_prob': 0.001,                         # probability that an artifact will occur for a given velocity measurement
                    'rate_artifact_max': 1,                             # maximum magnitude of the artifact in body rates (rad/s)
                    'rate_artifact_prob': 0.0002                        # probability that an artifact will occur for a given rate measurement
            }
            from rotorpy.sensors.external_mocap import MotionCapture
            self.mocap = MotionCapture(sampling_rate=sim_rate, mocap_params=mocap_params, with_artifacts=False)
        else:
            self.mocap = mocap

        if estimator is None:
            # In the likely case where an estimator is not supplied, default to the null state estimator.
            from rotorpy.estimators.nullestimator import NullEstimator
            self.estimator = NullEstimator()
        else:
            self.estimator = estimator

        return

    def run(self,   t_final      = 10,       # The maximum duration of the environment in seconds
                    use_mocap    = False,    # boolean determines if the controller should use
                    terminate    = False,
                    plot            = False,    # Boolean: plots the vehicle states and commands
                    plot_mocap      = True,     # Boolean: plots the motion capture pose and twist measurements
                    plot_estimator  = True,     # Boolean: plots the estimator filter states and covariance diagonal elements
                    plot_imu        = True,     # Boolean: plots the IMU measurements
                    animate_bool    = False,    # Boolean: determines if the animation of vehicle state will play.
                    animate_wind    = False,    # Boolean: determines if the animation will include a wind vector.
                    verbose         = False,    # Boolean: will print statistics regarding the simulation.
                    fname   = None      # Filename is specified if you want to save the animation. Default location is the home directory.
                    ):

        """
        Run the simulator
        """

        self.t_step = 1/self.sim_rate
        self.t_final = t_final
        self.t_final = t_final
        self.terminate = terminate
        self.use_mocap = use_mocap

        start_time = clk.time()
        (time, state, control, flat, imu_measurements, imu_gt, mocap_measurements, state_estimate, exit) = simulate(self.world,
                                                                                                                    self.vehicle.initial_state,
                                                                                                                    self.vehicle,
                                                                                                                    self.controller,
                                                                                                                    self.trajectory,
                                                                                                                    self.wind_profile,
                                                                                                                    self.imu,
                                                                                                                    self.mocap,
                                                                                                                    self.estimator,
                                                                                                                    self.t_final,
                                                                                                                    self.t_step,
                                                                                                                    self.safety_margin,
                                                                                                                    self.use_mocap,
                                                                                                                    terminate=self.terminate,
                                                                                                                    )
        if verbose:
            # Print relevant statistics or simulator status indicators here
            print('-------------------RESULTS-----------------------')
            print('SIM TIME -- %3.2f seconds | WALL TIME -- %3.2f seconds' % (min(self.t_final, time[-1]) , (clk.time()-start_time)))
            print('EXIT STATUS -- '+exit.value)

        self.result = dict(time=time, state=state, control=control, flat=flat, imu_measurements=imu_measurements, imu_gt=imu_gt, mocap_measurements=mocap_measurements, state_estimate=state_estimate, exit=exit)

        visualizer = Plotter(self.result, self.world)

        # Remove gif or mp4 in filename if it exists (the respective functions will add appropriate extensions)
        if fname is not None:
            if ".gif" in fname:
                fname = fname.replace(".gif", "")
            if ".mp4" in fname:
                fname = fname.replace(".mp4", "")

        if animate_bool:
            # Do animation here
            visualizer.animate_results(fname=fname, animate_wind=animate_wind)
        if plot:
            # Do plotting here
            visualizer.plot_results(fname=fname,plot_mocap=plot_mocap,plot_estimator=plot_estimator,plot_imu=plot_imu)
            if not animate_bool:
                plt.show()

        return self.result

    def save_to_csv(self, savepath=None):
        """
        Save the simulation data in self.results to a file.
        """

        if savepath is None:
            savepath = "rotorpy_simulation_results.csv"

        if self.result is None:
            print("Error: cannot save if no results have been generated! Aborting save.")
            return
        else:
            if not ".csv" in savepath:
                savepath = savepath + ".csv"
            dataframe = unpack_sim_data(self.result)
            dataframe.to_csv(savepath)


if __name__=="__main__":
    from rotorpy.vehicles.crazyflie_params import quad_params
    from rotorpy.trajectories.hover_traj import HoverTraj

    from rotorpy.vehicles.multirotor import Multirotor
    from rotorpy.controllers.quadrotor_control import SE3Control

    sim = Environment(vehicle=Multirotor(quad_params),
                      controller=SE3Control(quad_params),
                      trajectory=HoverTraj(),
                      sim_rate=100
                      )

    result = sim.run(t_final=1, plot=True)