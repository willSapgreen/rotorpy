import os
import numpy as np
import torch
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
import time as clk
from rotorpy.simulate import simulate, simulate_batch
from rotorpy.utils.plotter import *
from rotorpy.world import World
from rotorpy.utils.postprocessing import unpack_sim_data
from typing import Callable, List, Optional, Dict, Any, Union


@staticmethod
def check_len(name, lst, standard):
    if len(lst) != standard:
        raise ValueError(
            f"EnvironmentSwarm(): {name} length mismatch. "
            f"Expected {standard}, got {len(lst)}"
        )


######## ENVIRONMENT FOR A SINGLE VEHICLE ########

class Environment():
    """
    Sandbox represents an instance of the simulation environment containing a unique vehicle,
    controller, trajectory generator, wind profile.

    """

    def __init__(
        self,
        vehicle,
        controller,
        trajectory,
        wind_profile = None,
        imu = None,
        mocap = None,
        world        = None,
        estimator    = None,
        sim_rate     = 100,
        safety_margin = 0.25,
    ):
        """
        Initialize simulator.

        Parameters
        ----------
        vehicle : Vehicle
            Vehicle object (required).
        controller : Controller
            Controller object (required).
        trajectory : Trajectory
            Trajectory object (required).
        wind_profile : WindProfile, optional
            Wind profile. If None, no wind is applied.
        imu : IMU, optional
            IMU sensor. If None, a default IMU is used.
        mocap : MotionCapture, optional
            Motion capture system. If None, default is used.
        world : World, optional
            World environment object.
        estimator : Estimator, optional
            State estimator object.
        sim_rate : int, default=100
            Simulation update frequency in Hz.
        safety_margin : float, default=0.25
            Radius of safety region around robot.
        """

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


    def run(self,
            t_final      = 10,
            use_mocap    = False,
            terminate    = False,
            plot            = False,
            plot_mocap      = True,
            plot_estimator  = True,
            plot_imu        = True,
            animate_bool    = False,
            animate_wind    = False,
            verbose         = False,
            fname   = None
            ):

        """
        Run the simulation.

        Parameters
        ----------
        t_final : float, default=10
            Maximum simulation duration in seconds.

        use_mocap : bool, default=False
            If True, the controller uses motion capture measurements
            instead of ground-truth state.

        terminate : bool, default=False
            If True, simulation stops early when termination conditions are met.

        plot : bool, default=False
            If True, plots vehicle states and control commands.

        plot_mocap : bool, default=True
            If True, plots motion capture pose and twist measurements.

        plot_estimator : bool, default=True
            If True, plots estimator state outputs and covariance diagonal elements.

        plot_imu : bool, default=True
            If True, plots IMU measurements.

        animate_bool : bool, default=False
            If True, plays animation of vehicle state evolution.

        animate_wind : bool, default=False
            If True, includes wind vector visualization in animation.

        verbose : bool, default=False
            If True, prints simulation statistics.

        fname : str or None, optional
            Filename for saving the animation. If None, animation is not saved.
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


######## ENVIRONMENT FOR A SWARM OF VEHICLES ########

class EnvironmentBatch:
    """
    Batched simulation environment.

    Assumption:
      - All simulation states are torch tensors in a single batch.
      - vehicles/controllers/trajectories are batched objects.
      - coordinators can be None; if provided, they are batched coordinators.

    """

    def __init__(
        self,
        vehicles,                 # BatchedMultirotor
        controllers,              # BatchedSE3Control
        trajectories,             # BatchedVelocityReference (push-based)
        coordinators: Optional[List[Any]] = None,
        imus=None,                # BatchedIMU or None
        mocaps=None,              # BatchedMotionCapture or None
        estimators=None,          # BatchedEstimator or None
        world=None,
        wind_profile=None,        # Batched wind or None
        sim_rate: int = 100,
        safety_margin: float = 0.25,
    ) -> None:

        self.vehicles = vehicles
        self.controllers = controllers
        self.trajectories = trajectories

        # coordinators can be None
        self.coordinators = coordinators

        self.safety_margin = float(safety_margin)
        self.sim_rate = int(sim_rate)

        # World
        if world is None:
            wbound = 3
            self.world = World.empty((-wbound, wbound, -wbound, wbound, -wbound, wbound))
        else:
            self.world = world

        # Wind (batched)
        if wind_profile is None:
            from rotorpy.wind.default_winds import BatchedNoWind
            self.wind_profile = BatchedNoWind(self.vehicles.num_drones)
        else:
            self.wind_profile = wind_profile

        # IMU (batched)
        if imus is None:
            from rotorpy.sensors.imu import BatchedImu
            self.imus = BatchedImu(
                num_drones=self.vehicles.num_drones,
                sampling_rate=self.sim_rate
            )
        else:
            self.imus = imus

        # Mocap (batched)
        if mocaps is None:
            from rotorpy.sensors.external_mocap import BatchedMotionCapture

            mocap_params = {
                'pos_noise_density': 0.0005 * np.ones((3,)),
                'vel_noise_density': 0.0010 * np.ones((3,)),
                'att_noise_density': 0.0005 * np.ones((3,)),
                'rate_noise_density': 0.0005 * np.ones((3,)),
                'vel_artifact_max': 5,
                'vel_artifact_prob': 0.001,
                'rate_artifact_max': 1,
                'rate_artifact_prob': 0.0002
            }

            self.mocaps = BatchedMotionCapture(
                num_drones=self.vehicles.num_drones,
                sampling_rate=self.sim_rate,
                mocap_params=mocap_params,
                with_artifacts=False
            )
        else:
            self.mocaps = mocaps

        # Estimator (batched)
        if estimators is None:
            from rotorpy.estimators.nullestimator import BatchedNullEstimator
            self.estimators = BatchedNullEstimator(self.vehicles.num_drones)
        else:
            self.estimators = estimators

        self.initial_states: Optional[Dict[str, torch.Tensor]] = None
        self.result = None

    def set_init(self, initial_states: Dict[str, torch.Tensor]) -> None:
        """
        initial_states must be a dict of torch tensors with batch dimension B.
        """
        if not isinstance(initial_states, dict):
            raise ValueError("EnvironmentBatch.set_init(): initial_states must be a dict of torch tensors.")

        # Minimal validation: must contain 'x' with shape (B,3)
        if "x" not in initial_states or (not torch.is_tensor(initial_states["x"])):
            raise ValueError("EnvironmentBatch.set_init(): initial_states must include tensor 'x'.")

        B = self.vehicles.num_drones
        if initial_states["x"].shape[0] != B:
            raise ValueError(
                f"EnvironmentBatch.set_init(): batch size mismatch. vehicles.num_drones={B}, "
                f"initial_states['x'].shape[0]={initial_states['x'].shape[0]}"
            )

        self.initial_states = initial_states


    def run(
        self,
        t_final: Union[float, np.ndarray, List[float]] = 10.0,
        use_mocap: bool = False,
        terminate=None,
        start_times=None,
        animate_bool: bool = False,
        animate_wind: bool = False,
        verbose: bool = False,
        fname: Optional[str] = None,
        print_fps: bool = False,
    ):
        """
        Run batched simulator.
        """
        if self.initial_states is None:
            raise RuntimeError("EnvironmentBatch.run(): call set_init(...) before run().")

        self.t_step = 1.0 / float(self.sim_rate)
        self.t_final = t_final
        self.use_mocap = bool(use_mocap)

        start_time = clk.time()

        (
            times,
            states,
            controls,
            flats,
            imu_measurements,
            imu_gts,
            mocap_measurements,
            state_estimates,
            exit_status,
            exit_timesteps,
        ) = simulate_batch(
            world=self.world,
            initial_states=self.initial_states,
            vehicles=self.vehicles,
            controller=self.controllers,
            trajectories=self.trajectories,
            wind_profile=self.wind_profile,
            imu=self.imus,
            mocap=self.mocaps,
            estimator=self.estimators,
            t_final=self.t_final,
            t_step=self.t_step,
            safety_margin=self.safety_margin,
            use_mocap=self.use_mocap,
            coordinators=self.coordinators,   # None is allowed
            terminate=terminate,
            start_times=start_times,
            print_fps=print_fps,
        )

        if verbose:
            wall = clk.time() - start_time
            # times is (T,B) numpy array after packing
            last_times = times[-1, :]
            print('-------------------RESULTS-----------------------')
            print(f"SIM T_FINAL -- {self.t_final} | WALL TIME -- {wall:3.2f} s")
            print(f"LAST SIM TIMES (per drone) -- {last_times}")
            print(f"EXIT STATUSES (per drone) -- {[e.value if e is not None else None for e in exit_status]}")

        # Save raw result
        self.result = dict(
            time=times,                     # (T,B) numpy
            state=states,                   # dict with arrays (T,B,...)
            control=controls,
            flat=flats,
            imu_measurements=imu_measurements,
            imu_gt=imu_gts,
            mocap_measurements=mocap_measurements,
            state_estimate=state_estimates,
            exits=exit_status,
            exit_timesteps=exit_timesteps,
        )

        # ---- Animate ----
        # Your merged outputs are (T,B,...) already. No per-vehicle list stacking required.
        if animate_bool:
            # Determine valid horizon: use minimum exit timestep if present, else full
            T = times.shape[0]
            if exit_timesteps is not None and np.any(exit_timesteps > 0):
                N = int(np.min(exit_timesteps[exit_timesteps > 0]))
                N = max(1, min(N, T))
            else:
                N = T

            time_anim = times[:N, 0]  # use drone 0 time (common dt)

            pos = states['x'][:N, :, :]         # (N,B,3)
            wind = states['wind'][:N, :, :]     # (N,B,3)
            # rot from quaternions (N,B,4) -> (N,B,3,3)
            rot = np.stack(
                [Rotation.from_quat(states['q'][:N, b, :]).as_matrix() for b in range(pos.shape[1])],
                axis=1
            )

            if fname is not None:
                if fname.endswith(".gif"):
                    fname = fname[:-4]
                elif fname.endswith(".mp4"):
                    fname = fname[:-4]

            self.ani = animate(
                time_anim,
                pos,
                rot,
                wind,
                animate_wind=animate_wind,
                world=self.world,
                filename=fname,
            )
            plt.show()

        return self.result


    def save_to_csv(self, savepath: Optional[str] = None) -> None:
        if savepath is None:
            savepath = "rotorpy_simulation_results.csv"

        if self.result is None:
            print("Error: cannot save if no results have been generated! Aborting save.")
            return

        if ".csv" not in savepath:
            savepath = savepath + ".csv"

        dataframe = unpack_sim_data(self.result)
        dataframe.to_csv(savepath)


######## Unit test code for environments.py ########

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
