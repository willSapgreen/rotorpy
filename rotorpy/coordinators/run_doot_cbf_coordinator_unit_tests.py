import copy
import os
from typing import Callable, List, Optional, Dict, Any, Tuple

import numpy as np
import matplotlib.pyplot as plt
import cv2

from doot_cbf_coordinator import DootCbfCoordinator, DootConfig, CbfConfig


# =========================
# Helper: planar detection (same rule as coordinator.step)
# =========================
def detect_planar_axes(positions: np.ndarray, planar_std_threshold: float):
    std_xyz = np.std(positions, axis=0)  # (3,)
    planar = bool(np.any(std_xyz <= float(planar_std_threshold)))
    if planar:
        axes_2d = tuple(np.argsort(std_xyz)[-2:].tolist())  # keep best 2 axes
    else:
        axes_2d = None
    return planar, axes_2d

# -----------------------------
# Helper: compute_kde_local on a grid (truncated kernel)
# -----------------------------
def compute_kde_local_grid(pos_xy: np.ndarray, Xg: np.ndarray, Yg: np.ndarray, bw: float, R: float) -> np.ndarray:
    """
    For grid evaluation:
        - truncates kernel: include only agents with ||x - xi|| <= R
        - normalization: C = 2*pi*bw^2*(1 - exp(-R^2/(2*bw^2)))
        - rho(x) = sum_i exp(-||x-xi||^2/(2*bw^2)) / (N*C)
    """
    pos_xy = np.asarray(pos_xy, dtype=float)
    Nloc = pos_xy.shape[0]

    XYg = np.stack([Xg.ravel(), Yg.ravel()], axis=1)  # (M,2)
    diffs = XYg[:, None, :] - pos_xy[None, :, :]      # (M,N,2)
    sq = np.sum(diffs * diffs, axis=2)                # (M,N)

    mask = sq <= (R * R)
    K = np.zeros_like(sq, dtype=float)
    K[mask] = np.exp(-sq[mask] / (2.0 * bw * bw))

    C = 2.0 * np.pi * bw * bw * (1.0 - np.exp(-(R * R) / (2.0 * bw * bw)))
    rho = np.sum(K, axis=1) / (float(Nloc) * C)
    return rho.reshape(Xg.shape)

# -----------------------------
# Helper: compute_kde_local on an agent (truncated kernel)
# -----------------------------
def compute_kde_density_and_gradient(
        positions_prev: np.ndarray,
        vehicle_idx: int,
        num_vehicles: int,
        bw: float,
        radius: float,
        *,
        planar: bool,
        axes_2d: Optional[Tuple[int, int]],
    ) -> Tuple[float, np.ndarray]:
        """
        KDE rho and grad_rho at agent vehicle_idx using previous positions, consistent with DOOT plane.

        If planar=True:
        - compute KDE in the DOOT plane defined by axes_2d (two indices in {0,1,2})
        - return a (3,) gradient with nonzeros only on axes_2d

        If planar=False:
        - compute KDE in 3D and return full (3,) gradient
        """
        positions_prev = np.asarray(positions_prev, dtype=float)
        if positions_prev.shape != (num_vehicles, 3):
            raise ValueError(f"positions_prev must have shape {(num_vehicles, 3)}, got {positions_prev.shape}")
        if not (0 <= vehicle_idx < num_vehicles):
            raise IndexError(f"vehicle_idx out of range: {vehicle_idx}")

        if planar:
            if axes_2d is None or len(axes_2d) != 2:
                raise ValueError("planar=True requires axes_2d=(i,j).")

            # --- work in the selected 2D plane ---
            positions_2d = positions_prev[:, axes_2d]                      # (N,2)
            position_2d = positions_prev[vehicle_idx, axes_2d]            # (2,)
            diffs = positions_2d - position_2d                              # (N,2)
            dists = np.linalg.norm(diffs, axis=1)        # (N,)

            neigh_mask = (dists <= radius ) & (dists >= 0.0)

            kde_norm_const_2d_truncated = (
                2.0 * np.pi * bw**2 * (1.0 - np.exp(-radius**2 / (2*bw**2)))
            )

            if not np.any(neigh_mask):
                rho_m = 0.0
                grad_2d = np.zeros((2,), dtype=float)
            else:
                diffs_k = diffs[neigh_mask]                          # (K,2)
                dists_k = dists[neigh_mask]                          # (K,)
                w = np.exp(-(dists_k ** 2) / (2.0 * (bw ** 2)))      # (K,)

                rho_m = float(np.sum(w) / (float(num_vehicles) * kde_norm_const_2d_truncated))

                grad_2d = -(1.0 / (float(num_vehicles) * kde_norm_const_2d_truncated * (bw ** 2))) * np.sum(
                    diffs_k * w[:, None], axis=0
                )

            grad_3d = np.zeros((3,), dtype=float)
            grad_3d[axes_2d[0]] = grad_2d[0]
            grad_3d[axes_2d[1]] = grad_2d[1]
            return rho_m, grad_3d

        else:
            # --- 3D KDE ---
            diffs = positions_prev - positions_prev[vehicle_idx, :]                 # (N,3)
            dists = np.linalg.norm(diffs, axis=1)         # (N,)
            neigh_mask = (dists <= radius) & (dists > 0.0)

            # 3D Gaussian normalization
            kde_norm_const_3d_truncated = (2.0 * np.pi) ** (1.5) * (bw ** 3)

            if not np.any(neigh_mask):
                rho_m = 0.0
                grad_3d = np.zeros((3,), dtype=float)
            else:
                diffs_k = diffs[neigh_mask]                          # (K,3)
                dists_k = dists[neigh_mask]                          # (K,)
                w = np.exp(-(dists_k ** 2) / (2.0 * (bw ** 2)))      # (K,)

                rho_m = float(np.sum(w) / (float(num_vehicles) * kde_norm_const_3d_truncated))

                grad_3d = -(1.0 / (float(num_vehicles) * kde_norm_const_3d_truncated * (bw ** 2))) * np.sum(
                    diffs_k * w[:, None], axis=0
                )

            return rho_m, grad_3d

def unit_test_01():

    # Initialize the "world"
    num_vehicles = 20
    vehicles = [None] * num_vehicles

    targeted_positions = [[0.0, 0.0, 0.0]]

    sim_time = 200
    dt = 1
    intervals = int(sim_time / dt)

    # Initialize the vehicles' initial states
    bounds = np.array([[-5, 5], [-5, 5], [-0.5, 3]], dtype=float)
    mean_init = np.array([4.0, 4.0, 0.0], dtype=float)
    std_init  = np.sqrt(np.array([1.0, 3.0, 0.0], dtype=float))

    rng = np.random.default_rng(0)
    x0_positions = np.empty((0, 3), dtype=float)

    while x0_positions.shape[0] < num_vehicles:
        batch = rng.normal(loc=mean_init, scale=std_init, size=(num_vehicles, 3))
        batch[:, 2] = mean_init[2]
        ok = np.all((batch >= bounds[:, 0]) & (batch <= bounds[:, 1]), axis=1)
        x0_positions = np.vstack([x0_positions, batch[ok]])

    x0_positions = x0_positions[:num_vehicles]

    x0s = []
    for i in range(num_vehicles):
        x0s.append(
            {
                "x": x0_positions[i].astype(float),
                "v": np.zeros(3),
                "q": np.array([0.0, 0.0, 0.0, 1.0]),
                "w": np.zeros(3),
                "wind": np.zeros(3),
                "rotor_speeds": np.full(4, 1788.53),
            }
        )

    doot_config = DootConfig(
        num_neighbors=min(5, num_vehicles - 1),
        max_iter_primaldual=10,
        use_random_sampling=False,
        num_trial_move_samples=300,
        min_displacement_norm=0.1,
        planar_std_threshold=0.1,
    )

    coordinator = DootCbfCoordinator(
        vehicles=vehicles,
        velocity_max=1.0,
        targeted_positions=targeted_positions,
        doot_config=doot_config,
        idxs=list(range(num_vehicles)),
    )

    # Output path: same directory as doot_cbf_coordinator.py
    out_dir = os.path.dirname(os.path.abspath(__file__))
    avi_path = os.path.join(out_dir, "doot_cbf_coordinator_ut01.avi")

    cur_time = 0.0
    cur_x = copy.deepcopy(x0s)

    # --- Matplotlib 2D plot setup ---
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(bounds[0, 0], bounds[0, 1])
    ax.set_ylim(bounds[1, 0], bounds[1, 1])
    ax.set_aspect("equal")
    ax.grid(True)
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    target_xy = np.array(targeted_positions, dtype=float)[:, :2]
    ax.plot(target_xy[:, 0], target_xy[:, 1], "rx", markersize=10)

    # Initial xy
    xy0 = np.array([cur_x[i]["x"][:2] for i in range(num_vehicles)], dtype=float)

    # Random per-vehicle colors (different each run is OK)
    colors = np.random.rand(num_vehicles, 3)  # RGB in [0,1]

    scat = ax.scatter(xy0[:, 0], xy0[:, 1], s=20, c=colors)

    fig.canvas.draw()

    # --- OpenCV VideoWriter ---
    width, height = fig.canvas.get_width_height()
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    video = cv2.VideoWriter(avi_path, fourcc, 10, (width, height))

    for step in range(intervals):
        cur_time += dt
        coordinator.step(cur_time, cur_x)
        # transport_vel_fns = coordinator.get_transport_vel_fns()
        v_batch = coordinator.get_vel_cmds()

        for i in range(num_vehicles):
            # v = np.asarray(transport_vel_fns[i](cur_time), dtype=float).reshape(3,)
            # v = np.asarray(v_batch[i](cur_time), dtype=float).reshape(3,)
            v = np.asarray(v_batch[i], dtype=float).reshape(3,)
            cur_x[i]["x"] += v * dt

        xy = np.array([cur_x[i]["x"][:2] for i in range(num_vehicles)], dtype=float)
        scat.set_offsets(xy)
        ax.set_title(f"cur_time = {cur_time:.2f} s")

        fig.canvas.draw()

        # Backend-safe capture for TkAgg: ARGB buffer
        buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
        buf = buf.reshape(height, width, 4)

        # ARGB -> BGR for OpenCV (drop alpha)
        frame = buf[:, :, [3, 2, 1]]
        video.write(frame)

    video.release()
    plt.close(fig)


def unit_test_02():
    import copy
    import os

    import numpy as np
    import matplotlib.pyplot as plt
    import cv2
    from scipy.stats import multivariate_normal
    from matplotlib.colors import ListedColormap

    # =========================
    # Setup
    # =========================
    num_vehicles = 200
    vehicles = [None] * num_vehicles

    num_targets = 500
    components = 20
    dom_size = 5

    sim_time = 200
    dt = 1
    intervals = int(sim_time / dt)

    rng = np.random.default_rng(10)

    # =========================
    # Target distribution
    # =========================
    mean_des = 8.0 * (rng.random((components, 2)) - 0.5)
    var_des = np.array([0.25, 0.25], dtype=float)
    cov_des = np.diag(var_des)

    comp_ids = rng.integers(low=0, high=components, size=num_targets)
    samples_xy = np.empty((num_targets, 2), dtype=float)
    for k in range(components):
        idx = np.where(comp_ids == k)[0]
        if idx.size > 0:
            samples_xy[idx, :] = rng.multivariate_normal(
                mean=mean_des[k], cov=cov_des, size=idx.size
            )

    targeted_positions = np.hstack([samples_xy, np.zeros((num_targets, 1), dtype=float)]).tolist()

    # =========================
    # Initial agent distribution
    # =========================
    mean_init = np.array([5.0, 5.0], dtype=float)
    var_init = np.array([2.0, 6.0], dtype=float)
    cov_init = np.diag(var_init)

    pos_init_xy = rng.multivariate_normal(mean=mean_init, cov=cov_init, size=num_vehicles)

    x0s = []
    for i in range(num_vehicles):
        x0s.append(
            {
                "x": np.array([pos_init_xy[i, 0], pos_init_xy[i, 1], 0.0], dtype=float),
                "v": np.zeros(3),
                "q": np.array([0.0, 0.0, 0.0, 1.0]),
                "w": np.zeros(3),
                "wind": np.zeros(3),
                "rotor_speeds": np.full(4, 1788.53),
            }
        )

    # =========================
    # Configs
    # =========================
    doot_config = DootConfig(
        num_neighbors=min(10, num_vehicles - 1),
        max_iter_primaldual=20,
        use_random_sampling=False,
        num_trial_move_samples=300,
        min_displacement_norm=0.1,
        planar_std_threshold=0.1,
    )

    cbf_config = CbfConfig(
        density_upper_gain=100.0,
        density_lower_gain=1000.0,
        density_upper_bound=0.045,
        density_lower_bound=0.011,
        kde_bandwidth=0.3,
    )

    coordinator = DootCbfCoordinator(
        vehicles=vehicles,
        velocity_max=1.0,
        targeted_positions=targeted_positions,
        doot_config=doot_config,
        apply_cbf=False,
        cbf_config=cbf_config,
        idxs=list(range(num_vehicles)),
    )

    # =========================
    # Output video
    # =========================
    out_dir = os.path.dirname(os.path.abspath(__file__))
    avi_path = os.path.join(out_dir, "doot_cbf_coordinator_ut02_side_by_side.avi")

    cur_nom = copy.deepcopy(x0s)
    cur_cbf = copy.deepcopy(x0s)

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(12, 6))
    for ax in (ax_l, ax_r):
        ax.set_aspect("equal")
        ax.grid(True)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.scatter(samples_xy[:, 0], samples_xy[:, 1], s=8, c="k", alpha=0.15, linewidths=0)

    colors = rng.random((num_vehicles, 3))
    xy0 = np.array([cur_nom[i]["x"][:2] for i in range(num_vehicles)], dtype=float)

    # Agent scatters
    scatL = ax_l.scatter(xy0[:, 0], xy0[:, 1], s=20, c=colors)
    scatR = ax_r.scatter(xy0[:, 0], xy0[:, 1], s=20, c=colors)

    # Sparse squares
    sparse_l = ax_l.scatter(
        [], [], s=120, marker="s",
        facecolors=(1.0, 0.8, 0.2),
        edgecolors=(1.0, 0.6, 0.0),
        linewidths=1.5,
        alpha=0.4,
        zorder=5,
    )
    sparse_r = ax_r.scatter(
        [], [], s=120, marker="s",
        facecolors=(1.0, 0.8, 0.2),
        edgecolors=(1.0, 0.6, 0.0),
        linewidths=1.5,
        alpha=0.4,
        zorder=5,
    )

    # ===== Unsafe region overlay (grid mask) =====
    pts = np.linspace(-2 * dom_size, 2 * dom_size, 50)
    Xg, Yg = np.meshgrid(pts, pts)
    alphaVal = 1.0

    red_cmap = ListedColormap([(1.0, 0.0, 0.0, 1.0)])

    maskL = ax_l.imshow(
        np.zeros_like(Xg, dtype=float),
        extent=[pts[0], pts[-1], pts[0], pts[-1]],
        origin="lower",
        cmap=red_cmap,
        alpha=0.0,
        zorder=3,
    )
    maskR = ax_r.imshow(
        np.zeros_like(Xg, dtype=float),
        extent=[pts[0], pts[-1], pts[0], pts[-1]],
        origin="lower",
        cmap=red_cmap,
        alpha=0.0,
        zorder=3,
    )

    for ax in (ax_l, ax_r):
        ax.set_xlim(pts[0], pts[-1])
        ax.set_ylim(pts[0], pts[-1])

    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    video = cv2.VideoWriter(
        avi_path,
        cv2.VideoWriter_fourcc(*"XVID"),
        10,
        (width, height),
    )

    eps = float(cbf_config.density_upper_bound)
    eps_min = float(cbf_config.density_lower_bound)

    cur_time = 0.0

    # Initialize previous positions
    x_prev_nom = np.array([cur_nom[i]["x"] for i in range(num_vehicles)], dtype=float)
    x_prev_cbf = np.array([cur_cbf[i]["x"] for i in range(num_vehicles)], dtype=float)

    for step in range(intervals):
        # Previous positions for this step
        x_prev_nom = np.array([cur_nom[i]["x"] for i in range(num_vehicles)], dtype=float)
        x_prev_cbf = np.array([cur_cbf[i]["x"] for i in range(num_vehicles)], dtype=float)

        planar_nom, axes_2d_nom = detect_planar_axes(x_prev_nom, doot_config.planar_std_threshold)
        planar_cbf, axes_2d_cbf = detect_planar_axes(x_prev_cbf, doot_config.planar_std_threshold)

        # ---- DOOT (nominal) ----
        cur_time += dt
        coordinator.step(cur_time, cur_nom)
        # v_nom = np.array([coordinator.get_transport_vel_fns()[i](cur_time) for i in range(num_vehicles)], dtype=float)
        v_nom = np.asarray(coordinator.get_vel_cmds(), dtype=float).copy()
        v_nom[:, 2] = 0.0

        # ---- CBF projection using previous CBF positions ----
        v_cbf = v_nom.copy()
        for vehicle_idx in range(num_vehicles):
            rho_m, grad_rho_m = coordinator._compute_kde_density_and_gradient(
                x_prev_cbf, vehicle_idx, planar=planar_cbf, axes_2d=axes_2d_cbf
            )
            v_cbf[vehicle_idx, :] = coordinator._apply_cbf_projection(v_nom[vehicle_idx, :], rho_m, grad_rho_m)
        v_cbf[:, 2] = 0.0

        # ---- Integrate ----
        for i in range(num_vehicles):
            cur_nom[i]["x"] += dt * v_nom[i]
            cur_cbf[i]["x"] += dt * v_cbf[i]
            cur_nom[i]["x"][2] = 0.0
            cur_cbf[i]["x"][2] = 0.0

        xy_nom = np.array([cur_nom[i]["x"][:2] for i in range(num_vehicles)], dtype=float)
        xy_cbf = np.array([cur_cbf[i]["x"][:2] for i in range(num_vehicles)], dtype=float)

        scatL.set_offsets(xy_nom)
        scatR.set_offsets(xy_cbf)

        # ===== Unsafe region overlay : compute on PREVIOUS positions =====
        xy_prev_nom = x_prev_nom[:, :2]
        xy_prev_cbf = x_prev_cbf[:, :2]

        rho_grid_nom = compute_kde_local_grid(xy_prev_nom, Xg, Yg, bw=float(cbf_config.kde_bandwidth), R=float(cbf_config.kde_radius_bar))
        rho_grid_cbf = compute_kde_local_grid(xy_prev_cbf, Xg, Yg, bw=float(cbf_config.kde_bandwidth), R=float(cbf_config.kde_radius_bar))

        mask_nom = (rho_grid_nom >= eps).astype(float)
        mask_cbf = (rho_grid_cbf >= eps).astype(float)

        maskL.set_data(mask_nom)
        maskL.set_alpha(mask_nom * alphaVal)

        maskR.set_data(mask_cbf)
        maskR.set_alpha(mask_cbf * alphaVal)

        # ===== Sparse squares (agent-based, computed at agents on PREVIOUS positions) =====
        # Keep your agent-based rho for sparse markers
        rho_nom_agents = np.array(
            [
                compute_kde_density_and_gradient(
                    x_prev_nom, j, num_vehicles,
                    cbf_config.kde_bandwidth, cbf_config.kde_radius_bar,
                    planar=planar_nom, axes_2d=axes_2d_nom
                )[0]
                for j in range(num_vehicles)
            ],
            dtype=float,
        )
        rho_cbf_agents = np.array(
            [
                compute_kde_density_and_gradient(
                    x_prev_cbf, j, num_vehicles,
                    cbf_config.kde_bandwidth, cbf_config.kde_radius_bar,
                    planar=planar_cbf, axes_2d=axes_2d_cbf
                )[0]
                for j in range(num_vehicles)
            ],
            dtype=float,
        )

        sparse_l.set_offsets(xy_prev_nom[rho_nom_agents < eps_min])
        sparse_r.set_offsets(xy_prev_cbf[rho_cbf_agents < eps_min])

        ax_l.set_title(f"Nominal (no CBF)   cur_time = {cur_time:.2f} s")
        ax_r.set_title(f"With CBF          cur_time = {cur_time:.2f} s")

        # Write frame
        fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
        buf = buf.reshape(height, width, 4)
        frame = buf[:, :, [3, 2, 1]]  # ARGB -> BGR
        video.write(frame)

    video.release()
    plt.close(fig)

def unit_test_03():
    """
    MATLAB-log replay + video visualization (side-by-side like unit_test_02), WITH:
      - MATLAB-style unsafe-region overlay (grid mask: rho_grid >= eps)
      - Sparse-agent squares (agent-based: rho_agent < eps_min)

    Output: doot_cbf_coordinator_ut03_replay_side_by_side.avi
    """
    import os
    import time
    import numpy as np
    import matplotlib.pyplot as plt
    import cv2
    from scipy.io import loadmat
    from matplotlib.colors import ListedColormap

    # -----------------------------
    # Load MATLAB log
    # -----------------------------
    mat_path = "/data/doot_cbf_output.mat"

    print(f"[unit_test_03] Loaded {mat_path}")
    data = loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    params = data["params"]

    N = int(getattr(params, "N"))
    T = int(getattr(params, "T"))
    dt_mat = float(getattr(params, "dt"))
    n_neigh_mat = int(getattr(params, "n_neigh"))
    bw = float(getattr(params, "bw"))
    R = float(getattr(params, "R"))

    print(f"[unit_test_03] N={N}, T={T}, dt={dt_mat}, n_neigh={n_neigh_mat}, bw={bw}, R={R}")

    # Targets (MATLAB "samples") (M,2)
    samples_xy = np.asarray(data["samples"], dtype=float)
    if samples_xy.ndim != 2 or samples_xy.shape[1] != 2:
        raise ValueError(f"[unit_test_03] samples shape unexpected: {samples_xy.shape}")

    # Logs
    x_prev_nom_log = np.asarray(data["x_prev_nom_log"], dtype=float)   # (T,N,2)
    x_prev_cbf_log = np.asarray(data["x_prev_cbf_log"], dtype=float)   # (T,N,2)
    v_nom_log = np.asarray(data["v_nom_log"], dtype=float)             # (T,N,2)

    if x_prev_nom_log.shape != (T, N, 2):
        raise ValueError(f"[unit_test_03] x_prev_nom_log shape unexpected: {x_prev_nom_log.shape}")
    if x_prev_cbf_log.shape != (T, N, 2):
        raise ValueError(f"[unit_test_03] x_prev_cbf_log shape unexpected: {x_prev_cbf_log.shape}")
    if v_nom_log.shape != (T, N, 2):
        raise ValueError(f"[unit_test_03] v_nom_log shape unexpected: {v_nom_log.shape}")

    # -----------------------------
    # Coordinator (used only for KDE+CBF computations)
    # -----------------------------
    vehicles = [None] * N

    cbf_config = CbfConfig(
        density_upper_gain=100.0,
        density_lower_gain=1000.0,
        density_upper_bound=0.045,
        density_lower_bound=0.011,
        kde_bandwidth=bw,
    )

    doot_config = DootConfig(
        num_neighbors=n_neigh_mat,
        max_iter_primaldual=20,
        use_random_sampling=False,
        num_trial_move_samples=int(getattr(params, "n_trial_samples", 300)) if hasattr(params, "n_trial_samples") else 300,
        min_displacement_norm=0.1,
        planar_std_threshold=0.1,
    )

    targeted_positions = np.hstack([samples_xy, np.zeros((samples_xy.shape[0], 1), dtype=float)]).tolist()

    coordinator = DootCbfCoordinator(
        vehicles=vehicles,
        velocity_max=None,
        targeted_positions=targeted_positions,
        doot_config=doot_config,
        apply_cbf=False,   # we apply projection manually
        cbf_config=cbf_config,
        idxs=list(range(N)),
    )

    eps = float(cbf_config.density_upper_bound)
    eps_min = float(cbf_config.density_lower_bound)

    # -----------------------------
    # Video output + plot setup
    # -----------------------------
    out_dir = os.path.dirname(os.path.abspath(__file__))
    avi_path = os.path.join(out_dir, "doot_cbf_coordinator_ut03_replay_side_by_side.avi")

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(12, 6))
    for ax in (ax_l, ax_r):
        ax.set_aspect("equal")
        ax.grid(True)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.scatter(samples_xy[:, 0], samples_xy[:, 1], s=8, c="k", alpha=0.15, linewidths=0)

    # Deterministic colors
    rng = np.random.default_rng(0)
    colors = rng.random((N, 3))

    # Initial displayed positions (t=0 "prev")
    xy0_nom = x_prev_nom_log[0, :, :]
    xy0_cbf = x_prev_cbf_log[0, :, :]

    scatL = ax_l.scatter(xy0_nom[:, 0], xy0_nom[:, 1], s=20, c=colors, zorder=2)
    scatR = ax_r.scatter(xy0_cbf[:, 0], xy0_cbf[:, 1], s=20, c=colors, zorder=2)

    # ===== Unsafe region overlay (grid mask) =====
    dom_size = 5
    pts = np.linspace(-2 * dom_size, 2 * dom_size, 50)
    Xg, Yg = np.meshgrid(pts, pts)
    alphaVal = 0.75

    red_cmap = ListedColormap([(1.0, 0.0, 0.0, 1.0)])

    maskL = ax_l.imshow(
        np.zeros_like(Xg, dtype=float),
        extent=[pts[0], pts[-1], pts[0], pts[-1]],
        origin="lower",
        cmap=red_cmap,
        alpha=0.0,
        zorder=3,
    )
    maskR = ax_r.imshow(
        np.zeros_like(Xg, dtype=float),
        extent=[pts[0], pts[-1], pts[0], pts[-1]],
        origin="lower",
        cmap=red_cmap,
        alpha=0.0,
        zorder=3,
    )

    # ===== Sparse-agent squares (agent-based) =====
    sparse_l = ax_l.scatter(
        [], [], s=120, marker="s",
        facecolors=(1.0, 0.8, 0.2),
        edgecolors=(1.0, 0.6, 0.0),
        linewidths=1.5,
        alpha=0.4,
        zorder=6,
    )
    sparse_r = ax_r.scatter(
        [], [], s=120, marker="s",
        facecolors=(1.0, 0.8, 0.2),
        edgecolors=(1.0, 0.6, 0.0),
        linewidths=1.5,
        alpha=0.4,
        zorder=6,
    )

    for ax in (ax_l, ax_r):
        ax.set_xlim(pts[0], pts[-1])
        ax.set_ylim(pts[0], pts[-1])

    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    video = cv2.VideoWriter(
        avi_path,
        cv2.VideoWriter_fourcc(*"XVID"),
        10,
        (width, height),
    )

    # -----------------------------
    # Replay loop
    # -----------------------------
    print("[unit_test_03] Writing replay video (forced v_nom + Python CBF) ...")

    for t in range(T):
        # MATLAB "previous" positions for this step (used for overlays)
        xy_prev_nom = x_prev_nom_log[t, :, :]  # (N,2)
        xy_prev_cbf = x_prev_cbf_log[t, :, :]  # (N,2)

        # Forced nominal command from MATLAB
        v_nom_forced = v_nom_log[t, :, :]      # (N,2)

        # Compute Python CBF velocity from same previous CBF positions (keeps this a true replay)
        pos_prev_cbf_3d = np.hstack([xy_prev_cbf, np.zeros((N, 1), dtype=float)])   # (N,3)
        v_nom_forced_3d = np.hstack([v_nom_forced, np.zeros((N, 1), dtype=float)]) # (N,3)

        # (Not plotted directly here, but ensures your Python CBF pipeline stays exercised)
        for m in range(N):
            rho_m, grad_m_3d = coordinator._compute_kde_density_and_gradient(
                pos_prev_cbf_3d, m, planar=True, axes_2d=(0, 1)
            )
            _ = coordinator._apply_cbf_projection(v_nom_forced_3d[m, :], rho_m, grad_m_3d)

        # Display "current" positions = next step's prev
        if t + 1 < T:
            xy_cur_nom = x_prev_nom_log[t + 1, :, :]
            xy_cur_cbf = x_prev_cbf_log[t + 1, :, :]
        else:
            xy_cur_nom = xy_prev_nom
            xy_cur_cbf = xy_prev_cbf

        scatL.set_offsets(xy_cur_nom)
        scatR.set_offsets(xy_cur_cbf)

        # ===== Unsafe region overlay: grid KDE on PREVIOUS positions =====
        rho_grid_nom = compute_kde_local_grid(xy_prev_nom, Xg, Yg, bw=bw, R=R)
        rho_grid_cbf = compute_kde_local_grid(xy_prev_cbf, Xg, Yg, bw=bw, R=R)

        mask_nom = (rho_grid_nom >= eps).astype(float)
        mask_cbf = (rho_grid_cbf >= eps).astype(float)

        maskL.set_data(mask_nom)
        maskL.set_alpha(mask_nom * alphaVal)

        maskR.set_data(mask_cbf)
        maskR.set_alpha(mask_cbf * alphaVal)

        # ===== Sparse squares: agent KDE on PREVIOUS positions (agent-based) =====
        pos_prev_nom_3d = np.hstack([xy_prev_nom, np.zeros((N, 1), dtype=float)])   # (N,3)

        rho_nom_agents = np.zeros((N,), dtype=float)
        rho_cbf_agents = np.zeros((N,), dtype=float)

        for m in range(N):
            rho_m, _ = coordinator._compute_kde_density_and_gradient(
                pos_prev_nom_3d, m, planar=True, axes_2d=(0, 1)
            )
            rho_nom_agents[m] = rho_m

            rho_m, _ = coordinator._compute_kde_density_and_gradient(
                pos_prev_cbf_3d, m, planar=True, axes_2d=(0, 1)
            )
            rho_cbf_agents[m] = rho_m

        sparse_l.set_offsets(xy_prev_nom[rho_nom_agents < eps_min])
        sparse_r.set_offsets(xy_prev_cbf[rho_cbf_agents < eps_min])

        ax_l.set_title(f"Unconstrained, t={t}")
        ax_r.set_title(f"B-CBF Constraint, t={t}")

        fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
        buf = buf.reshape(height, width, 4)
        frame = buf[:, :, [3, 2, 1]]  # ARGB -> BGR
        video.write(frame)

        if (t + 1) % 20 == 0:
            pass
            # quick visibility check
            # print(f"  ...wrote frame t={t} | sparse_nom={int(np.sum(rho_nom_agents < eps_min))} sparse_cbf={int(np.sum(rho_cbf_agents < eps_min))}")

    video.release()
    plt.close(fig)

    print(f"[unit_test_03] Wrote video: {avi_path}")


if __name__ == "__main__":
    import time

    print("==== Start DootCbfCoordinator Unit Tests ====")

    t0 = time.perf_counter()
    unit_test_01()
    t1 = time.perf_counter()
    print(f"[unit_test_01] Complete multiple vehicles - one target unit test in {t1 - t0:.3f} seconds")

    t2 = time.perf_counter()
    unit_test_02()
    t3 = time.perf_counter()
    print(f"[unit_test_02] Complete multiple vehicles - multiple targets w/ and w/o CBF test in {t3 - t2:.3f} seconds")

    t4 = time.perf_counter()
    unit_test_03()
    t5 = time.perf_counter()
    print(f"[unit_test_03] Complete MATLAB log replay comparison in {t5 - t4:.3f} seconds")


    print("==== Finish DootCbfCoordinator Unit Tests ====")