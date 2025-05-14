# visualize_ocp.py

import os
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.patches as patches

import src.optimal_control.analysis as analysis

# Constants
WAYPOINT_CAPTURE_RADIUS = 0.1


def load_data(data_dir: str, identifier: str):
    """
    Load state & control trajectories plus their timestamps.
    Returns:
        t: (1xNT) state time vector
        x, y, theta: (1xNT) states
        t_u: (1xNU) control time vector
        u_s, u_h: (1xNU) controls (speed, heading)
    """
    t = np.load(os.path.join(data_dir, f'{identifier}_time.npy'))
    x = np.load(os.path.join(data_dir, f'{identifier}_x.npy'))
    y = np.load(os.path.join(data_dir, f'{identifier}_y.npy'))

    # Handle different naming convention for theta/heading in RL data
    if identifier == "RL" and os.path.exists(os.path.join(data_dir, f'{identifier}_heading.npy')):
        theta = np.load(os.path.join(data_dir, f'{identifier}_heading.npy'))
    else:
        theta = np.load(os.path.join(data_dir, f'{identifier}_theta.npy'))

    t_u = np.load(os.path.join(data_dir, f'{identifier}_time_u.npy'))
    u_s = np.load(os.path.join(data_dir, f'{identifier}_speed.npy'))

    # Handle different naming convention for heading controls in RL data
    if identifier == "RL" and os.path.exists(os.path.join(data_dir, f'{identifier}_steering_command.npy')):
        u_h = np.load(os.path.join(data_dir, f'{identifier}_steering_command.npy'))
    else:
        u_h = np.load(os.path.join(data_dir, f'{identifier}_heading.npy'))

    # Ensure consistent shape - reshape to (1, N) if data is just (N,)
    if len(t.shape) == 1:
        t = t.reshape(1, -1)
    if len(x.shape) == 1:
        x = x.reshape(1, -1)
    if len(y.shape) == 1:
        y = y.reshape(1, -1)
    if len(theta.shape) == 1:
        theta = theta.reshape(1, -1)
    if len(t_u.shape) == 1:
        t_u = t_u.reshape(1, -1)
    if len(u_s.shape) == 1:
        u_s = u_s.reshape(1, -1)
    if len(u_h.shape) == 1:
        u_h = u_h.reshape(1, -1)

    return t, x, y, theta, t_u, u_s, u_h


def visualize_oc_solns():
    ### Load the waypoints (only x,y used for plotting)
    csv_path = "/Users/blake/repos/18.065-Final-Project/src/camera_ready/comparison_and_analysis/race_track_waypoints/square_waypoints.csv"
    race_track_waypoints = pd.read_csv(csv_path).values.tolist()

    ### Load both solutions
    data_dir = "/Users/blake/repos/18.065-Final-Project/src/camera_ready/optimal_control/data"
    t_MT, x_MT, y_MT, th_MT, t_u_MT, u_s_MT, u_h_MT = load_data(data_dir, "MT")
    t_LQR, x_LQR, y_LQR, th_LQR, t_u_LQR, u_s_LQR, u_h_LQR = load_data(data_dir, "LQR")

    ### Load RL solution
    rl_data_dir = "/Users/blake/repos/18.065-Final-Project/src/camera_ready/rl_control/data"
    t_RL, x_RL, y_RL, th_RL, t_u_RL, u_s_RL, u_h_RL = load_data(rl_data_dir, "RL")

    ### Round times for nicer axes
    t_MT = np.round(t_MT, 2)
    t_u_MT = np.round(t_u_MT, 2)
    t_LQR = np.round(t_LQR, 2)
    t_u_LQR = np.round(t_u_LQR, 2)
    t_RL = np.round(t_RL, 2)
    t_u_RL = np.round(t_u_RL, 2)

    ## Calculate actual speeds from position data (L2 norm of velocity)
    # For MT trajectory
    dx_MT = np.diff(x_MT[0])
    dy_MT = np.diff(y_MT[0])
    dt_MT = np.diff(t_MT[0])
    # Distance between consecutive points
    dist_MT = np.sqrt(dx_MT ** 2 + dy_MT ** 2)
    # Speed = distance / time (handle zero time differences to avoid NaN)
    dt_MT = np.maximum(dt_MT, 1e-10)  # Avoid division by zero
    actual_speed_MT = dist_MT / dt_MT

    # For LQR trajectory
    dx_LQR = np.diff(x_LQR[0])
    dy_LQR = np.diff(y_LQR[0])
    dt_LQR = np.diff(t_LQR[0])
    # Distance between consecutive points
    dist_LQR = np.sqrt(dx_LQR ** 2 + dy_LQR ** 2)
    # Speed = distance / time (handle zero time differences to avoid NaN)
    dt_LQR = np.maximum(dt_LQR, 1e-10)  # Avoid division by zero
    actual_speed_LQR = dist_LQR / dt_LQR

    # For RL trajectory
    dx_RL = np.diff(x_RL[0])
    dy_RL = np.diff(y_RL[0])
    dt_RL = np.diff(t_RL[0])
    # Distance between consecutive points
    dist_RL = np.sqrt(dx_RL ** 2 + dy_RL ** 2)
    # Speed = distance / time (handle zero time differences to avoid NaN)
    dt_RL = np.maximum(dt_RL, 1e-10)  # Avoid division by zero
    actual_speed_RL = dist_RL / dt_RL

    # Handle any remaining NaN values
    actual_speed_MT = np.nan_to_num(actual_speed_MT)
    actual_speed_LQR = np.nan_to_num(actual_speed_LQR)
    actual_speed_RL = np.nan_to_num(actual_speed_RL)

    # Find global min/max speed for consistent colormap
    min_speed = min(np.min(actual_speed_MT), np.min(actual_speed_LQR), np.min(actual_speed_RL))
    max_speed_LQR = np.max(actual_speed_LQR)
    max_speed_MT = np.max(actual_speed_MT)
    max_speed_RL = np.max(actual_speed_RL)
    max_speed = max(max_speed_LQR, max_speed_MT, max_speed_RL)

    print(f"Speed ranges - MT: {np.min(actual_speed_MT):.3f} to {np.max(actual_speed_MT):.3f}, "
          f"LQR: {np.min(actual_speed_LQR):.3f} to {np.max(actual_speed_LQR):.3f}, "
          f"RL: {np.min(actual_speed_RL):.3f} to {np.max(actual_speed_RL):.3f}")
    print(f"Global speed range: {min_speed:.3f} to {max_speed:.3f}")

    # Add after calculating speeds
    min_dt = np.min(dt_LQR)
    problematic_indices = np.where(dt_LQR < 1e-5)[0]
    if len(problematic_indices) > 0:
        print(f"Warning: Very small time steps at indices: {problematic_indices}")
        print(f"Minimum dt: {min_dt}")

    max_speed = np.max(actual_speed_LQR)
    problematic_indices = np.where(actual_speed_LQR > 0.05)[0]
    if len(problematic_indices) > 0:
        print(f"Warning: Very large speed at indices: {problematic_indices}")
        print(f"Maximum speed: {max_speed}")

    # Plot speed histograms
    # MT Speed Histogram
    plt.figure(figsize=(10, 6))
    plt.hist(actual_speed_MT, bins=20, alpha=0.7, color='blue')
    plt.title('MT Trajectory - Speed Distribution')
    plt.xlabel('Speed (distance/time)')
    plt.ylabel('Count')
    plt.xlim(0, max_speed_MT * 1.2)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

    # LQR Speed Histogram
    plt.figure(figsize=(10, 6))
    plt.hist(actual_speed_LQR, bins=20, alpha=0.7, color='red')
    plt.title('LQR Trajectory - Speed Distribution')
    plt.xlabel('Speed (distance/time)')
    plt.ylabel('Count')
    plt.xlim(0, max_speed_LQR * 1.2)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

    # RL Speed Histogram
    plt.figure(figsize=(10, 6))
    plt.hist(actual_speed_RL, bins=20, alpha=0.7, color='green')
    plt.title('RL Trajectory - Speed Distribution')
    plt.xlabel('Speed (distance/time)')
    plt.ylabel('Count')
    plt.xlim(0, max_speed_RL * 1.2)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

    ## Plot Paths (x-y plane)
    plt.figure(figsize=(10, 8))

    # Set axis limits
    plt.xlim(min(np.min(x_MT), np.min(x_LQR), np.min(x_RL)) - 0.5,
             max(np.max(x_MT), np.max(x_LQR), np.max(x_RL)) + 0.5)
    plt.ylim(min(np.min(y_MT), np.min(y_LQR), np.min(y_RL)) - 0.5,
             max(np.max(y_MT), np.max(y_LQR), np.max(y_RL)) + 0.5)

    # overlay waypoints as light green circles with radius WAYPOINT_CAPTURE_RADIUS
    ax = plt.gca()
    for p in race_track_waypoints:
        circle = patches.Circle((p[0], p[1]), WAYPOINT_CAPTURE_RADIUS, color='lightgreen', alpha=0.4, zorder=1)
        ax.add_patch(circle)

    # Create points and segments for MT path
    points_MT = np.array([x_MT[0], y_MT[0]]).T.reshape(-1, 1, 2)
    segments_MT = np.concatenate([points_MT[:-1], points_MT[1:]], axis=1)

    # Create points and segments for LQR path
    points_LQR = np.array([x_LQR[0], y_LQR[0]]).T.reshape(-1, 1, 2)
    segments_LQR = np.concatenate([points_LQR[:-1], points_LQR[1:]], axis=1)

    # Create points and segments for RL path
    points_RL = np.array([x_RL[0], y_RL[0]]).T.reshape(-1, 1, 2)
    segments_RL = np.concatenate([points_RL[:-1], points_RL[1:]], axis=1)

    # Create a color map
    norm = plt.Normalize(min_speed, max_speed)
    cmap = plt.cm.viridis

    # Create line collections with actual speeds
    lc_MT = LineCollection(segments_MT, cmap=cmap, norm=norm, zorder=2)
    lc_MT.set_array(actual_speed_MT)  # Use actual speed for coloring
    lc_MT.set_linewidth(10)

    lc_LQR = LineCollection(segments_LQR, cmap=cmap, norm=norm, zorder=2)
    lc_LQR.set_array(actual_speed_LQR)  # Use actual speed for coloring
    lc_LQR.set_linewidth(10)

    lc_RL = LineCollection(segments_RL, cmap=cmap, norm=norm, zorder=2)
    lc_RL.set_array(actual_speed_RL)  # Use actual speed for coloring
    lc_RL.set_linewidth(10)

    # Add the collections to the plot
    line_MT = plt.gca().add_collection(lc_MT)
    line_LQR = plt.gca().add_collection(lc_LQR)
    line_RL = plt.gca().add_collection(lc_RL)

    # Add thin black lines to mark the paths for legend
    plt.plot(x_MT[0], y_MT[0], 'b-', alpha=0.5, linewidth=2, label=f'MT path (Time: {t_MT[0][-1]:.2f}s)', zorder=3)
    plt.plot(x_LQR[0], y_LQR[0], 'r-', alpha=0.5, linewidth=2, label=f'LQR path (Time: {t_LQR[0][-1]:.2f}s)', zorder=3)
    plt.plot(x_RL[0], y_RL[0], 'g-', alpha=0.5, linewidth=2, label=f'RL path (Time: {t_RL[0][-1]:.2f}s)', zorder=3)

    # Add colorbar
    cbar = plt.colorbar(line_MT)
    cbar.set_label('Actual Speed (distance/time)')

    # Add text annotations for final times
    plt.figtext(0.5, 0.01, f'Final Times - MT: {t_MT[0][-1]:.2f}s, LQR: {t_LQR[0][-1]:.2f}s, RL: {t_RL[0][-1]:.2f}s',
                ha='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.8))

    plt.legend()
    plt.title('Trajectories & Waypoints (colored by actual speed)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis('equal')
    plt.show()

    # ### Plot Controls
    # # Speed
    # plt.figure()
    # plt.plot(t_u_MT[0],  u_s_MT[0],  'k.-', label='MT speed')
    # plt.plot(t_u_LQR[0], u_s_LQR[0], 'r.-', label='LQR speed')
    # plt.plot(t_u_RL[0], u_s_RL[0], 'g.-', label='RL speed')
    # plt.legend()
    # plt.title('Speed Command')
    # plt.xlabel('Time (s)')
    # plt.show()

    # ## Plot Paths (time vs. x)
    # plt.figure()
    # plt.plot(t_MT[0],  x_MT[0],  'k.-', label='MT path')
    # plt.plot(t_LQR[0], x_LQR[0], 'r.-', label='LQR path')
    # plt.plot(t_RL[0], x_RL[0], 'g.-', label='RL path')
    # plt.title('Trajectories & Waypoints')
    # plt.xlabel('Time (s)')
    # plt.ylabel('x')
    # plt.show()

    # ### Animate both cars together
    # # Stack into shape (3, NT) and (3, NU) to include RL
    # ts = np.vstack([t_MT, t_LQR, t_RL])
    # xs = np.vstack([x_MT, x_LQR, x_RL])
    # ys = np.vstack([y_MT, y_LQR, y_RL])
    # thetas = np.vstack([th_MT, th_LQR, th_RL])
    # # analysis.animateMultipleCars(ts, xs, ys, thetas, race_track_waypoints)
    # # ts = ts[:, :100]
    # # xs = xs[:, :100]
    # # ys = ys[:, :100]
    # # thetas = thetas[:, :100]
    # analysis.animateMultipleCars_to_mp4(ts, xs, ys, thetas, race_track_waypoints, output_path='src/camera_ready/optimal_control/data/animation.mp4')

    print("Complete")


if __name__ == "__main__":
    visualize_oc_solns()
