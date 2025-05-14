#!/usr/bin/env python3

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate
import numpy.linalg as la

# Constants
WAYPOINT_CAPTURE_RADIUS = 0.1


def load_data(data_dir: str, identifier: str):
    """
    Load state & control trajectories and their timestamps
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


def calculate_total_race_time(t):
    """Calculate total race time from time array"""
    return t[0][-1]


def calculate_waypoint_times(x, y, t, waypoints):
    """
    Calculate the time at which each waypoint is reached
    Returns a list of times and the time differences between waypoints
    """
    waypoint_times = []
    current_wp_idx = 0

    for i in range(len(t[0])):
        # Check distance to current waypoint
        if current_wp_idx < len(waypoints):
            wp = waypoints[current_wp_idx]
            dist = np.sqrt((x[0][i] - wp[0]) ** 2 + (y[0][i] - wp[1]) ** 2)

            if dist < WAYPOINT_CAPTURE_RADIUS:
                waypoint_times.append(t[0][i])
                current_wp_idx += 1

    # Calculate time differences between waypoints
    if len(waypoint_times) > 1:
        waypoint_time_diffs = np.diff(waypoint_times)
        return waypoint_times, waypoint_time_diffs

    return waypoint_times, []


def calculate_control_effort(u_s, u_h, t_u):
    """
    Calculate control effort as the integral of squared control inputs.
    Normalize speed and heading controls to make them comparable.
    """
    # Normalize controls to [0,1] range for fair comparison
    u_s_norm = u_s[0] / np.max(np.abs(u_s[0])) if np.max(np.abs(u_s[0])) > 0 else u_s[0]
    u_h_norm = u_h[0] / np.max(np.abs(u_h[0])) if np.max(np.abs(u_h[0])) > 0 else u_h[0]

    # Calculate time differences for integration
    dt = np.diff(t_u[0])
    dt = np.append(dt, dt[-1])  # Duplicate last dt for matching dimensions

    # Calculate sum of squared control inputs, weighted by time step
    speed_effort = np.sum(u_s_norm ** 2 * dt)
    heading_effort = np.sum(u_h_norm ** 2 * dt)
    total_effort = speed_effort + heading_effort

    return total_effort, speed_effort, heading_effort


def point_to_line_distance(point, line_start, line_end):
    """Calculate perpendicular distance from a point to a line segment"""
    # Vector from line_start to line_end
    line_vec = line_end - line_start
    line_len = la.norm(line_vec)
    if line_len == 0:
        return la.norm(point - line_start)

    # Unit vector in direction of line
    line_unit_vec = line_vec / line_len

    # Vector from line_start to point
    pnt_vec = point - line_start

    # Project pnt_vec onto line_unit_vec
    line_proj = np.dot(pnt_vec, line_unit_vec)

    # If projection is outside line segment, use distance to nearest endpoint
    if line_proj < 0:
        return la.norm(point - line_start)
    elif line_proj > line_len:
        return la.norm(point - line_end)

    # Calculate perpendicular distance to line
    closest_point = line_start + line_proj * line_unit_vec
    return la.norm(point - closest_point)


def calculate_path_deviation(x, y, waypoints):
    """
    Calculate sum of squared distances from the straight line paths between waypoints
    Returns the total squared deviation and a list of squared deviations per segment
    """
    if len(waypoints) < 2:
        return 0, []

    total_sq_deviation = 0
    segment_deviations = []

    # Iterate through waypoint segments
    for i in range(len(waypoints) - 1):
        wp_start = np.array(waypoints[i])
        wp_end = np.array(waypoints[i + 1])

        # Find trajectory points in this segment
        # For simplicity, we'll use euclidean distance to determine which points belong to which segment
        sq_deviations = []

        for j in range(len(x[0])):
            point = np.array([x[0][j], y[0][j]])

            # Check if point is close to this segment
            dist_to_start = la.norm(point - wp_start)
            dist_to_end = la.norm(point - wp_end)

            # If point is closer to this segment than the next or previous, include it
            if (i == 0 or dist_to_start < la.norm(point - np.array(waypoints[i - 1]))) and \
                    (i == len(waypoints) - 2 or dist_to_end < la.norm(point - np.array(waypoints[i + 2]))):
                # Calculate perpendicular distance to line segment
                dist = point_to_line_distance(point, wp_start, wp_end)
                sq_deviations.append(dist ** 2)

        # Add squared deviations for this segment
        if sq_deviations:
            segment_deviation = np.sum(sq_deviations)
            segment_deviations.append(segment_deviation)
            total_sq_deviation += segment_deviation

    return total_sq_deviation, segment_deviations


def main():
    # Load waypoints
    csv_path = "/Users/blake/repos/18.065-Final-Project/src/camera_ready/comparison_and_analysis/race_track_waypoints/square_waypoints.csv"
    race_track_waypoints = pd.read_csv(csv_path).values.tolist()

    # Load trajectory data
    oc_data_dir = "/Users/blake/repos/18.065-Final-Project/src/camera_ready/optimal_control/data"
    rl_data_dir = "/Users/blake/repos/18.065-Final-Project/src/camera_ready/rl_control/data"

    # Load MT data
    t_MT, x_MT, y_MT, th_MT, t_u_MT, u_s_MT, u_h_MT = load_data(oc_data_dir, "MT")

    # Load LQR data
    t_LQR, x_LQR, y_LQR, th_LQR, t_u_LQR, u_s_LQR, u_h_LQR = load_data(oc_data_dir, "LQR")

    # Load RL data
    t_RL, x_RL, y_RL, th_RL, t_u_RL, u_s_RL, u_h_RL = load_data(rl_data_dir, "RL")

    # Create result dictionary
    results = {
        "MT": {},
        "LQR": {},
        "RL": {}
    }

    # Calculate metrics for each method
    for method, t, x, y, t_u, u_s, u_h in [
        ("MT", t_MT, x_MT, y_MT, t_u_MT, u_s_MT, u_h_MT),
        ("LQR", t_LQR, x_LQR, y_LQR, t_u_LQR, u_s_LQR, u_h_LQR),
        ("RL", t_RL, x_RL, y_RL, t_u_RL, u_s_RL, u_h_RL)
    ]:
        # 1. Total race time
        total_time = calculate_total_race_time(t)
        results[method]["total_time"] = total_time

        # 2. Waypoint times
        wp_times, wp_time_diffs = calculate_waypoint_times(x, y, t, race_track_waypoints)
        results[method]["waypoint_times"] = wp_times
        results[method]["waypoint_time_diffs"] = wp_time_diffs

        # 3. Control effort
        total_effort, speed_effort, heading_effort = calculate_control_effort(u_s, u_h, t_u)
        results[method]["total_control_effort"] = total_effort
        results[method]["speed_control_effort"] = speed_effort
        results[method]["heading_control_effort"] = heading_effort

        # 4. Path deviation
        total_deviation, segment_deviations = calculate_path_deviation(x, y, race_track_waypoints)
        results[method]["total_path_deviation"] = total_deviation
        results[method]["segment_path_deviations"] = segment_deviations

    # Print results as a table
    metrics_table = []
    headers = ["Metric", "MT", "LQR", "RL"]

    # Add total race time
    metrics_table.append(["Total Race Time (s)",
                          f"{results['MT']['total_time']:.2f}",
                          f"{results['LQR']['total_time']:.2f}",
                          f"{results['RL']['total_time']:.2f}"])

    # Add waypoints reached
    metrics_table.append(["Waypoints Reached",
                          f"{len(results['MT']['waypoint_times'])}/{len(race_track_waypoints)}",
                          f"{len(results['LQR']['waypoint_times'])}/{len(race_track_waypoints)}",
                          f"{len(results['RL']['waypoint_times'])}/{len(race_track_waypoints)}"])

    # Add average time per waypoint (if applicable)
    mt_avg_wp_time = np.mean(results['MT']['waypoint_time_diffs']) if len(
        results['MT']['waypoint_time_diffs']) > 0 else "N/A"
    lqr_avg_wp_time = np.mean(results['LQR']['waypoint_time_diffs']) if len(
        results['LQR']['waypoint_time_diffs']) > 0 else "N/A"
    rl_avg_wp_time = np.mean(results['RL']['waypoint_time_diffs']) if len(
        results['RL']['waypoint_time_diffs']) > 0 else "N/A"

    metrics_table.append(["Avg. Time per Waypoint (s)",
                          f"{mt_avg_wp_time:.2f}" if isinstance(mt_avg_wp_time, float) else mt_avg_wp_time,
                          f"{lqr_avg_wp_time:.2f}" if isinstance(lqr_avg_wp_time, float) else lqr_avg_wp_time,
                          f"{rl_avg_wp_time:.2f}" if isinstance(rl_avg_wp_time, float) else rl_avg_wp_time])

    # Add control effort
    metrics_table.append(["Total Control Effort",
                          f"{results['MT']['total_control_effort']:.4f}",
                          f"{results['LQR']['total_control_effort']:.4f}",
                          f"{results['RL']['total_control_effort']:.4f}"])

    metrics_table.append(["Speed Control Effort",
                          f"{results['MT']['speed_control_effort']:.4f}",
                          f"{results['LQR']['speed_control_effort']:.4f}",
                          f"{results['RL']['speed_control_effort']:.4f}"])

    metrics_table.append(["Heading Control Effort",
                          f"{results['MT']['heading_control_effort']:.4f}",
                          f"{results['LQR']['heading_control_effort']:.4f}",
                          f"{results['RL']['heading_control_effort']:.4f}"])

    # Add path deviation
    metrics_table.append(["Total Path Deviation",
                          f"{results['MT']['total_path_deviation']:.4f}",
                          f"{results['LQR']['total_path_deviation']:.4f}",
                          f"{results['RL']['total_path_deviation']:.4f}"])

    # Print the table
    print(tabulate(metrics_table, headers=headers, tablefmt="grid"))

    # Plot path deviations per segment for each method
    plt.figure(figsize=(12, 6))

    # Get maximum number of segments
    max_segments = max(len(results['MT']['segment_path_deviations']),
                       len(results['LQR']['segment_path_deviations']),
                       len(results['RL']['segment_path_deviations']))

    segment_indices = np.arange(1, max_segments + 1)

    # Pad segment deviations to have equal length
    mt_deviations = np.pad(results['MT']['segment_path_deviations'],
                           (0, max_segments - len(results['MT']['segment_path_deviations'])),
                           'constant', constant_values=0)
    lqr_deviations = np.pad(results['LQR']['segment_path_deviations'],
                            (0, max_segments - len(results['LQR']['segment_path_deviations'])),
                            'constant', constant_values=0)
    rl_deviations = np.pad(results['RL']['segment_path_deviations'],
                           (0, max_segments - len(results['RL']['segment_path_deviations'])),
                           'constant', constant_values=0)

    plt.bar(segment_indices - 0.2, mt_deviations, width=0.2, label='MT', color='blue', alpha=0.7)
    plt.bar(segment_indices, lqr_deviations, width=0.2, label='LQR', color='red', alpha=0.7)
    plt.bar(segment_indices + 0.2, rl_deviations, width=0.2, label='RL', color='green', alpha=0.7)

    plt.xlabel('Waypoint Segment')
    plt.ylabel('Squared Path Deviation')
    plt.title('Path Deviation per Waypoint Segment')
    plt.legend()
    plt.xticks(segment_indices)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

    # Plot waypoint times for each method
    plt.figure(figsize=(12, 6))

    # Get maximum number of waypoints
    max_waypoints = max(len(results['MT']['waypoint_times']),
                        len(results['LQR']['waypoint_times']),
                        len(results['RL']['waypoint_times']))

    waypoint_indices = np.arange(1, max_waypoints + 1)

    # Pad waypoint times to have equal length
    mt_times = results['MT']['waypoint_times'] + [np.nan] * (max_waypoints - len(results['MT']['waypoint_times']))
    lqr_times = results['LQR']['waypoint_times'] + [np.nan] * (max_waypoints - len(results['LQR']['waypoint_times']))
    rl_times = results['RL']['waypoint_times'] + [np.nan] * (max_waypoints - len(results['RL']['waypoint_times']))

    plt.plot(waypoint_indices, mt_times, 'o-', label='MT', color='blue')
    plt.plot(waypoint_indices, lqr_times, 'o-', label='LQR', color='red')
    plt.plot(waypoint_indices, rl_times, 'o-', label='RL', color='green')

    plt.xlabel('Waypoint Number')
    plt.ylabel('Time (s)')
    plt.title('Time to Reach Each Waypoint')
    plt.legend()
    plt.xticks(waypoint_indices)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
