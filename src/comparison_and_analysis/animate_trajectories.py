#!/usr/bin/env python3
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use('Qt5Agg')  # Use Qt5Agg backend for interactive display
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle, Circle
import matplotlib.transforms as transforms
from matplotlib.animation import FuncAnimation
from rich.progress import track
import imageio


def load_rl_data(data_dir):
    """Load RL simulation data."""
    try:
        t = np.load(os.path.join(data_dir, 'RL_time.npy'))
        x = np.load(os.path.join(data_dir, 'RL_x.npy'))
        y = np.load(os.path.join(data_dir, 'RL_y.npy'))
        theta = np.load(os.path.join(data_dir, 'RL_heading.npy'))

        # Reshape to match the format from optimal control (add batch dimension if needed)
        if len(t.shape) == 1:
            t = t.reshape(1, -1)
        if len(x.shape) == 1:
            x = x.reshape(1, -1)
        if len(y.shape) == 1:
            y = y.reshape(1, -1)
        if len(theta.shape) == 1:
            theta = theta.reshape(1, -1)

        # Ensure all arrays have the same length
        min_len = min(t.shape[1], x.shape[1], y.shape[1], theta.shape[1])
        t = t[:, :min_len]
        x = x[:, :min_len]
        y = y[:, :min_len]
        theta = theta[:, :min_len]

        print(f"RL data loaded: time shape {t.shape}, x shape {x.shape}, y shape {y.shape}, theta shape {theta.shape}")

        return {
            'time': t,
            'x': x,
            'y': y,
            'heading': theta,
        }
    except Exception as e:
        print(f"Error loading RL data: {e}")
        return None


def load_optimal_control_data(oc_dir):
    """Load optimal control data for LQR and MT methods."""
    data = {}

    for method in ['LQR', 'MT']:
        try:
            data[method] = {}
            prefix = f"{method}_"

            # Load time and position data
            time_file = os.path.join(oc_dir, f"{prefix}time.npy")
            if not os.path.exists(time_file):
                print(f"Warning: {time_file} not found, skipping {method}")
                continue

            data[method]['time'] = np.load(time_file)
            if len(data[method]['time'].shape) == 1:
                data[method]['time'] = data[method]['time'].reshape(1, -1)

            # Load remaining data with consistency checks
            for name in ['x', 'y', 'theta']:
                file_path = os.path.join(oc_dir, f"{prefix}{name}.npy")
                if os.path.exists(file_path):
                    values = np.load(file_path)
                    # Ensure data has batch dimension
                    if len(values.shape) == 1:
                        values = values.reshape(1, -1)

                    # Ensure same length as time array
                    if values.shape[1] != data[method]['time'].shape[1]:
                        min_len = min(values.shape[1], data[method]['time'].shape[1])
                        values = values[:, :min_len]
                        data[method]['time'] = data[method]['time'][:, :min_len]

                    data[method][name] = values

            print(f"{method} data loaded: time shape {data[method]['time'].shape}")
            if 'x' in data[method]:
                print(f"  x shape {data[method]['x'].shape}")
            if 'y' in data[method]:
                print(f"  y shape {data[method]['y'].shape}")
            if 'theta' in data[method]:
                print(f"  theta shape {data[method]['theta'].shape}")

        except Exception as e:
            print(f"Error loading {method} data: {e}")

    return data


def load_waypoints(waypoints_csv):
    """Load waypoints from CSV file."""
    try:
        print(f"Loading waypoints from {waypoints_csv}")
        df = pd.read_csv(waypoints_csv)
        waypoints = df[['x', 'y']].values
        print(f"Loaded {len(waypoints)} waypoints")
        return waypoints
    except Exception as e:
        print(f"Could not load waypoints from {waypoints_csv}: {e}")
        return None


def create_vehicle_polygon(x, y, theta, scale=0.03):
    """Create a polygon representing a vehicle at the given position and orientation."""
    # Vehicle shape (car-like)
    length = scale * 2
    width = scale

    # Define corners of the vehicle in local coordinates
    corners = np.array([
        [length / 2, width / 2],  # front-right
        [length / 2, -width / 2],  # front-left
        [-length / 2, -width / 2],  # rear-left
        [-length / 2, width / 2]  # rear-right
    ])

    # Rotation matrix
    rot = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])

    # Rotate and translate corners
    rotated = np.dot(corners, rot.T)
    positioned = rotated + np.array([x, y])

    return positioned


def animate_trajectories(rl_data, oc_data, waypoints, output_file='trajectory_animation.mp4',
                         dt=0.1, fps=30, display=False):
    """Create an animation comparing different trajectories."""
    # Setup figure and axes
    fig, ax = plt.subplots(figsize=(10, 8))
    plt.title('Trajectory Comparison')

    # Colors for different methods
    colors = {
        'RL': 'blue',
        'LQR': 'green',
        'MT': 'red'
    }

    # Collect all x/y points for axis limits
    all_x, all_y = [], []

    # Add RL data if available
    if rl_data is not None:
        all_x.extend(rl_data['x'].flatten())
        all_y.extend(rl_data['y'].flatten())

    # Add optimal control data if available
    if oc_data is not None:
        for method in oc_data:
            if 'x' in oc_data[method] and 'y' in oc_data[method]:
                all_x.extend(oc_data[method]['x'].flatten())
                all_y.extend(oc_data[method]['y'].flatten())

    # Add waypoints
    if waypoints is not None:
        # Plot waypoints
        ax.plot(waypoints[:, 0], waypoints[:, 1], 'ko', markersize=8, label='Waypoints')
        all_x.extend(waypoints[:, 0])
        all_y.extend(waypoints[:, 1])

    if not all_x or not all_y:
        print("Error: No valid trajectory data found.")
        return None

    # Set plot limits with margins
    margin = 0.2
    ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
    ax.set_ylim(min(all_y) - margin, max(all_y) + margin)
    ax.set_aspect('equal')
    ax.grid(True)

    # Find maximum time across all trajectories
    max_time = 0
    if rl_data is not None and 'time' in rl_data:
        max_time = max(max_time, np.max(rl_data['time']))

    if oc_data is not None:
        for method in oc_data:
            if 'time' in oc_data[method]:
                max_time = max(max_time, np.max(oc_data[method]['time']))

    # Create uniform time array for interpolation
    uniform_time = np.arange(0, max_time + dt / 2, dt).round(2)
    num_frames = len(uniform_time)

    # Initialize vehicle polygons and trails
    vehicles = {}
    trails = {}

    # Pre-compute interpolated arrays for smoother animation
    rl_interp = {}
    oc_interp = {}

    # RL vehicle
    if rl_data is not None:
        # Unwrap heading for smooth interpolation
        heading_unwrapped = np.unwrap(rl_data['heading'])

        # Initial position
        x0, y0 = rl_data['x'][0, 0], rl_data['y'][0, 0]
        theta0 = rl_data['heading'][0, 0]

        # Create polygon
        vehicles['RL'] = Polygon(
            create_vehicle_polygon(x0, y0, theta0),
            closed=True, fc=colors['RL'], ec='black', alpha=0.7, label='RL Control'
        )
        ax.add_patch(vehicles['RL'])

        # Create trail
        trails['RL'], = ax.plot([], [], '-', color=colors['RL'], linewidth=1.5, alpha=0.5)

        # Pre-compute interpolation
        print(f"Interpolating RL data: time shape {rl_data['time'].shape}, heading shape {heading_unwrapped.shape}")
        rl_interp['x'] = np.interp(uniform_time, rl_data['time'][0], rl_data['x'][0])
        rl_interp['y'] = np.interp(uniform_time, rl_data['time'][0], rl_data['y'][0])
        rl_interp['heading'] = np.interp(uniform_time, rl_data['time'][0], heading_unwrapped[0])

    # Optimal control vehicles (LQR and MT)
    if oc_data is not None:
        for method in oc_data:
            if 'x' in oc_data[method] and 'y' in oc_data[method] and 'time' in oc_data[method]:
                # Get heading from theta
                if 'theta' in oc_data[method]:
                    heading = oc_data[method]['theta']
                    heading_key = 'theta'
                else:
                    heading = np.zeros_like(oc_data[method]['x'])
                    heading_key = None

                # Unwrap heading for smooth interpolation
                heading_unwrapped = np.unwrap(heading)

                # Initial position
                x0, y0 = oc_data[method]['x'][0, 0], oc_data[method]['y'][0, 0]
                theta0 = heading[0, 0]

                # Create polygon
                vehicles[method] = Polygon(
                    create_vehicle_polygon(x0, y0, theta0),
                    closed=True, fc=colors[method], ec='black', alpha=0.7, label=f'{method} Control'
                )
                ax.add_patch(vehicles[method])

                # Create trail
                trails[method], = ax.plot([], [], '-', color=colors[method], linewidth=1.5, alpha=0.5)

                # Pre-compute interpolation
                print(f"Interpolating {method} data: time shape {oc_data[method]['time'].shape}")

                oc_interp[method] = {}
                oc_interp[method]['x'] = np.interp(
                    uniform_time, oc_data[method]['time'][0], oc_data[method]['x'][0]
                )
                oc_interp[method]['y'] = np.interp(
                    uniform_time, oc_data[method]['time'][0], oc_data[method]['y'][0]
                )

                # Store unwrapped heading
                oc_data[method]['heading_unwrapped'] = heading_unwrapped

                if heading_key is not None:
                    oc_interp[method]['heading'] = np.interp(
                        uniform_time, oc_data[method]['time'][0], heading_unwrapped[0]
                    )

    # Add legend
    ax.legend(loc='upper right')

    # Time text display
    time_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, fontsize=10,
                        verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    # Initialize buffer for frames
    frames = []

    # Progress tracking for terminal
    print(f"Rendering {num_frames} animation frames...")

    # Generate all frames
    uniform_time = uniform_time[0:200]
    for idx, t in enumerate(uniform_time):
        if idx % 10 == 0:
            print(f"  Frame {idx}/{num_frames} - t={t:.2f}s")

        # Update time text
        time_text.set_text(f'Time: {t:.2f} s')

        # Update RL vehicle
        if rl_data is not None and 'x' in rl_interp:
            x = rl_interp['x'][idx]
            y = rl_interp['y'][idx]
            heading = rl_interp['heading'][idx]

            # Update polygon
            vehicles['RL'].set_xy(create_vehicle_polygon(x, y, heading))

            # Update trail with all points up to the current frame
            trails['RL'].set_data(rl_interp['x'][:idx + 1], rl_interp['y'][:idx + 1])

        # Update optimal control vehicles
        if oc_data is not None:
            for method in oc_data:
                if method in oc_interp:
                    x = oc_interp[method]['x'][idx]
                    y = oc_interp[method]['y'][idx]

                    if 'heading' in oc_interp[method]:
                        heading = oc_interp[method]['heading'][idx]
                    else:
                        # Compute heading from position if not available
                        if idx > 0:
                            dx = oc_interp[method]['x'][idx] - oc_interp[method]['x'][idx - 1]
                            dy = oc_interp[method]['y'][idx] - oc_interp[method]['y'][idx - 1]
                            heading = np.arctan2(dy, dx)
                        else:
                            heading = 0

                    # Update polygon
                    vehicles[method].set_xy(create_vehicle_polygon(x, y, heading))

                    # Update trail with all points up to the current frame
                    trails[method].set_data(oc_interp[method]['x'][:idx + 1],
                                            oc_interp[method]['y'][:idx + 1])

        # Capture frame
        fig.canvas.draw()
        buf, size = fig.canvas.print_to_buffer()
        w, h = size
        img = np.frombuffer(buf, dtype='uint8').reshape(h, w, 4)[..., :3]
        frames.append(img)

    # Save the animation
    if output_file:
        print(f"Saving animation to {output_file}...")
        imageio.mimsave(output_file, frames, fps=fps)
        print(f"Animation saved to {output_file}")

    # Display the animation if requested
    if display:
        plt.show()
    else:
        plt.close()

    return frames


def parse_arguments():
    parser = argparse.ArgumentParser(description='Generate animation comparing trajectories')
    parser.add_argument('--rl_data_dir', default="src/camera_ready/rl_control/data",
                        help='Directory containing RL data')
    parser.add_argument('--oc_data_dir', default="src/camera_ready/optimal_control/data",
                        help='Directory containing optimal control data')
    parser.add_argument('--waypoints', default="src/camera_ready/analysis/data/square_waypoints.csv",
                        help='CSV file containing waypoints')
    parser.add_argument('--output_dir', default="src/camera_ready/comparison_and_analysis/results",
                        help='Directory to save results')
    parser.add_argument('--output_file', default="trajectory_animation.mp4",
                        help='Name of the output animation file')
    parser.add_argument('--dt', type=float, default=0.1,
                        help='Time step for animation (seconds)')
    parser.add_argument('--fps', type=int, default=30,
                        help='Frames per second for output video')
    return parser.parse_args()


def main():
    # Parse command line arguments
    args = parse_arguments()

    # Set up paths
    output_dir = args.output_dir
    output_file = os.path.join(output_dir, args.output_file)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    waypoints = load_waypoints(args.waypoints)
    rl_data = load_rl_data(args.rl_data_dir)
    oc_data = load_optimal_control_data(args.oc_data_dir)

    # Create animation
    print("Creating animation...")
    animate_trajectories(rl_data, oc_data, waypoints, output_file, dt=args.dt, fps=args.fps)

    print("Done!")


if __name__ == "__main__":
    main()
