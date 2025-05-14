#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use('Qt5Agg')  # Use Qt5Agg backend for interactive display
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
import matplotlib.transforms as transforms
from rich.progress import track
import imageio


def load_rl_data(data_dir):
    """Load RL simulation data."""
    t = np.load(os.path.join(data_dir, 'RL_time.npy'))
    x = np.load(os.path.join(data_dir, 'RL_x.npy'))
    y = np.load(os.path.join(data_dir, 'RL_y.npy'))
    theta = np.load(os.path.join(data_dir, 'RL_heading.npy'))
    t_u = np.load(os.path.join(data_dir, 'RL_time_u.npy'))
    u_s = np.load(os.path.join(data_dir, 'RL_speed_command.npy'))
    u_h = np.load(os.path.join(data_dir, 'RL_steering_command.npy'))

    # Reshape to match the format from optimal control
    t = t.reshape(1, -1)
    x = x.reshape(1, -1)
    y = y.reshape(1, -1)
    theta = theta.reshape(1, -1)
    t_u = t_u.reshape(1, -1)
    u_s = u_s.reshape(1, -1)
    u_h = u_h.reshape(1, -1)

    return t, x, y, theta, t_u, u_s, u_h


def load_oc_data(data_dir, identifier):
    """
    Load state & control trajectories plus their timestamps.
    Returns:
        t     : (1×NT) state time vector
        x, y, theta : (1×NT) states
        t_u   : (1×NU) control time vector
        u_s, u_h   : (1×NU) controls (speed, heading)
    """
    t = np.load(os.path.join(data_dir, f'{identifier}_time.npy'))
    x = np.load(os.path.join(data_dir, f'{identifier}_x.npy'))
    y = np.load(os.path.join(data_dir, f'{identifier}_y.npy'))
    theta = np.load(os.path.join(data_dir, f'{identifier}_theta.npy'))
    t_u = np.load(os.path.join(data_dir, f'{identifier}_time_u.npy'))
    u_s = np.load(os.path.join(data_dir, f'{identifier}_speed.npy'))
    u_h = np.load(os.path.join(data_dir, f'{identifier}_heading.npy'))
    return t, x, y, theta, t_u, u_s, u_h


def animate_multiple_trajectories(ts, xs, ys, thetas, path, labels, dt=0.1, fps=30,
                                  output_path='comparison_animation.mp4'):
    """
    Generates an animation comparing multiple trajectory solutions.
    
    Parameters:
    - ts, xs, ys, thetas: arrays with shape (n_trajectories, n_steps)
    - path: list of (x,y) waypoints
    - labels: list of labels for each trajectory
    - dt: time step for interpolation
    - fps: frames per second in output video
    - output_path: where to save the animation
    """
    n_trajectories = ts.shape[0]

    # Build uniform time vector
    t_max = np.max([t.max() for t in ts])
    uniform_t = np.arange(0.0, t_max + dt / 2, dt).round(2)
    n_frames = len(uniform_t)

    # Unwrap angles for smooth interpolation
    thetas_un = np.array([np.unwrap(th) for th in thetas])

    # Precompute interpolated trajectories
    xs_i = np.zeros((n_trajectories, n_frames))
    ys_i = np.zeros((n_trajectories, n_frames))
    ths_i = np.zeros((n_trajectories, n_frames))

    for i in range(n_trajectories):
        xs_i[i] = np.interp(uniform_t, ts[i][0], xs[i][0], right=xs[i][0, -1])
        ys_i[i] = np.interp(uniform_t, ts[i][0], ys[i][0], right=ys[i][0, -1])
        ths_i[i] = np.interp(uniform_t, ts[i][0], thetas_un[i][0], right=thetas_un[i][0, -1])

    # Set up figure
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_aspect('equal')

    # Determine plot bounds
    all_x = np.concatenate([x[0] for x in xs])
    all_y = np.concatenate([y[0] for y in ys])
    margin = 0.2
    x_min, x_max = all_x.min() - margin, all_x.max() + margin
    y_min, y_max = all_y.min() - margin, all_y.max() + margin

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    # Draw waypoints
    for px, py in path:
        ax.add_patch(Circle((px, py), 0.05, facecolor='none',
                            edgecolor='green', linewidth=1.5, alpha=0.8))

    # Colors and styles for each trajectory
    colors = ['blue', 'green', 'red']
    markers = ['o', 's', '^']

    # Create vehicle patches
    car_length, car_width = 0.1, 0.05
    patches, fronts, trails = [], [], []

    for i in range(n_trajectories):
        # Add vehicle rectangle
        rect = Rectangle((-car_length / 2, -car_width / 2), car_length, car_width,
                         fc=colors[i], alpha=0.7, label=labels[i])
        ax.add_patch(rect)
        patches.append(rect)

        # Add front marker
        front = Circle((car_length / 2, 0), 0.01, fc='black')
        ax.add_patch(front)
        fronts.append(front)

        # Add trail line
        trail, = ax.plot([], [], '-', color=colors[i], linewidth=1.5, alpha=0.5)
        trails.append(trail)

    # Add legend and title
    ax.legend(loc='upper right')
    title = ax.text(0.5, 1.02, '', transform=ax.transAxes, ha='center')

    # Initialize frame buffer
    frames = []

    # Generate all frames
    for idx, t in track(enumerate(uniform_t), total=n_frames, description="Rendering animation frames"):
        # Update title with current time
        title.set_text(f"Time: {t:.2f} s")

        # Update each vehicle
        for i in range(n_trajectories):
            # Update vehicle transform
            trans = (transforms.Affine2D()
                     .rotate(ths_i[i, idx])
                     .translate(xs_i[i, idx], ys_i[i, idx])
                     + ax.transData)

            patches[i].set_transform(trans)
            fronts[i].set_transform(trans)

            # Update trail
            trails[i].set_data(xs_i[i, :idx + 1], ys_i[i, :idx + 1])

        # Capture frame
        fig.canvas.draw()
        buf, size = fig.canvas.print_to_buffer()
        w, h = size
        img = np.frombuffer(buf, dtype='uint8').reshape(h, w, 4)[..., :3]
        frames.append(img)

    plt.close(fig)

    # Create and save MP4
    print(f"Saving animation to {output_path}...")
    imageio.mimsave(output_path, frames, fps=fps)
    print(f"Animation saved to {output_path}")

    return frames


def plot_comparisons(rl_data, lqr_data, mt_data, waypoints, output_dir):
    """Generate static comparison plots."""
    os.makedirs(output_dir, exist_ok=True)

    # Extract data
    t_rl, x_rl, y_rl, th_rl, t_u_rl, u_s_rl, u_h_rl = rl_data
    t_lqr, x_lqr, y_lqr, th_lqr, t_u_lqr, u_s_lqr, u_h_lqr = lqr_data
    t_mt, x_mt, y_mt, th_mt, t_u_mt, u_s_mt, u_h_mt = mt_data

    # 1. Trajectory comparison
    plt.figure(figsize=(10, 8))
    plt.plot(x_rl[0], y_rl[0], '-', color='blue', linewidth=2, label='RL')
    plt.plot(x_lqr[0], y_lqr[0], '-', color='green', linewidth=2, label='LQR')
    plt.plot(x_mt[0], y_mt[0], '-', color='red', linewidth=2, label='MT')

    # Plot waypoints
    wp_x = [p[0] for p in waypoints]
    wp_y = [p[1] for p in waypoints]
    plt.plot(wp_x, wp_y, 'go', markersize=8, label='Waypoints')

    plt.title('Trajectory Comparison')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.savefig(os.path.join(output_dir, 'trajectory_comparison.png'), dpi=300)
    plt.close()

    # 2. Speed vs. Time
    plt.figure(figsize=(10, 6))
    plt.plot(t_rl[0], u_s_rl[0], '-', color='blue', linewidth=2, label='RL')
    plt.plot(t_lqr[0], u_s_lqr[0], '-', color='green', linewidth=2, label='LQR')
    plt.plot(t_mt[0], u_s_mt[0], '-', color='red', linewidth=2, label='MT')

    plt.title('Speed Command Comparison')
    plt.xlabel('Time (s)')
    plt.ylabel('Speed Command')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'speed_comparison.png'), dpi=300)
    plt.close()

    # 3. Steering vs. Time
    plt.figure(figsize=(10, 6))
    plt.plot(t_rl[0], u_h_rl[0], '-', color='blue', linewidth=2, label='RL')
    plt.plot(t_lqr[0], u_h_lqr[0], '-', color='green', linewidth=2, label='LQR')
    plt.plot(t_mt[0], u_h_mt[0], '-', color='red', linewidth=2, label='MT')

    plt.title('Steering Command Comparison')
    plt.xlabel('Time (s)')
    plt.ylabel('Steering Command')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'steering_comparison.png'), dpi=300)
    plt.close()

    print(f"Saved comparison plots to {output_dir}")


def main():
    # Set paths here (modify as needed)
    rl_data_dir = "src/camera_ready/rl_control/data"
    oc_data_dir = "src/camera_ready/optimal_control/data"
    waypoints_csv = "src/camera_ready/analysis/data/square_waypoints.csv"
    output_dir = "src/camera_ready/comparison_and_analysis/results"
    animation_path = os.path.join(output_dir, "comparison_animation.mp4")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load waypoints
    waypoints = pd.read_csv(waypoints_csv).values.tolist()

    # Load data
    rl_data = load_rl_data(rl_data_dir)
    lqr_data = load_oc_data(oc_data_dir, "LQR")
    mt_data = load_oc_data(oc_data_dir, "MT")

    # Round times for nicer axes
    rl_data[0] = np.round(rl_data[0], 2)  # State time
    rl_data[4] = np.round(rl_data[4], 2)  # Control time
    lqr_data[0] = np.round(lqr_data[0], 2)
    lqr_data[4] = np.round(lqr_data[4], 2)
    mt_data[0] = np.round(mt_data[0], 2)
    mt_data[4] = np.round(mt_data[4], 2)

    # Generate static comparison plots
    plot_comparisons(rl_data, lqr_data, mt_data, waypoints, output_dir)

    # Create animation
    # Stack time, x, y, theta for animation
    ts = np.vstack([rl_data[0], lqr_data[0], mt_data[0]])
    xs = np.vstack([rl_data[1], lqr_data[1], mt_data[1]])
    ys = np.vstack([rl_data[2], lqr_data[2], mt_data[2]])
    thetas = np.vstack([rl_data[3], lqr_data[3], mt_data[3]])

    # Labels for legend
    labels = ["RL Control", "LQR Control", "Min Time Control"]

    # Generate animation
    animate_multiple_trajectories(ts, xs, ys, thetas, waypoints, labels,
                                  dt=0.1, fps=30, output_path=animation_path)

    print(f"All visualizations complete. Animation saved to {animation_path}")


if __name__ == "__main__":
    main()
