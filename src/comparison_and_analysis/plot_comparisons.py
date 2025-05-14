#!/usr/bin/env python3
import os
import numpy as np
import matplotlib.pyplot as plt


def load_optimal_control_data(oc_dir):
    oc = {}
    for var in ['LQR', 'MT']:
        prefix = var + '_'
        for name in ['x', 'y', 'heading', 'speed', 'theta', 'time', 'time_u']:
            fname = f"{prefix}{name}.npy"
            path = os.path.join(oc_dir, fname)
            if os.path.isfile(path):
                oc[prefix + name] = np.load(path)
    return oc


def plot_comparisons(states_rl, time_rl, oc, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # Trajectory
    plt.figure()
    plt.plot(states_rl[:, 0], states_rl[:, 1], label='RL')
    if 'LQR_x' in oc and 'LQR_y' in oc:
        plt.plot(oc['LQR_x'], oc['LQR_y'], '--', label='LQR')
    if 'MT_x' in oc and 'MT_y' in oc:
        plt.plot(oc['MT_x'], oc['MT_y'], ':', label='MT')
    plt.xlabel('X');
    plt.ylabel('Y');
    plt.legend();
    plt.title('Trajectory')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'trajectory.png'))
    plt.close()

    # Heading vs time
    plt.figure()
    plt.plot(time_rl, states_rl[:, 2], label='RL')
    if 'LQR_time' in oc and 'LQR_heading' in oc:
        plt.plot(oc['LQR_time'], oc['LQR_heading'], '--', label='LQR')
    if 'MT_time' in oc and 'MT_heading' in oc:
        plt.plot(oc['MT_time'], oc['MT_theta'], ':', label='MT')
    plt.xlabel('Time');
    plt.ylabel('Heading');
    plt.legend();
    plt.title('Heading vs Time')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'heading.png'))
    plt.close()

    # Speed vs time
    plt.figure()
    plt.plot(time_rl, states_rl[:, 3], label='RL')
    if 'LQR_time' in oc and 'LQR_speed' in oc:
        plt.plot(oc['LQR_time'], oc['LQR_speed'], '--', label='LQR')
    if 'MT_time' in oc and 'MT_speed' in oc:
        plt.plot(oc['MT_time'], oc['MT_speed'], ':', label='MT')
    plt.xlabel('Time');
    plt.ylabel('Speed');
    plt.legend();
    plt.title('Speed vs Time')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'speed.png'))
    plt.close()


def main():
    # Set paths here instead of using command line arguments
    rl_states_path = "rl_states.npy"  # Path to RL states numpy file
    oc_dir = "optimal_control_data"  # Directory containing optimal control npy files
    plots_dir = "plots"  # Output directory for plots
    dt = 0.05  # Time step (adjust as needed)

    # Load RL states
    states_rl = np.load(rl_states_path)
    print(f"Loaded RL states from {rl_states_path}")

    # Load optimal control data
    oc_data = load_optimal_control_data(oc_dir)

    # Generate time array
    time_rl = np.arange(states_rl.shape[0]) * dt

    # Create comparison plots
    plot_comparisons(states_rl, time_rl, oc_data, plots_dir)
    print(f"Saved comparison plots to {plots_dir}")


if __name__ == '__main__':
    main()
