#!/usr/bin/env python3
import argparse
import os
import numpy as np
import pandas as pd
import torch

from rl_control.train.torch_env import VectorizedRaceBikeDojo
from rl_control.train.ppo_agent import PPOAgent


def run_rl_simulation(rl_actor_model_path, waypoint_csv, max_steps, device):
    # Load RL agent checkpoint
    ckpt = torch.load(rl_actor_model_path, map_location=device)
    hidden_dim = ckpt['actor_mean.0.weight'].shape[0]
    agent = PPOAgent(obs_dim=4, hidden_dim=hidden_dim, action_dim=2, device=device)
    agent.load_state_dict(ckpt)
    agent.to(device).eval()

    # Load waypoints
    df = pd.read_csv(waypoint_csv)
    waypoints = df[['x','y']].values.astype(np.float32)
    num_wps = len(waypoints)
    wp_idx = 0

    # Create environment without auto-reset
    env = VectorizedRaceBikeDojo(num_envs=1, device=device, max_steps=max_steps)
    env.reset_if_done = lambda done_mask: None
    env.reset()
    env.waypoints = torch.from_numpy(waypoints[0]).to(env.device).unsqueeze(0)

    # Initialize relative observation
    gx, gy, gth, gsp = env.state[0].cpu().numpy()
    dx = waypoints[0,0] - gx
    dy = waypoints[0,1] - gy
    ang = np.arctan2(dy, dx) - gth
    ang = np.arctan2(np.sin(ang), np.cos(ang))
    obs = torch.from_numpy(np.array([dx, dy, ang, gsp], dtype=np.float32)).unsqueeze(0).to(device)

    # Pre-allocate state and control buffer
    states = []
    controls = []
    
    for step in range(max_steps):
        with torch.no_grad():
            action, *_ = agent.get_action_and_value(obs)
            
        # Store control action (convert from tensor to numpy)
        controls.append(action[0].cpu().numpy())

        _, _, _, _, state = env.step(action)
        gx, gy, gth, gsp = env.state[0].cpu().numpy()
        states.append([gx, gy, gth, gsp])

        # Check if current waypoint reached
        dx = waypoints[wp_idx,0] - gx
        dy = waypoints[wp_idx,1] - gy
        if np.hypot(dx, dy) < env.capture_threshold:
            if wp_idx < num_wps - 1:
                wp_idx += 1
                env.waypoints = torch.from_numpy(waypoints[wp_idx]).to(env.device).unsqueeze(0)
            else:
                break

        # Update observation for next step
        ang = np.arctan2(dy, dx) - gth
        ang = np.arctan2(np.sin(ang), np.cos(ang))
        obs = torch.from_numpy(np.array([dx, dy, ang, gsp], dtype=np.float32)).unsqueeze(0).to(device)

    states = np.array(states)
    controls = np.array(controls)
    
    # Get dt from environment or use default
    dt = getattr(env, 'dt', 0.05)
    
    # Generate time arrays (similar to optimal control format)
    time = np.arange(len(states)) * dt
    time_u = np.arange(len(controls)) * dt
    
    return states, controls, time, time_u, dt


def save_data(states, controls, time, time_u, output_dir):
    """Save RL simulation data in the same format as optimal control data."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save state information
    np.save(os.path.join(output_dir, 'RL_x.npy'), states[:, 0])
    np.save(os.path.join(output_dir, 'RL_y.npy'), states[:, 1])
    np.save(os.path.join(output_dir, 'RL_heading.npy'), states[:, 2])
    np.save(os.path.join(output_dir, 'RL_speed.npy'), states[:, 3])
    np.save(os.path.join(output_dir, 'RL_time.npy'), time)
    
    # Save control information 
    np.save(os.path.join(output_dir, 'RL_speed_command.npy'), controls[:, 0])
    np.save(os.path.join(output_dir, 'RL_steering_command.npy'), controls[:, 1])
    np.save(os.path.join(output_dir, 'RL_time_u.npy'), time_u)
    
    # Save full state and control arrays for convenience
    np.save(os.path.join(output_dir, 'RL_states.npy'), states)
    np.save(os.path.join(output_dir, 'RL_controls.npy'), controls)
    
    print(f"Saved RL simulation data to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Generate RL State Data')
    parser.add_argument('--rl_actor_model', required=True, help='RL model checkpoint (.pt)')
    parser.add_argument('--waypoints', required=True, help='CSV of waypoints')
    parser.add_argument('--max_steps', type=int, default=2000, help='Max RL steps')
    parser.add_argument('--device', choices=['cpu','cuda'], default='cpu', help='Device')
    parser.add_argument('--output_dir', default='rl_data', help='Output directory for RL data')
    args = parser.parse_args()

    device = torch.device(args.device)
    states, controls, time, time_u, dt = run_rl_simulation(
        args.rl_actor_model, 
        args.waypoints, 
        args.max_steps, 
        device
    )
    
    save_data(states, controls, time, time_u, args.output_dir)
    print(f"Time step (dt): {dt}")
    print(f"Total simulation time: {time[-1]:.2f} seconds")
    print(f"Number of states: {len(states)}")
    print(f"Number of control actions: {len(controls)}")


if __name__ == '__main__':
    main()
