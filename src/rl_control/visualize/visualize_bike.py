import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib import animation
import argparse
from torch_env import VectorizedRaceBikeDojo
from train_bike import PolicyNetwork


def visualize_policy(policy_path=None, num_episodes=3, max_steps=200):
    """Visualize the trained policy or a random policy"""
    # Set up environment (single instance)
    device = torch.device("cpu")  # Visualization works best on CPU
    env = VectorizedRaceBikeDojo(num_envs=1, device=device)
    
    # Initialize policy (either load from file or create random)
    policy = PolicyNetwork().to(device)
    if policy_path:
        try:
            policy.load_state_dict(torch.load(policy_path, map_location=device))
            print(f"Loaded policy from {policy_path}")
            policy.eval()
        except:
            print(f"Failed to load policy from {policy_path}, using random policy")
    else:
        print("Using random policy")
    
    # Run visualization for each episode
    for episode in range(num_episodes):
        # Reset environment
        obs = env.reset()
        
        # Set up visualization
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_aspect('equal')
        ax.grid(True)
        ax.set_title(f'Race Bike Episode {episode+1}')
        
        # Create objects for visualization
        bike = plt.Circle((0, 0), 0.05, fc='blue', alpha=0.7)
        waypoint = plt.Circle((env.waypoints[0, 0].item(), env.waypoints[0, 1].item()), 
                             env.capture_threshold, fc='green', alpha=0.3)
        heading_line = plt.Line2D([0, 0.1], [0, 0], lw=2, color='red')
        
        ax.add_patch(bike)
        ax.add_patch(waypoint)
        ax.add_line(heading_line)
        
        # Create trail to show bike's path
        trail_x, trail_y = [], []
        trail_line, = ax.plot([], [], 'b-', alpha=0.5, lw=1)
        
        # Text for stats
        stats_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
        
        # Store trajectories for animation
        bike_positions = []
        bike_headings = []
        waypoint_positions = []
        trails = []
        rewards = []
        
        # Run episode
        done = False
        total_reward = 0
        step = 0
        
        while not done and step < max_steps:
            # Get action from policy
            with torch.no_grad():
                action = policy(obs)
            
            # Take step in environment
            next_obs, reward, dones, _, _ = env.step(action)
            done = dones.item()
            total_reward += reward.item()
            
            # Store trajectory data for animation
            bike_pos = env.state[0, :2].cpu().numpy()
            bike_heading = env.state[0, 2].item()
            waypoint_pos = env.waypoints[0].cpu().numpy()
            
            bike_positions.append(bike_pos)
            bike_headings.append(bike_heading)
            waypoint_positions.append(waypoint_pos)
            
            trail_x.append(bike_pos[0])
            trail_y.append(bike_pos[1])
            trails.append((trail_x.copy(), trail_y.copy()))
            rewards.append(total_reward)
            
            # Update observation
            obs = next_obs
            step += 1
        
        # Create animation
        num_frames = len(bike_positions)
        
        def animate(i):
            # Update bike position
            bike.center = bike_positions[i]
            
            # Update heading line
            heading = bike_headings[i]
            heading_line.set_data(
                [bike_positions[i][0], bike_positions[i][0] + 0.1 * np.cos(heading)],
                [bike_positions[i][1], bike_positions[i][1] + 0.1 * np.sin(heading)]
            )
            
            # Update waypoint
            waypoint.center = waypoint_positions[i]
            
            # Update trail
            trail_line.set_data(trails[i])
            
            # Update stats text
            stats_text.set_text(f'Step: {i+1}\nReward: {rewards[i]:.2f}')
            
            return bike, waypoint, heading_line, trail_line, stats_text
        
        # Create animation
        anim = animation.FuncAnimation(
            fig, animate, frames=num_frames, interval=50, blit=True
        )
        
        # Save animation
        anim.save(f'race_bike_episode_{episode+1}.gif', writer='pillow', fps=20)
        plt.close()
        
        print(f"Episode {episode+1} completed with total reward: {total_reward:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize Race Bike Policy')
    parser.add_argument('--policy', type=str, default=None, help='Path to policy model file')
    parser.add_argument('--episodes', type=int, default=3, help='Number of episodes to visualize')
    
    args = parser.parse_args()
    
    visualize_policy(policy_path=args.policy, num_episodes=args.episodes) 
