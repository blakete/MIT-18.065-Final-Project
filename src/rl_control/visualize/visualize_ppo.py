import argparse
import os
from pathlib import Path
import sys

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib import animation

from rl_control.train.torch_env import VectorizedRaceBikeDojo
from rl_control.train.ppo_agent import PPOAgent


def visualize_ppo_agent(model_path=None, random_agent=False, num_episodes=3, max_steps=200, deterministic=True, hidden_dim=64, success_frames=100):
    """Visualize the trained PPO agent or a random agent"""
    # Set up environment (single instance)
    device = torch.device("cpu")  # Visualization works best on CPU
    env = VectorizedRaceBikeDojo(num_envs=1, device=device, max_steps=max_steps)
    
    # Initialize agent with specified hidden dimension
    agent = PPOAgent(obs_dim=4, action_dim=2, device=device, hidden_dim=hidden_dim).to(device)
    
    # Set up output directory for saving GIFs
    if random_agent:
        vis_dir = "."
        model_filename = "random_agent"
        print("Using random agent as requested")
    elif model_path:
        # Extract directory and filename from model path
        model_dir = os.path.dirname(model_path)
        model_filename = os.path.basename(model_path).replace('.pt', '')
        
        # Create visualizations directory
        vis_dir = os.path.join(model_dir, "visualizations")
        os.makedirs(vis_dir, exist_ok=True)
        print(f"Visualizations will be saved to: {vis_dir}")
        
        # Load model
        try:
            # Load model, ensuring it's on the CPU for visualization
            state_dict = torch.load(model_path, map_location=device, weights_only=True)
            agent.load_state_dict(state_dict)
            print(f"Loaded agent from {model_path}")
            agent.eval()
        except Exception as e:
            print(f"ERROR: Failed to load agent from {model_path}: {e}")
            print("To run with a random agent, use the --random-agent flag")
            sys.exit(1)
    else:
        print("ERROR: Either --model or --random-agent must be specified")
        sys.exit(1)
    
    # Run visualization for each episode
    for episode in range(num_episodes):
        # Reset environment
        obs = env.reset()
        
        # Log actual capture threshold
        capture_threshold = env.capture_threshold
        print(f"Environment capture threshold: {capture_threshold}")
        
        # Set up visualization
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_aspect('equal')
        ax.grid(True)
        ax.set_title(f'Race Bike Episode {episode+1}')
        
        # Create objects for visualization
        # The bike size should be accurate for collision detection
        bike_radius = 0.05  # This should match what the environment uses for collision
        bike = plt.Circle((0, 0), bike_radius, fc='blue', alpha=0.7)
        
        # The waypoint size should match the capture threshold
        waypoint_x, waypoint_y = env.waypoints[0, 0].item(), env.waypoints[0, 1].item()
        waypoint = plt.Circle((waypoint_x, waypoint_y), capture_threshold, fc='green', alpha=0.3)
        
        heading_line = plt.Line2D([0, 0.1], [0, 0], lw=2, color='red')
        speed_line = plt.Line2D([0, 0.1], [0, 0], lw=5, color='green')

        # Add circle to represent the unit circle
        unit_circle = plt.Circle((0, 0), 1.0, fill=False, color='gray', linestyle='--')
        ax.add_patch(unit_circle)
        
        ax.add_patch(bike)
        ax.add_patch(waypoint)
        ax.add_line(speed_line)
        ax.add_line(heading_line)
        
        # Create trail to show bike's path
        trail_x, trail_y = [], []
        trail_line, = ax.plot([], [], 'b-', alpha=0.5, lw=1)
        
        # Text for stats and success message
        stats_text = ax.text(0.02, 0.8, '', transform=ax.transAxes)
        success_text = ax.text(0.5, 0.5, '', transform=ax.transAxes, 
                               fontsize=24, weight='bold', ha='center', va='center',
                               color='red', alpha=0)
        
        # Store trajectories for animation
        bike_positions = []
        bike_headings = []
        bike_speeds = []
        waypoint_positions = []
        trails = []
        rewards = []
        actions = []
        show_success_frames = []  # Track which frames show success message
        
        # Run episode
        done = False
        total_reward = 0
        step = 0
        waypoints_captured = 0
        capture_step = -1  # Track which step had a capture
        
        while not done and step < max_steps:
            print("Step: ", step)
            print("Obs: ")
            print("\t Dx: ", obs[0, 0].item())
            print("\t Dy: ", obs[0, 1].item())
            print("\t Heading: ", obs[0, 2].item())
            print("\t Speed: ", obs[0, 3].item())
            
            # Calculate distance to waypoint for explicit capture detection
            bike_pos = env.state[0, :2]
            waypoint_pos = env.waypoints[0]
            dist_to_waypoint = torch.sqrt(torch.sum((waypoint_pos - bike_pos)**2))
            is_capture = dist_to_waypoint < env.capture_threshold
            
            # Log detailed position information for debugging
            print(f"Bike position: [{bike_pos[0].item():.4f}, {bike_pos[1].item():.4f}]")
            print(f"Waypoint position: [{waypoint_pos[0].item():.4f}, {waypoint_pos[1].item():.4f}]")
            print(f"Distance: {dist_to_waypoint.item():.4f}, Threshold: {env.capture_threshold}")
            
            # Check if we've just captured the waypoint
            if is_capture and capture_step < 0:
                capture_step = step
                waypoints_captured += 1
                print(f"CAPTURE DETECTED at step {step}! Distance: {dist_to_waypoint.item():.4f}")
                print(f"Bike position: [{bike_pos[0].item():.4f}, {bike_pos[1].item():.4f}]")
                print(f"Waypoint position: [{waypoint_pos[0].item():.4f}, {waypoint_pos[1].item():.4f}]")
                print(f"Capture threshold: {env.capture_threshold}")
            
            # Get action from policy
            with torch.no_grad():
                if deterministic:
                    # Get mean action first
                    mean_action = agent.actor_mean(obs)
                    
                    # Pass the mean action to get_action_and_value to get properly
                    # transformed action without sampling (deterministic)
                    # action, logprob, entropy, value = agent.get_action_and_value(obs, action=mean_action)
                    scaled_action, logprob, entropy, value, _ = agent.get_action_and_value(obs, raw_action=mean_action)

                    print("Action (deterministic): ")
                    print("\t δθ: ", scaled_action[0][0].item())
                    print("\t δv: ", scaled_action[0][1].item())

                    if step == 0:
                        print("Using deterministic actions (policy mean)")
                else:
                    # Use stochastic actions (sample from distribution)
                    # action, logprob, entropy, value = agent.get_action_and_value(obs)
                    scaled_action, logprob, entropy, value, _ = agent.get_action_and_value(obs)
                    
                    print("Action (stochastic): ")
                    print("\t δθ: ", scaled_action[0][0].item())
                    print("\t δv: ", scaled_action[0][1].item())

                    if step == 0:
                        print("Using stochastic actions (sampling from policy)")
            
            # Take step in environment (only if not captured yet)
            if capture_step < 0:
                next_obs, reward, dones, _, _ = env.step(scaled_action)
                reward_value = reward.item()
                total_reward += reward_value
                
                # Check if environment says we're done (timeout or other terminal condition)
                if dones.item():
                    print(f"Episode ending from environment at step {step} - likely timeout")
                    done = True
            else:
                # If we've already captured, just use the same observation
                next_obs = obs
                reward_value = 0
                
                # End episode after showing success message for the specified number of frames
                if step - capture_step >= success_frames:
                    done = True
            
            # Store trajectory data for animation
            bike_pos = env.state[0, :2].cpu().numpy()
            bike_heading = env.state[0, 2].item()
            bike_speed = env.state[0, 3].item()
            waypoint_pos = env.waypoints[0].cpu().numpy()
            
            # If capture occurred, freeze the bike's position and stop it
            if capture_step >= 0:
                if step == capture_step:
                    # First frame after capture - use current position
                    frozen_pos = bike_pos
                    frozen_heading = bike_heading
                    frozen_waypoint_pos = waypoint_pos
                else:
                    # Use position at capture time
                    frozen_pos = bike_positions[capture_step]
                    frozen_heading = bike_headings[capture_step]
                    frozen_waypoint_pos = waypoint_positions[capture_step]
                
                bike_positions.append(frozen_pos)
                bike_headings.append(frozen_heading)
                bike_speeds.append(0.0)  # Stop the bike
                waypoint_positions.append(frozen_waypoint_pos)
                actions.append(np.zeros(2))  # Zero action during success state
            else:
                bike_positions.append(bike_pos)
                bike_headings.append(bike_heading)
                bike_speeds.append(bike_speed)
                waypoint_positions.append(waypoint_pos)
                actions.append(scaled_action[0].cpu().numpy())
            
            # Only add to trail if not captured yet
            if capture_step < 0:
                trail_x.append(bike_pos[0])
                trail_y.append(bike_pos[1])
            trails.append((trail_x.copy(), trail_y.copy()))
            rewards.append(total_reward)
            
            # Mark this frame as showing success message if after capture
            show_success_frames.append(capture_step >= 0)
            
            # Update for next iteration
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
            speed_line.set_data(
                [bike_positions[i][0], bike_positions[i][0] + bike_speeds[i] * np.cos(heading)],
                [bike_positions[i][1], bike_positions[i][1] + bike_speeds[i] * np.sin(heading)]
            )

            # Update waypoint
            waypoint.center = waypoint_positions[i]
            
            # Update trail
            trail_line.set_data(trails[i])
            
            # Update stats text with actions
            if i < len(actions):
                action_text = f"Action: [δθ={actions[i][0]:.4f}, δv={actions[i][1]:.4f}]"
            else:
                action_text = ""
                
            # Get bike and waypoint positions for this frame
            bike_x, bike_y = bike_positions[i]
            waypoint_x, waypoint_y = waypoint_positions[i]
            speed = bike_speeds[i]
            
            # Calculate distance to waypoint
            dist_to_waypoint = np.sqrt((bike_x - waypoint_x)**2 + (bike_y - waypoint_y)**2)
            
            # Combine all information in the stats_text
            stats_text.set_text(f'Step: {i+1}\nReward: {rewards[i]:.4f}\n{action_text}\n'
                              f'Bike: x={bike_x:.4f}, y={bike_y:.4f}, h={heading:.4f}, v={speed:.4f}\n'
                              f'Waypoint: x={waypoint_x:.4f}, y={waypoint_y:.4f}\n'
                              f'Distance: {dist_to_waypoint:.3f}')
            
            # Update success text
            if i < len(show_success_frames) and show_success_frames[i]:
                success_text.set_text("SUCCESS!")
                success_text.set_alpha(1.0)
            else:
                success_text.set_alpha(0)
            
            return bike, waypoint, heading_line, trail_line, stats_text, success_text
        
        # Create animation
        anim = animation.FuncAnimation(
            fig, animate, frames=num_frames, interval=50, blit=True
        )
        
        # Save animation to the visualizations directory
        output_path = os.path.join(vis_dir, f'{model_filename}_episode_{episode+1}.gif')
        try:
            anim.save(output_path, writer='pillow', fps=20)
            print(f"Animation saved to {output_path}")
        except Exception as e:
            print(f"Failed to save animation: {e}")
            
        plt.close()
        
        print(f"Episode {episode+1} completed with total reward: {total_reward:.4f}")
        print(f"Waypoints captured: {waypoints_captured}")
        print("---------------------------------------")
        
        # Print capture information if a waypoint was captured
        if capture_step >= 0:
            print("\n===== CAPTURE INFORMATION =====")
            print(f"Capture detected at step {capture_step}")
            captured_bike_pos = bike_positions[capture_step]
            captured_waypoint_pos = waypoint_positions[capture_step]
            captured_dist = np.sqrt((captured_bike_pos[0] - captured_waypoint_pos[0])**2 + 
                                   (captured_bike_pos[1] - captured_waypoint_pos[1])**2)
            print(f"Bike position at capture: [{captured_bike_pos[0]:.4f}, {captured_bike_pos[1]:.4f}]")
            print(f"Waypoint position at capture: [{captured_waypoint_pos[0]:.4f}, {captured_waypoint_pos[1]:.4f}]")
            print(f"Distance at capture: {captured_dist:.4f}")
            print(f"Capture threshold: {env.capture_threshold}")
            print("===============================\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize PPO Race Bike Agent')
    parser.add_argument('--model', type=str, default=None, help='Path to model file')
    parser.add_argument('--random-agent', action='store_true', help='Use a random agent instead of a trained model')
    parser.add_argument('--episodes', type=int, default=3, help='Number of episodes to visualize')
    parser.add_argument('--steps', type=int, default=200, help='Maximum steps per episode')
    parser.add_argument('--stochastic', action='store_true', help='Use stochastic actions instead of deterministic ones')
    parser.add_argument('--hidden-dim', type=int, default=64, help='Hidden dimension of network (must match training value)')
    parser.add_argument('--success-frames', type=int, default=15, help='Number of frames to show success message')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.model and not args.random_agent:
        print("ERROR: Either --model or --random-agent must be specified")
        sys.exit(1)
    
    visualize_ppo_agent(
        model_path=args.model, 
        random_agent=args.random_agent,
        num_episodes=args.episodes, 
        max_steps=args.steps,
        deterministic=not args.stochastic,
        hidden_dim=args.hidden_dim,
        success_frames=args.success_frames
    ) 