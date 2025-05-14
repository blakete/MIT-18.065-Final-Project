import time
import os
import logging
from datetime import datetime
from pathlib import Path
import argparse
import io

import numpy as np
import torch
import wandb
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib import animation
from PIL import Image

from rl_control.train.torch_env import VectorizedRaceBikeDojo
from rl_control.train.ppo_agent import PPOAgent, compute_gae


def setup_experiment_dir(base_path="/Users/blake/repos/18.065-Final-Project/experiment_runs"):
    """Create experiment directory with timestamp and set up logging"""
    # Create timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create experiment directory
    exp_dir = Path(base_path) / f"ppo_bike_{timestamp}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up logging
    log_file = exp_dir / "training.log"
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()  # Also log to console
        ]
    )
    
    logger = logging.getLogger("ppo_bike_training")
    logger.info(f"Created experiment directory at {exp_dir}")
    
    return exp_dir, logger


def create_agent_visualization(agent, env, device, deterministic=True, max_steps=200):
    """
    Create a visualization of the agent's behavior in the environment

    Parameters:
        agent: The agent to visualize
        env: The environment to run the agent in (not used, we create a new one)
        device: The device to run the agent on (will be overridden to CPU)
        deterministic: Whether to use deterministic actions
        max_steps: Maximum number of steps to run

    Returns:
        frames: List of frames (PIL Images) showing the agent's behavior
        total_reward: Total reward achieved in the episode
        waypoints_captured: Number of waypoints captured
    """
    # Force CPU for visualization to avoid device mismatch issues
    viz_device = torch.device("cpu")

    # Create a new agent instance on CPU and load the state dict
    agent_cpu = PPOAgent(
        obs_dim=4,
        hidden_dim=agent.actor_mean[0].weight.shape[0],
        action_dim=2,
        device=viz_device
    ).to(viz_device)

    # Load parameters onto CPU agent
    cpu_state_dict = {k: v.cpu() for k, v in agent.state_dict().items()}
    agent_cpu.load_state_dict(cpu_state_dict)

    # Create a fresh single-env on CPU
    viz_env = VectorizedRaceBikeDojo(num_envs=1, device=viz_device, max_steps=max_steps)
    obs = viz_env.reset().to(viz_device)

    # Set up plotting
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
    ax.grid(True)
    ax.set_title('Race Bike Visualization')

    bike = Circle((0, 0), 0.05, fc='blue', alpha=0.7)
    waypoint_x = viz_env.waypoints[0, 0].item()
    waypoint_y = viz_env.waypoints[0, 1].item()
    waypoint = Circle((waypoint_x, waypoint_y), viz_env.capture_threshold, fc='green', alpha=0.3)
    heading_line = plt.Line2D([0, 0.1], [0, 0], lw=2, color='red')
    speed_line   = plt.Line2D([0, 0.1], [0, 0], lw=5, color='green')
    unit_circle  = Circle((0, 0), 1.0, fill=False, color='gray', linestyle='--')

    ax.add_patch(unit_circle)
    ax.add_patch(bike)
    ax.add_patch(waypoint)
    ax.add_line(heading_line)
    ax.add_line(speed_line)

    trail_x, trail_y = [], []
    trail_line, = ax.plot([], [], 'b-', alpha=0.5, lw=1)
    stats_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

    frames = []
    total_reward = 0.0
    waypoints_captured = 0
    step = 0
    done = False

    while not done and step < max_steps:
        with torch.no_grad():
            if deterministic:
                # Use the policy mean as the raw sample for a deterministic rollout
                mean_raw = agent_cpu.actor_mean(obs)
                scaled_action, logprob, entropy, value, raw_action = \
                    agent_cpu.get_action_and_value(obs, raw_action=mean_raw)
            else:
                # Stochastic sampling
                scaled_action, logprob, entropy, value, raw_action = \
                    agent_cpu.get_action_and_value(obs)

        # Step the env with the scaled action
        if scaled_action.device != viz_device:
            scaled_action = scaled_action.to(viz_device)
        next_obs, reward, dones, _, _ = viz_env.step(scaled_action)

        done = dones.item()
        total_reward += reward.item()

        # Update visualization elements
        bike_pos     = viz_env.state[0, :2].cpu().numpy()
        bike_heading = viz_env.state[0, 2].item()
        bike_speed   = viz_env.state[0, 3].item()
        waypoint_pos = viz_env.waypoints[0].cpu().numpy()

        trail_x.append(bike_pos[0])
        trail_y.append(bike_pos[1])

        bike.center    = bike_pos
        waypoint.center= waypoint_pos
        heading_line.set_data(
            [bike_pos[0], bike_pos[0] + 0.1 * np.cos(bike_heading)],
            [bike_pos[1], bike_pos[1] + 0.1 * np.sin(bike_heading)]
        )
        speed_line.set_data(
            [bike_pos[0], bike_pos[0] + bike_speed * np.cos(bike_heading)],
            [bike_pos[1], bike_pos[1] + bike_speed * np.sin(bike_heading)]
        )
        trail_line.set_data(trail_x, trail_y)

        action_text = f"Action: [δθ={scaled_action[0,0].item():.2f}, Speed={scaled_action[0,1].item():.2f}]"
        stats_text.set_text(f"Step: {step+1}\nReward: {total_reward:.2f}\n{action_text}")

        # Capture frame
        fig.canvas.draw()
        frame = Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
        frames.append(frame)

        obs = next_obs.to(viz_device)
        step += 1

    plt.close(fig)
    return frames, total_reward, waypoints_captured


def create_animated_gif(frames, output_path=None):
    """Create an animated gif from a list of frames"""
    if output_path:
        frames[0].save(
            output_path,
            save_all=True,
            append_images=frames[1:],
            optimize=False,
            duration=50,
            loop=0
        )
        return output_path
    else:
        # Save to bytes buffer if no output path specified
        buffer = io.BytesIO()
        frames[0].save(
            buffer,
            format='GIF',
            save_all=True,
            append_images=frames[1:],
            optimize=False,
            duration=50,
            loop=0
        )
        buffer.seek(0)
        return buffer


def train_ppo(
    seed=1,
    num_envs=8,
    learning_rate=3e-4,
    total_timesteps=1_000_000,
    num_steps=2048,
    gamma=0.99,
    gae_lambda=0.95,
    clip_ratio=0.2,
    update_epochs=10,
    minibatch_size=128,
    hidden_dim=64,
    entropy_coef=0.05,
    max_grad_norm=0.5,
    device='cpu',
    use_wandb=True,
    wandb_project="ppo-bike-training",
    wandb_entity=None,
    exp_name="PPO-BikeRacer",
    exp_dir=None,
    logger=None,
    visualization_interval=5,  # How often to create and log visualizations (in updates)
    max_steps=200  # Maximum steps per episode
):
    """
    Train a PPO agent on the race bike environment
    
    Parameters:
        seed: random seed
        num_envs: number of parallel environments
        learning_rate: learning rate
        total_timesteps: total number of timesteps
        num_steps: number of steps per rollout
        update_epochs: number of PPO update epochs
        minibatch_size: minibatch size
        hidden_dim: hidden dimension of the policy network
        device: device to run the training on
        use_wandb: whether to use wandb for logging
        wandb_project: wandb project name
        wandb_entity: wandb entity name
        exp_name: experiment name
        exp_dir: experiment directory
        logger: logger instance
        visualization_interval: how often to create and log visualizations (in updates)
        max_steps: maximum steps per episode
    """
    logger.info(f"Training PPO agent on {num_envs} environments with {total_timesteps} timesteps")
    logger.info(f"Max steps per episode: {max_steps}")
    logger.info("---------------------------------------")
    
    # Initialize wandb if enabled
    if use_wandb:
        # Convert exp_dir to string and get absolute path for better traceability
        exp_dir_str = str(exp_dir.absolute())
        
        run = wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            name=exp_name,
            config={
                "seed": seed,
                "num_envs": num_envs,
                "learning_rate": learning_rate,
                "total_timesteps": total_timesteps,
                "num_steps": num_steps,
                "gamma": gamma,
                "gae_lambda": gae_lambda,
                "clip_ratio": clip_ratio,
                "update_epochs": update_epochs,
                "minibatch_size": minibatch_size,
                "hidden_dim": hidden_dim,
                "device": device,
                "experiment_dir": exp_dir_str,  # Log the full experiment directory path
                "env_max_steps": max_steps,  # Add max_steps for clarity in wandb
            },
            sync_tensorboard=True,
            monitor_gym=True,
            dir=exp_dir_str  # Save wandb files in experiment directory
        )
        
        # Also log the experiment directory as a wandb summary field for easy access
        wandb.run.summary["experiment_dir"] = exp_dir_str
        logger.info(f"Logged experiment directory to wandb: {exp_dir_str}")
    
    # Set random seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Initialize environment with specified max_steps
    logger.info(f"Using device: {device}")
    env = VectorizedRaceBikeDojo(num_envs=num_envs, device=device, logger=logger, max_steps=max_steps)
    env_max_steps = env.max_steps  # Get max steps from environment
    
    # Log the max steps for debugging
    logger.info(f"Environment max steps: {env_max_steps}")
    
    # Initialize agent
    agent = PPOAgent(
        obs_dim=4,  # [relative_x, relative_y, relative_heading, speed]
        hidden_dim=hidden_dim,
        action_dim=2,  # [delta_heading, delta_speed]
        learning_rate=learning_rate,
        clip_ratio=clip_ratio,
        entropy_coef=entropy_coef,
        max_grad_norm=max_grad_norm,
        device=device
    ).to(device)
    
    # Explicitly log the device placement of agent and its components
    logger.info(f"Agent on device: {next(agent.parameters()).device}")
    
    # Calculate number of updates
    num_updates = total_timesteps // (num_steps * num_envs)
    
    # Initialize arrays for storing episode data
    obs = torch.zeros((num_steps, num_envs, 4)).to(device)
    actions = torch.zeros((num_steps, num_envs, 2)).to(device)
    logprobs = torch.zeros((num_steps, num_envs)).to(device)
    rewards = torch.zeros((num_steps, num_envs)).to(device)
    dones = torch.zeros((num_steps, num_envs)).to(device)
    values = torch.zeros((num_steps, num_envs)).to(device)
    
    # Initialize metrics
    global_step = 0
    start_time = time.time()
    episode_rewards = []
    episode_lengths = []
    
    # For tracking ongoing episode rewards and lengths
    current_episode_rewards = torch.zeros(num_envs, device=device)
    current_episode_lengths = torch.zeros(num_envs, dtype=torch.int, device=device)
    
    # New variables to track episode steps within the env object itself
    # This allows us to accurately determine when episodes hit the max step limit
    env_steps = torch.zeros(num_envs, dtype=torch.int, device=device)
    
    # Done flags for each environment - to track which episodes completed in this rollout
    env_done_flags = torch.zeros(num_envs, dtype=torch.bool, device=device)
    
    # Models directory
    models_dir = exp_dir / "models"
    models_dir.mkdir(exist_ok=True)
    
    # Visualization directory
    vis_dir = exp_dir / "visualizations"
    vis_dir.mkdir(exist_ok=True)
    
    # Counters for tracking timeout episodes vs successful captures
    timeout_episodes = 0
    capture_episodes = 0
    
    # Training loop - iterate number of episodes
    for update in range(1, num_updates + 1):
        logger.info("---------------------------------------")
        logger.info(f"Episode: {update}/{num_updates}")
        
        # Verify agent is on the correct device at the start of each update
        agent_device = next(agent.parameters()).device
        if agent_device != device:
            logger.warning(f"Agent device mismatch at start of update! On {agent_device}, should be on {device}")
            # Move agent to the correct device
            agent = agent.to(device)
            logger.info(f"Agent moved to {next(agent.parameters()).device}")
        
        # Start environment
        next_obs = env.reset()
        # Explicitly ensure observations are on the correct device
        next_obs = next_obs.to(device)
        
        next_done = torch.zeros(num_envs, dtype=torch.bool, device=device)
        
        # Reset any done flags from previous update
        env_done_flags.fill_(False)
        
        # Reset environment steps counter for this rollout
        env_steps.fill_(0)
    
        # Learning rate anneal with sqrt decay instead of linear
        frac = (1.0 - (update - 1.0) / num_updates) ** 0.5
        lrnow = frac * learning_rate
        agent.optimizer.param_groups[0]["lr"] = lrnow
        
        # Collect rollout data for each episode
        for step in range(0, num_steps):
            global_step += num_envs
            
            # Store current observation, done flag
            obs[step] = next_obs
            dones[step] = next_done
            
            # Get action from policy
            with torch.no_grad():
                # Double-check device consistency
                if next_obs.device != device:
                    logger.warning(f"Device mismatch! next_obs on {next_obs.device}, expected {device}")
                    next_obs = next_obs.to(device)
                
                if agent.actor_mean[0].weight.device != device:
                    logger.warning(f"Agent device mismatch! Agent on {agent.actor_mean[0].weight.device}, expected {device}")
                    agent = agent.to(device)
                
                # action, logprob, _, value = agent.get_action_and_value(next_obs)
                scaled_action, logprob, _, value, raw_action = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            
            # actions[step] = action
            actions[step] = raw_action       # store *pre-squash* sample
            logprobs[step] = logprob
            
            # Execute action in environment
            next_obs, reward, next_done, info, _ = env.step(scaled_action)
            
            # Ensure tensors from environment are on the correct device
            if next_obs.device != device:
                next_obs = next_obs.to(device)
            if reward.device != device:
                reward = reward.to(device)
            if next_done.device != device:
                next_done = next_done.to(device)
            
            rewards[step] = reward
            
            # Increment environment step counters BEFORE processing dones
            # This ensures we don't increment counters for environments that just reset
            env_steps += 1  # This correctly tracks steps in each env
            
            # Update episode statistics for active (not yet done) environments
            current_episode_rewards += reward
            current_episode_lengths += 1
            
            # Track which environments have completed an episode during this rollout
            env_done_flags = torch.logical_or(env_done_flags, next_done)
            
            # Record completed episode stats for this step and RESET counters
            # This must be done BEFORE incrementing counters for the next step
            for i, done in enumerate(next_done):
                if done:
                    # Check if episode terminated due to timeout or waypoint capture
                    is_timeout = env_steps[i] >= env_max_steps
                    is_capture = not is_timeout  # If not timeout, must be capture
                    
                    # Make sure episode length doesn't exceed max steps
                    actual_episode_length = min(current_episode_lengths[i].item(), env_max_steps)
                    
                    # Only record episode stats for newly completed episodes
                    episode_rewards.append(current_episode_rewards[i].item())
                    episode_lengths.append(actual_episode_length)
                    
                    # Increment the appropriate counter
                    if is_timeout:
                        timeout_episodes += 1
                    else:
                        capture_episodes += 1
                    
                    # Calculate the ratio of captures to total episodes
                    total_episodes = timeout_episodes + capture_episodes
                    capture_ratio = capture_episodes / max(1, total_episodes)
                    
                    # Log detailed episode information to wandb
                    if use_wandb:
                        wandb.log({
                            "episode_reward": current_episode_rewards[i].item(),
                            "episode_length": actual_episode_length,
                            "timeout_episode": 1 if is_timeout else 0,
                            "capture_episode": 1 if is_capture else 0,
                            "capture_ratio": capture_ratio,  # Log the capture ratio for each episode
                            "global_step": global_step
                        })
                    
                    # Log detailed info to console for debugging
                    if i == 0:  # Only log for first environment to avoid spam
                        reason = "timeout" if is_timeout else "capture"
                        logger.debug(f"Episode completed ({reason}) - Length: {actual_episode_length}, Reward: {current_episode_rewards[i].item():.2f}")
                    
                    # Reset episode trackers for this environment
                    current_episode_rewards[i] = 0
                    current_episode_lengths[i] = 0
                    env_steps[i] = 0
        
        # Compute returns and advantages
        with torch.no_grad():
            # Get value prediction for final state
            next_value = agent.get_value(next_obs).flatten()
            
            # Option 1: GAE calculation on GPU (if possible)
            if device.type == 'cuda':
                try:
                    # Try to perform GAE calculation on GPU
                    advantages = torch.zeros_like(rewards)
                    last_gae_lam = 0
                    
                    for t in reversed(range(num_steps)):
                        if t == num_steps - 1:
                            next_non_terminal = 1.0 - next_done.float()
                            next_val = next_value
                        else:
                            next_non_terminal = 1.0 - dones[t + 1].float()
                            next_val = values[t + 1]
                            
                        delta = rewards[t] + gamma * next_val * next_non_terminal - values[t]
                        last_gae_lam = delta + gamma * gae_lambda * next_non_terminal * last_gae_lam
                        advantages[t] = last_gae_lam
                    
                    returns = advantages + values
                except Exception as e:
                    logger.warning(f"GPU GAE calculation failed, falling back to CPU: {e}")
                    # Fall back to CPU calculation
                    rewards_np = rewards.cpu().numpy()
                    values_np = values.cpu().numpy()
                    dones_np = dones.cpu().numpy()
                    next_value_np = next_value.cpu().numpy()
                    
                    advantages_np, returns_np = compute_gae(
                        rewards_np, values_np, dones_np, next_value_np, gamma, gae_lambda
                    )
                    
                    advantages = torch.FloatTensor(advantages_np).to(device)
                    returns = torch.FloatTensor(returns_np).to(device)
            else:
                # CPU calculation
                rewards_np = rewards.cpu().numpy()
                values_np = values.cpu().numpy()
                dones_np = dones.cpu().numpy()
                next_value_np = next_value.cpu().numpy()
                
                advantages_np, returns_np = compute_gae(
                    rewards_np, values_np, dones_np, next_value_np, gamma, gae_lambda
                )
                
                advantages = torch.FloatTensor(advantages_np).to(device)
                returns = torch.FloatTensor(returns_np).to(device)
        
        # Flatten rollout data
        b_obs = obs.reshape((-1, 4))
        b_actions = actions.reshape((-1, 2))
        b_logprobs = logprobs.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        
        # Update the policy
        metrics = agent.update(
            b_obs, b_actions, b_logprobs, b_returns, b_advantages, 
            update_epochs=update_epochs, minibatch_size=minibatch_size
        )
        
        # Save the model at each update
        model_path = models_dir / f"agent_update_{update}.pt"
        torch.save(agent.state_dict(), model_path)
        logger.info(f"Saved model to {model_path}")
        
        # Create and log visualization if it's time
        if use_wandb and (update == 1 or update % visualization_interval == 0):
            logger.info(f"Creating visualization for update {update}...")
            try:
                # Save original agent's device
                original_device = next(agent.parameters()).device
                logger.info(f"Agent was on device {original_device} before visualization")
                
                # Always use CPU for visualization
                viz_device = torch.device("cpu")
                logger.info(f"Starting visualization process using device: {viz_device}")
                
                # Generate visualization
                frames, viz_reward, viz_captures = create_agent_visualization(
                    agent=agent,
                    env=None,  # We create a new environment in the function
                    device=viz_device,
                    deterministic=True,
                    max_steps=env_max_steps
                )
                
                # Explicitly move agent back to original device
                agent = agent.to(original_device)
                logger.info(f"Agent moved back to device {next(agent.parameters()).device} after visualization")
                
                # Verify agent is on the correct device
                if next(agent.parameters()).device != original_device:
                    logger.warning(f"Agent device mismatch after visualization! On {next(agent.parameters()).device}, should be on {original_device}")
                    # Force move to correct device
                    agent = agent.to(original_device)
                
                # Save visualization to file
                gif_path = vis_dir / f"agent_update_{update}.gif"
                create_animated_gif(frames, output_path=gif_path)
                logger.info(f"Saved visualization to {gif_path}")
                
                # Log visualization to wandb
                wandb.log({
                    "agent_behavior": wandb.Video(str(gif_path), fps=20, format="gif"),
                    "viz_reward": viz_reward,
                    "viz_waypoints_captured": viz_captures,
                    "update": update,
                    "global_step": global_step
                })
                
                logger.info(f"Logged visualization to wandb (reward: {viz_reward:.2f}, captures: {viz_captures})")
            except Exception as e:
                logger.error(f"Failed to create visualization: {e}")
                import traceback
                logger.error(f"Error stack trace: {traceback.format_exc()}")
                
                # Ensure agent is on the correct device even if visualization fails
                agent = agent.to(device)
                logger.info(f"Agent moved to device {next(agent.parameters()).device} after visualization error")
        
        # Log training metrics to wandb
        if use_wandb:
            # Calculate statistics of advantages and returns
            advantages_mean = advantages.mean().item()
            advantages_std = advantages.std().item()
            returns_mean = returns.mean().item()
            returns_std = returns.std().item()
            
            # Calculate mean reward over recent episodes
            mean_reward = np.mean(episode_rewards[-100:]) if episode_rewards else 0
            mean_length = np.mean(episode_lengths[-100:]) if episode_lengths else 0
            
            # Calculate the ratio of timeout episodes to total episodes
            total_episodes = timeout_episodes + capture_episodes
            timeout_ratio = timeout_episodes / max(1, total_episodes)
            capture_ratio = capture_episodes / max(1, total_episodes)
            
            # Log all metrics
            wandb.log({
                "update": update,
                "global_step": global_step,
                "policy_loss": metrics["policy_loss"],
                "value_loss": metrics["value_loss"],
                "entropy": metrics["entropy"],
                "approx_kl": metrics.get("approx_kl", 0),
                "clip_fraction": metrics["clipfrac"],
                "learning_rate": lrnow,
                "advantages_mean": advantages_mean,
                "advantages_std": advantages_std,
                "returns_mean": returns_mean, 
                "returns_std": returns_std,
                "mean_reward_100": mean_reward,
                "mean_episode_length_100": mean_length,
                "episodes_completed": env_done_flags.sum().item(),
                "timeout_ratio": timeout_ratio,
                "capture_ratio": capture_ratio,  # Add the capture ratio to wandb logs
                "total_timeout_episodes": timeout_episodes,
                "total_capture_episodes": capture_episodes,
                "fps": int(global_step / (time.time() - start_time))
            }, step=global_step)
        
        # Print training progress
        if update % 1 == 0:
            # Calculate elapsed time
            elapsed_time = time.time() - start_time
            
            # Calculate metrics
            mean_reward = np.mean(episode_rewards[-100:]) if episode_rewards else 0
            mean_length = np.mean(episode_lengths[-100:]) if episode_lengths else 0
            fps = int(global_step / elapsed_time)
            
            # Calculate capture ratio for logging
            total_episodes = timeout_episodes + capture_episodes
            capture_ratio = capture_episodes / max(1, total_episodes)
            
            # Log progress
            logger.info(f"Update: {update}/{num_updates} | Steps: {global_step}")
            logger.info(f"Mean reward: {mean_reward:.2f} | Mean length: {mean_length:.2f}")
            logger.info(f"Timeout episodes: {timeout_episodes} | Capture episodes: {capture_episodes}")
            logger.info(f"Capture ratio: {capture_ratio:.3f} (higher is better)")
            logger.info(f"Policy loss: {metrics['policy_loss']:.4f} | Value loss: {metrics['value_loss']:.4f}")
            logger.info(f"Entropy: {metrics['entropy']:.4f} | Clip fraction: {metrics['clipfrac']:.4f}")
            logger.info(f"FPS: {fps} | Learning rate: {lrnow:.6f}")
            logger.info(f"Episodes completed this update: {env_done_flags.sum().item()}/{num_envs}")
            logger.info("---------------------------------------")
    
    # Create a final visualization
    if use_wandb:
        logger.info("Creating final visualization...")
        try:
            # Save original agent's device
            original_device = next(agent.parameters()).device
            logger.info(f"Agent was on device {original_device} before final visualization")
            
            # Always use CPU for visualization
            viz_device = torch.device("cpu")
            logger.info(f"Starting final visualization process using device: {viz_device}")
            
            # Generate visualization using the same approach as during training
            frames, viz_reward, viz_captures = create_agent_visualization(
                agent=agent,  # Will be copied inside the visualization function
                env=None,  # We create a new environment in the function
                device=viz_device,
                deterministic=True,
                max_steps=env_max_steps
            )
            
            # Explicitly move agent back to original device
            agent = agent.to(original_device)
            logger.info(f"Agent moved back to device {next(agent.parameters()).device} after final visualization")
            
            # Verify agent is on the correct device
            if next(agent.parameters()).device != original_device:
                logger.warning(f"Agent device mismatch after final visualization! On {next(agent.parameters()).device}, should be on {original_device}")
                # Force move to correct device
                agent = agent.to(original_device)
            
            # Save visualization to file
            gif_path = vis_dir / "agent_final.gif"
            create_animated_gif(frames, output_path=gif_path)
            logger.info(f"Saved final visualization to {gif_path}")
            
            # Log visualization to wandb
            wandb.log({
                "final_agent_behavior": wandb.Video(str(gif_path), fps=20, format="gif"),
                "final_viz_reward": viz_reward,
                "final_viz_waypoints_captured": viz_captures,
                "global_step": global_step
            })
            
            logger.info(f"Logged final visualization to wandb (reward: {viz_reward:.2f}, captures: {viz_captures})")
        except Exception as e:
            logger.error(f"Failed to create final visualization: {e}")
            import traceback
            logger.error(f"Error stack trace: {traceback.format_exc()}")
            
            # Ensure agent is on the correct device even if visualization fails
            agent = agent.to(device)
            logger.info(f"Agent moved to device {next(agent.parameters()).device} after visualization error")
    
    # Save the final trained model
    final_model_path = exp_dir / "ppo_bike_agent_final.pt"
    torch.save(agent.state_dict(), final_model_path)
    logger.info(f"Training complete! Final model saved to {final_model_path}")
    
    # Log final statistics
    total_episodes = timeout_episodes + capture_episodes
    capture_ratio = capture_episodes / max(1, total_episodes)
    logger.info(f"Total episodes completed: {len(episode_rewards)}")
    logger.info(f"Timeout episodes: {timeout_episodes} ({100*timeout_episodes/max(1,len(episode_rewards)):.1f}%)")
    logger.info(f"Capture episodes: {capture_episodes} ({100*capture_episodes/max(1,len(episode_rewards)):.1f}%)")
    logger.info(f"Final capture ratio: {capture_ratio:.3f}")
    
    # Close wandb if enabled
    if use_wandb:
        wandb.finish()
    
    return agent, episode_rewards, exp_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO agent for race bike environment")
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    parser.add_argument("--num-envs", type=int, default=8, help="Number of parallel environments")
    parser.add_argument("--total-timesteps", type=int, default=1_000_000, help="Total number of timesteps")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--num-steps", type=int, default=2048, help="Number of steps per rollout")
    parser.add_argument("--update-epochs", type=int, default=10, help="Number of PPO update epochs")
    parser.add_argument("--hidden-dim", type=int, default=64, help="Hidden layer dimension")
    parser.add_argument("--minibatch-size", type=int, default=128, help="Size of minibatches for updates")
    parser.add_argument("--entropy-coef", type=float, default=0.05, help="Entropy coefficient")
    parser.add_argument("--clip-ratio", type=float, default=0.2, help="PPO clip ratio for policy updates")
    parser.add_argument("--max-grad-norm", type=float, default=0.5, help="Maximum gradient norm for clipping")
    parser.add_argument("--use-wandb", action="store_true", help="Whether to use wandb for logging")
    parser.add_argument("--wandb-project", type=str, default="ppo-bike-training", help="Wandb project name")
    parser.add_argument("--wandb-entity", type=str, default=None, help="Wandb entity name")
    parser.add_argument("--exp-name", type=str, default="PPO-BikeRacer", help="Experiment name")
    parser.add_argument("--gpu", type=int, default=None, help="Specific GPU device ID to use (e.g., 0 or 1)")
    parser.add_argument("--max-steps", type=int, default=200, help="Maximum steps per episode")
    
    args = parser.parse_args()
    
    # Use CUDA if available, with specific GPU if specified
    if torch.cuda.is_available():
        if args.gpu is not None and args.gpu >= 0 and args.gpu < torch.cuda.device_count():
            device = torch.device(f"cuda:{args.gpu}")
            torch.cuda.set_device(args.gpu)
            print(f"Using GPU: {args.gpu} - {torch.cuda.get_device_name(args.gpu)}")
        else:
            device = torch.device("cuda")
            print(f"Using default CUDA device: {torch.cuda.get_device_name()}")
    else:
        device = torch.device("cpu")
        print("CUDA not available, using CPU")
    
    # Set up experiment directory and logging
    exp_dir, logger = setup_experiment_dir()
    
    # Train agent
    agent, rewards, exp_dir = train_ppo(
        seed=args.seed,
        num_envs=args.num_envs,
        learning_rate=args.learning_rate,
        total_timesteps=args.total_timesteps,
        num_steps=args.num_steps,
        update_epochs=args.update_epochs,
        hidden_dim=args.hidden_dim,
        minibatch_size=args.minibatch_size,
        entropy_coef=args.entropy_coef,
        clip_ratio=args.clip_ratio,
        max_grad_norm=args.max_grad_norm,
        device=device,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        exp_name=args.exp_name,
        exp_dir=exp_dir,
        logger=logger,
        max_steps=args.max_steps
    )
