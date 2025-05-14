import gym
from gym import spaces
import numpy as np
import torch
import math
import logging

class VectorizedRaceBikeDojo(gym.Env):
    """
    Vectorized environment for the race bike task
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, num_envs=1, device='cpu', logger=None, max_steps=200):
        super(VectorizedRaceBikeDojo, self).__init__()
        self.num_envs = num_envs
        self.device = torch.device(device) if isinstance(device, str) else device
        # State is a tensor of shape (num_envs, 4)
        # The state is [x, y, theta, speed]
        self.state = None  # NOTE: gets initialized in reset()
        
        # Set up logger (use default if none provided)
        self.logger = logger or logging.getLogger("torch_env")
        if not self.logger.handlers:
            self.logger.setLevel(logging.INFO)
            handler = logging.StreamHandler()
            handler.setLevel(logging.DEBUG)  # Ensure handler level is at least DEBUG
            self.logger.addHandler(handler)
        
        # Simulation parameters
        self.min_speed = 0.0 
        self.max_speed = 0.05  # m/s - Changed from 0.1 to 0.05 as per requirements
        self.max_speed_cmd = 0.01  # Changed from 0.02 to 0.01 as per requirements
        # Max steering change is 10 degrees per step (converted to radians)
        self.max_steering_change = 10 * (math.pi / 180)  # Changed from math.pi to 10 degrees in radians
        self.dt = 0.1  # time step
        self.capture_threshold = 0.1  # Reduced from 0.1 to make capture harder
        self.max_steps = max_steps
        self.min_steps_before_capture = 0  # Minimum steps required before capture is allowed
        
        # Reward parameters
        self.lambda_penalty = 0.2  # Increased penalty for moving away from waypoint (from 0.1 to 0.2)
        self.speed_reward_coef = 0.05  # Reduced from 0.1 to 0.05 to make non-capture rewards smaller
        self.min_speed_for_reward = 0.01  # Minimum speed to receive positive speed reward
        
        # Reward scaling factors
        self.capture_reward_factor = 200.0  # Drastically increased from 10.0 to 30.0 to make capture extremely attractive00
        self.time_penalty = -0.05  # Increased time penalty from -0.01 to -05 to discourage lingering
        
        # Progress reward scaling factors - reduced to make intermediate progress less rewarding
        self.progress_reward_factor = 1.0  # Scale for delta distance reward
        self.directional_reward_factor = 0.1  # Scale for heading alignment reward
        
        # Observation: relative x, relative y, relative heading, speed
        self.observation_space = spaces.Box(
            low=np.array([-float("inf"), -float("inf"), -math.pi, self.min_speed]),
            high=np.array([float("inf"), float("inf"), math.pi, self.max_speed]),
            dtype=np.float32
        )

        # Action: [delta_heading, target_speed]
        self.action_space = spaces.Box(
            low=np.array([-self.max_steering_change, self.min_speed]),
            high=np.array([self.max_steering_change, self.max_speed]),
            dtype=np.float32
        )

        # Store observation space bounds as tensors on the right device for faster checking
        self.obs_low = torch.tensor(self.observation_space.low, dtype=torch.float32, device=self.device)
        self.obs_high = torch.tensor(self.observation_space.high, dtype=torch.float32, device=self.device)

        # Initialize tensors
        self.reset()

    def reset(self):
        """
        Reset all environments with random initialization as specified:
        - All bikes start at (0,0)
        - Random heading between 0 and 2π
        - Speed always starts at 0 (changed from random)
        - Random waypoints in a unit circle
        """
        # Initialize bikes at (0,0) with random heading and zero speed
        x = torch.zeros(self.num_envs, device=self.device)
        y = torch.zeros(self.num_envs, device=self.device)
        theta = torch.rand(self.num_envs, device=self.device) * 2 * math.pi  # Random heading [0, 2π]
        speed = torch.zeros(self.num_envs, device=self.device)  # Always start with zero speed
        
        # Initialize state: [x, y, theta, speed]
        self.state = torch.stack([x, y, theta, speed], dim=1)
        
        # Initialize waypoints in unit circle using spherical coordinates
        theta_w = torch.rand(self.num_envs, device=self.device) * 2 * math.pi  # Random angle [0, 2π]
        # Ensure waypoints never spawn within capture radius (0.05)
        radius_w = 0.1 + 0.9 * torch.rand(self.num_envs, device=self.device)  # Random radius [0.1, 1.0]
        
        # Convert to Cartesian coordinates
        waypoint_x = radius_w * torch.cos(theta_w)
        waypoint_y = radius_w * torch.sin(theta_w)
        self.waypoints = torch.stack([waypoint_x, waypoint_y], dim=1)
        
        # Initialize step counter and previous distances
        self.steps = torch.zeros(self.num_envs, dtype=torch.int, device=self.device)
        self.prev_dist_to_waypoint = torch.sqrt(torch.sum(self.waypoints**2, dim=1))  # Initial distance to waypoint
        
        # Store initial distance for calculating capture reward
        self.initial_dist_to_waypoint = self.prev_dist_to_waypoint.clone()
        
        # LOG DEBUG INFO
        self.logger.info(f"States:")
        for i in range(min(5, self.num_envs)):  # Show only first 5 environments to avoid cluttering
            state_np = self.state[i].detach().cpu().numpy().round(2)
            self.logger.info(f"  {state_np[0]:.2f}, {state_np[1]:.2f}, {state_np[2]:.2f}, {state_np[3]:.2f}")
        if self.num_envs > 5:
            self.logger.info(f"  ...")
            
        self.logger.info(f"Waypoints:")
        for i in range(min(5, self.num_envs)):
            wp_np = self.waypoints[i].detach().cpu().numpy().round(2)
            self.logger.info(f"  {wp_np[0]:.2f}, {wp_np[1]:.2f}")
        if self.num_envs > 5:
            self.logger.info(f"  ...")
            
        self.logger.info(f"Initial dist to waypoint:")
        for i in range(min(5, self.num_envs)):
            dist = self.prev_dist_to_waypoint[i].detach().cpu().item()
            self.logger.info(f"  {dist:.2f}")
        if self.num_envs > 5:
            self.logger.info(f"  ...")
        
        # Get observations
        return self._get_obs()

    def _get_obs(self):
        """
        Returns the observation vector:
        - relative x position to waypoint
        - relative y position to waypoint
        - relative heading to waypoint
        - speed
        """
        # Extract bike position and waypoint position
        bike_pos = self.state[:, :2]  # x, y
        bike_heading = self.state[:, 2]  # theta
        speed = self.state[:, 3]  # speed
        
        # Calculate relative position (waypoint - bike position)
        rel_pos = self.waypoints - bike_pos
        
        # Calculate relative angle between bike heading and waypoint
        waypoint_angle = torch.atan2(rel_pos[:, 1], rel_pos[:, 0])
        rel_heading = waypoint_angle - bike_heading
        
        # Normalize the angle to [-π, π]
        rel_heading = torch.atan2(torch.sin(rel_heading), torch.cos(rel_heading))
        
        # Combine into observation
        obs = torch.cat([rel_pos, rel_heading.unsqueeze(1), speed.unsqueeze(1)], dim=1)
        
        # Debug check for observation bounds
        if torch.any(obs < self.obs_low) or torch.any(obs > self.obs_high):
            self.logger.warning("--------------------------------")
            self.logger.warning("Warning: Observation outside declared bounds!")
            self.logger.warning("--------------------------------")
        
        return obs

    def reset_if_done(self, done_mask):
        """
        Reset only the environments that are done
        
        Parameters:
            done_mask: boolean tensor indicating which environments to reset
        
        Returns:
            new_obs: observations for the reset environments
        """
        if not torch.any(done_mask):
            # No environments to reset
            return None
            
        # Count how many environments we're resetting
        num_resets = done_mask.sum().item()
        self.logger.debug(f"Resetting {num_resets} environments")
        
        # Initialize new states for the done environments
        x_new = torch.zeros(num_resets, device=self.device)
        y_new = torch.zeros(num_resets, device=self.device)
        theta_new = torch.rand(num_resets, device=self.device) * 2 * math.pi  # Random heading [0, 2π]
        speed_new = torch.zeros(num_resets, device=self.device)  # Always start with zero speed
        
        # Initialize new waypoints in unit circle using spherical coordinates
        theta_w_new = torch.rand(num_resets, device=self.device) * 2 * math.pi  # Random angle [0, 2π]
        radius_w_new = 0.1 + 0.9 * torch.rand(num_resets, device=self.device)  # Random radius [0.1, 1.0]
        
        # Convert to Cartesian coordinates
        waypoint_x_new = radius_w_new * torch.cos(theta_w_new)
        waypoint_y_new = radius_w_new * torch.sin(theta_w_new)
        
        # Get indices of done environments
        done_indices = torch.where(done_mask)[0]
        
        # Update state for done environments
        new_state = torch.stack([x_new, y_new, theta_new, speed_new], dim=1)
        self.state[done_indices] = new_state
        
        # Update waypoints for done environments
        new_waypoints = torch.stack([waypoint_x_new, waypoint_y_new], dim=1)
        self.waypoints[done_indices] = new_waypoints
        
        # Reset step counter for done environments
        self.steps[done_indices] = 0
        
        # Calculate and update previous distance for the reset environments
        dist_to_waypoint_new = torch.sqrt(torch.sum(new_waypoints**2, dim=1))
        self.prev_dist_to_waypoint[done_indices] = dist_to_waypoint_new
        
        # Store initial distance for capture reward calculation
        self.initial_dist_to_waypoint[done_indices] = dist_to_waypoint_new.clone()
        
        # Return new observations for the reset environments
        return self._get_obs()

    def step(self, action):
        """
        Vectorized step function for multiple environments
        
        Parameters:
            action: torch.Tensor of shape (num_envs, 2) with [delta_heading, target_speed]
            
        Returns:
            obs: observations
            rewards: rewards for each environment
            dones: done flags for each environment
            info: additional information
        """
        # Convert actions to torch tensor if they aren't already
        if not isinstance(action, torch.Tensor):
            action = torch.tensor(action, device=self.device)
        elif action.device != self.device:
            action = action.to(self.device)
        
        # Debug log actions for first environment
        if self.steps[0] % 10 == 0 or self.steps[0] > 180:  # Every 10 steps or near the end
            self.logger.debug(f"Step:  {self.steps[0].item()}")
            obs = self._get_obs()
            self.logger.debug("Obs: ")
            self.logger.debug(f"         Dx:  {obs[0, 0].item()}")
            self.logger.debug(f"         Dy:  {obs[0, 1].item()}")
            self.logger.debug(f"         Heading:  {obs[0, 2].item()}")
            self.logger.debug(f"         Speed:  {obs[0, 3].item()}")
            self.logger.debug("Action: ")
            self.logger.debug(f"         δθ:  {action[0, 0].item()}")
            self.logger.debug(f"         Speed target:  {action[0, 1].item()}")
        
        # Extract current state components
        x, y, heading, speed = self.state[:, 0], self.state[:, 1], self.state[:, 2], self.state[:, 3]
        
        # Apply heading action (clipped to action space)
        delta_heading = torch.clamp(action[:, 0], -self.max_steering_change, self.max_steering_change)
        
        # Apply speed action - directly set the target speed instead of applying delta
        # The agent directly outputs the desired speed value
        target_speed = torch.clamp(action[:, 1], self.min_speed, self.max_speed)
        
        # Update heading and speed
        heading = (heading + delta_heading) % (2 * math.pi)
        speed = target_speed  # Directly set the speed to the target value
        
        # Update position
        x = x + speed * torch.cos(heading) * self.dt
        y = y + speed * torch.sin(heading) * self.dt
        
        # Update state
        self.state = torch.stack([x, y, heading, speed], dim=1)
        
        # Calculate current distance to waypoint
        dist_to_waypoint = torch.sqrt(torch.sum((self.waypoints - self.state[:, :2])**2, dim=1))
        
        # Calculate rewards
        rewards = torch.zeros(self.num_envs, device=self.device)
        
        # Apply time penalty to discourage needlessly long episodes
        # Small penalty applied each step to encourage efficient paths
        rewards = rewards + self.time_penalty
        
        # Calculate capture reward = capture_reward_factor * initial_distance
        # This makes the capture reward much larger than cumulative rewards for navigation
        # The further the waypoint, the bigger the reward for capturing it
        capture_rewards = self.capture_reward_factor * self.initial_dist_to_waypoint
        
        # Add bonus for capturing waypoints quickly
        # The sooner you capture, the bigger the bonus (based on remaining steps)
        steps_remaining = torch.maximum(torch.zeros_like(self.steps, device=self.device), 
                                        self.max_steps - self.steps)
        quick_capture_bonus = 10.0 * steps_remaining / self.max_steps  # Increased from 2.0 to 10.0 to prioritize fast captures
        
        # Reward for capturing waypoint - use the scaled reward based on initial distance
        # Plus the bonus for capturing quickly
        capture_mask = (dist_to_waypoint < self.capture_threshold)
        full_capture_reward = capture_rewards + quick_capture_bonus
        rewards = torch.where(capture_mask, full_capture_reward, rewards)
        
        # Calculate directional reward: reward for facing the waypoint
        # Get angle to waypoint
        rel_pos = self.waypoints - self.state[:, :2]
        waypoint_angle = torch.atan2(rel_pos[:, 1], rel_pos[:, 0])
        
        # Calculate how aligned the bike heading is with the direction to the waypoint
        # 1.0 means perfectly aligned, -1.0 means facing opposite direction
        heading_alignment = torch.cos(waypoint_angle - heading)
        
        # Calculate delta distance reward with a higher coefficient (changed from 1.0 to 2.0)
        delta_dist = self.prev_dist_to_waypoint - dist_to_waypoint
        delta_dist_reward = torch.where(
            delta_dist < 0,  # If moving away from the waypoint
            2.0 * self.progress_reward_factor * delta_dist,  # Higher negative reward for moving away
            3.0 * self.progress_reward_factor * delta_dist  # Higher positive reward for getting closer, but scaled
        )
        
        # Add directional reward (only for non-capture steps)
        # Scale based on speed - faster speeds get more directional reward
        directional_reward = self.directional_reward_factor * heading_alignment * speed / self.max_speed
        
        # Apply distance and directional rewards (only for non-capture steps)
        rewards = torch.where(~capture_mask, rewards + delta_dist_reward + directional_reward, rewards)
        
        # Reward higher speeds at all times - no slowing down for waypoints
        # This encourages the agent to go full speed directly into the waypoint
        speed_reward = self.speed_reward_coef * (speed / self.max_speed)  # Linear reward based on speed
        
        # Apply speed reward (for all steps)
        rewards = rewards + speed_reward
        
        # Update previous distance
        self.prev_dist_to_waypoint = dist_to_waypoint
        
        # Update step counter
        self.steps += 1
        
        # Check if episodes are done
        dones = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        
        # Only allow capture after minimum steps to prevent early termination
        valid_capture_mask = torch.logical_and(
            capture_mask,
            self.steps >= self.min_steps_before_capture
        )
        
        # Done if waypoint captured or max steps reached
        dones = torch.logical_or(dones, valid_capture_mask)
        dones = torch.logical_or(dones, self.steps >= self.max_steps)
        
        # Get observations (before resetting done environments)
        obs = self._get_obs()
        
        # Store dones for GAE calculation before resetting
        dones_for_return = dones.clone()
        
        # Reset done environments
        # Note: we don't need to update the observations since the reset
        # environments will only be used in the next step
        if torch.any(dones):
            num_resets = dones.sum().item()
            self.logger.debug(f"Resetting {num_resets}/{self.num_envs} environments")
        self.reset_if_done(dones)
        
        # Additional info
        info = {}
        
        return obs, rewards, dones_for_return, info, self.state

    def render(self, mode='human'):
        """Simple rendering for debugging purposes"""
        if self.num_envs == 1:
            self.logger.info(f"State: {self.state[0]}, Waypoint: {self.waypoints[0]}")
        else:
            self.logger.info(f"Showing first environment of {self.num_envs}:")
            self.logger.info(f"State: {self.state[0]}, Waypoint: {self.waypoints[0]}")

# Sample usage
if __name__ == "__main__":
    env = VectorizedRaceBikeDojo(num_envs=4, device='cpu')
    obs = env.reset()
    print("Initial observation:", obs)
    
    # Random action
    action = torch.rand(4, 2) * 2 - 1  # Random actions between -1 and 1
    obs, rewards, dones, info, state = env.step(action)
    
    print("After step:")
    print("Observations:", obs)
    print("Rewards:", rewards)
    print("Dones:", dones) 