import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import math


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """
    Initialize the weights of a layer using orthogonal initialization
    """
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class PPOAgent(nn.Module):
    """
    PPO agent class
    """
    def __init__(
        self,
        obs_dim=4,
        hidden_dim=64,
        action_dim=2,
        learning_rate=3e-4,
        clip_ratio=0.2,
        entropy_coef=0.01,
        max_grad_norm=0.5,
        device='cpu'
    ):
        super(PPOAgent, self).__init__()
        self.device = device
        self.clip_ratio = clip_ratio
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        
        # Actor network (policy)
        # The actor network takes the observation and outputs the mean of the action distribution
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, action_dim), std=0.01)
        )
        
        # Log standard deviation (learnable)
        # The log standard deviation is a learnable parameter that is used to scale the action distribution
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))
        
        # Critic network (value function)
        # The critic network takes the observation and outputs the value of the state
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, 1), std=1.0)
        )
        
        # Action scaling to match environment's action space
        # - max_steering_change: 10 degrees in radians
        # - max_speed: 0.05 (direct speed control rather than delta)
        self.action_scale = torch.tensor([10 * (math.pi / 180), 0.05]).to(device)
        
        # Optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        
    def get_value(self, obs):
        """Compute value function for given observations"""
        return self.critic(obs)
        
    def get_action_and_value(self, obs, raw_action=None):
        """
        Sample a *raw* action from policy or evaluate log_prob of a given raw_action,
        then squash+scale it for the env.

        Parameters:
            obs:      Tensor[batch, obs_dim]
            raw_action:  Optional Tensor[batch, action_dim] (pre-squash Gaussian sample)

        Returns:
            scaled_action  Tensor[batch, action_dim]  — what you send to env.step()
            log_prob       Tensor[batch]             — log π(raw_action|obs) with Jacobians
            entropy        Tensor[batch]             — entropy of the Gaussian
            value          Tensor[batch,1]           — critic value
            raw_action     Tensor[batch, action_dim] — the pre-squash sample
        """
        action_mean   = self.actor_mean(obs)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std    = torch.clamp(action_logstd.exp(), min=1e-6, max=1.0)

        dist = Normal(action_mean, action_std)

        # 1) SAMPLE or reuse the *raw* normal sample
        if raw_action is None:
            raw_action = dist.sample()

        # 2) TRANSFORM
        heading_raw = raw_action[:, 0:1]
        speed_raw   = raw_action[:, 1:2]

        scaled_heading = torch.tanh(heading_raw) * self.action_scale[0]
        scaled_speed   = torch.sigmoid(speed_raw) * self.action_scale[1]
        scaled_action  = torch.cat([scaled_heading, scaled_speed], dim=1)

        # 3) LOG-PROB of the *raw* action + Jacobian corrections
        log_prob = dist.log_prob(raw_action).sum(dim=-1)

        eps = 1e-6
        # tanh Jacobian:  log(1 − tanh²)
        log_prob -= torch.log(1 - torch.tanh(heading_raw).pow(2) + eps).sum(dim=-1)
        # sigmoid Jacobian: log(σ(x)(1−σ(x)))
        log_prob -= torch.log(
            torch.sigmoid(speed_raw) * (1 - torch.sigmoid(speed_raw)) + eps
        ).sum(dim=-1)

        entropy = dist.entropy().sum(dim=-1)
        value   = self.get_value(obs).flatten()

        return scaled_action, log_prob, entropy, value, raw_action

    
    def update(self, obs, actions, old_log_probs, returns, advantages, update_epochs=10, minibatch_size=64):
        """
        Update agent using PPO algorithm
        
        Parameters:
            obs: list of observations
            actions: list of actions
            old_log_probs: list of old log probabilities
            returns: list of returns
            advantages: list of advantages
            update_epochs: number of PPO update epochs
        
        Returns:
            Dictionary containing the following metrics:
            - policy_loss: mean policy loss
            - value_loss: mean value loss
            - entropy: mean entropy
            - clipfrac: mean clip fraction
        """
        # Ensure inputs are tensors on the correct device
        def _to_tensor(x):
            if isinstance(x, torch.Tensor):
                return x.to(self.device)
            else:
                return torch.FloatTensor(x).to(self.device)
        
        b_obs = _to_tensor(obs)
        b_actions = _to_tensor(actions)
        b_old_log_probs = _to_tensor(old_log_probs)
        b_returns = _to_tensor(returns)
        b_advantages = _to_tensor(advantages)
        
        # Normalize advantages (reduces variance)
        epsilon = 1e-8  # Avoid division by zero
        if torch.isnan(b_advantages).any() or torch.isinf(b_advantages).any():
            print("WARNING: NaN or Inf values in advantages - fixing...")
            b_advantages = torch.nan_to_num(b_advantages, nan=0.0, posinf=10.0, neginf=-10.0)
            
        # Only normalize if advantages are not all zeros
        if b_advantages.std() > epsilon:
            b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + epsilon)
        
        # Calculate batch size and number of mini-batches
        batch_size = b_obs.size(0)
        minibatch_size = min(minibatch_size, batch_size)
        
        clipfracs = []  # Track percentage of clipped policy updates
        
        # PPO update epochs
        for _ in range(update_epochs):
            # Generate random indices for mini-batches
            indices = torch.randperm(batch_size)
            
            # Iterate over mini-batches
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                if end > batch_size:
                    break
                    
                minibatch_indices = indices[start:end]
                
                # Recompute log-prob, entropy, and value from the same raw_action we stored
                _, newlog_prob, new_entropy, newvalue, _ = self.get_action_and_value(
                    b_obs[minibatch_indices],
                    raw_action=b_actions[minibatch_indices]
                )

                # Importance ratio
                ratio = torch.exp(newlog_prob - b_old_log_probs[minibatch_indices])

                # Track clip fraction (once per minibatch)
                with torch.no_grad():
                    clipfrac = ((ratio - 1.0).abs() > self.clip_ratio).float().mean().item()
                    clipfracs.append(clipfrac)

                # Calculate policy loss (clipped surrogate objective)
                mini_batch_advantages = b_advantages[minibatch_indices]
                pg_loss1 = -mini_batch_advantages * ratio
                pg_loss2 = -mini_batch_advantages * torch.clamp(
                    ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Calculate value loss (clipped)
                mini_batch_returns = b_returns[minibatch_indices]
                v_loss_unclipped = (newvalue - mini_batch_returns) ** 2
                value_pred_clipped = mini_batch_returns + torch.clamp(
                    newvalue - mini_batch_returns, -self.clip_ratio, self.clip_ratio
                )
                v_loss_clipped = (value_pred_clipped - mini_batch_returns) ** 2
                v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()

                # Entropy bonus (use new_entropy, not undefined `entropy`)
                entropy_loss = new_entropy.mean()

                # Total loss
                loss = pg_loss + 0.5 * v_loss - self.entropy_coef * entropy_loss

                # Update the policy
                self.optimizer.zero_grad()
                loss.backward()
                
                # Enhanced gradient clipping
                for param in self.parameters():
                    if param.grad is not None:
                        # Check for NaN in gradients
                        if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                            param.grad = torch.nan_to_num(param.grad, nan=0.0, posinf=1.0, neginf=-1.0)
                
                # Stronger gradient clipping using the configurable parameter
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=self.max_grad_norm)
                
                self.optimizer.step()
                
                # Verify no NaN values in the parameters after update
                for name, param in self.named_parameters():
                    if torch.isnan(param).any():
                        print(f"WARNING: NaN values detected in {name} after update")
                        param.data = torch.nan_to_num(param.data, nan=0.0)
        
        return {
            "policy_loss": pg_loss.item(),
            "value_loss": v_loss.item(),
            "entropy": entropy_loss.item(),
            "clipfrac": np.mean(clipfracs)
        }


def compute_gae(rewards, values, dones, next_value, gamma=0.99, gae_lambda=0.95):
    """
    Compute Generalized Advantage Estimation (GAE)
    
    Parameters:
        rewards: list of rewards
        values: list of values
        dones: list of done flags
        next_value: next value
        gamma: discount factor
        gae_lambda: lambda for GAE
    
    Returns:
        advantages: list of advantages
        returns: list of returns
    """
    advantages = np.zeros_like(rewards)
    last_gae_lam = 0
    
    # Calculate advantages in reverse order
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_non_terminal = 1.0 - dones[-1]
            next_val = next_value
        else:
            next_non_terminal = 1.0 - dones[t + 1]
            next_val = values[t + 1]
            
        delta = rewards[t] + gamma * next_val * next_non_terminal - values[t]
        last_gae_lam = delta + gamma * gae_lambda * next_non_terminal * last_gae_lam
        advantages[t] = last_gae_lam
        
    returns = advantages + values
    
    return advantages, returns 