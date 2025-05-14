#!/usr/bin/env python3
"""
Train a new PPO agent with the fixed environment and parameters
"""
import sys
import os
import subprocess
import argparse

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from rl_control.train.train_ppo_bike import train_ppo, setup_experiment_dir
import torch
import wandb


def parse_args():
    parser = argparse.ArgumentParser(description="Train a new PPO agent for the bike racing task")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID to use for training (default: 0)")
    parser.add_argument("--timesteps", type=int, default=1_000_000, help="Total timesteps for training (default: 1M)")
    parser.add_argument("--num-envs", type=int, default=16, help="Number of parallel environments (default: 16)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--wandb-project", type=str, default="ppo-bike-training-corrected",
                        help="Wandb project name (default: 'ppo-bike-training-corrected')")
    parser.add_argument("--max-steps", type=int, default=200, help="Maximum steps per episode (default: 200)")
    return parser.parse_args()


if __name__ == "__main__":
    # Parse command-line arguments
    args = parse_args()

    # Check if wandb is logged in
    try:
        # Check if wandb is logged in
        result = subprocess.run(['wandb', 'status'], capture_output=True, text=True)

        # Check if the status indicates user is logged in
        if "not logged in" in result.stdout:
            print("You need to log in to Weights & Biases first.")
            print("Run 'wandb login' in your terminal and follow the instructions.")
            sys.exit(1)
        else:
            print("Wandb is logged in and ready to sync data.")
    except Exception as e:
        print(f"Error checking wandb status: {e}")
        print("Please make sure wandb is installed and run 'wandb login' before running this script.")
        print("Continuing without wandb verification...")

    # Set up experiment directory
    exp_dir, logger = setup_experiment_dir()

    # Check for CUDA availability and set device
    if torch.cuda.is_available():
        # Check if the specified GPU ID is valid
        if args.gpu >= 0 and args.gpu < torch.cuda.device_count():
            device = torch.device(f"cuda:{args.gpu}")
            torch.cuda.set_device(args.gpu)
            logger.info(f"Training on GPU {args.gpu}: {torch.cuda.get_device_name(args.gpu)}")

            # Free up memory if possible
            torch.cuda.empty_cache()

            # Log GPU memory info
            allocated = torch.cuda.memory_allocated(device) / (1024 ** 3)
            max_allocated = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
            logger.info(f"GPU memory allocated: {allocated:.2f} GB")
            logger.info(f"GPU max memory allocated: {max_allocated:.2f} GB")
        else:
            available_gpus = torch.cuda.device_count()
            logger.warning(f"Specified GPU ID {args.gpu} is not valid. Available GPUs: 0-{available_gpus - 1}")
            logger.warning(f"Defaulting to GPU 0: {torch.cuda.get_device_name(0)}")
            device = torch.device("cuda:0")
            torch.cuda.set_device(0)
    else:
        device = torch.device("cpu")
        logger.info("CUDA not available, training on CPU")

    # Log max steps
    logger.info(f"Using max_steps={args.max_steps} per episode")

    # Log PyTorch version
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA version: {torch.version.cuda}")

    # Train agent with improved parameters
    try:
        train_ppo(
            seed=args.seed,
            num_envs=args.num_envs,
            learning_rate=5e-4,  # Increased from 3e-4 to 5e-4 for faster learning
            total_timesteps=args.timesteps,
            num_steps=2048,
            gamma=0.99,
            gae_lambda=0.95,
            clip_ratio=0.2,
            update_epochs=10,
            minibatch_size=256,  # Larger minibatch for better gradient estimates
            hidden_dim=256,  # Increased from 128 to 256 for more model capacity
            entropy_coef=0.08,  # Increased from 0.05 to 0.08 for more exploration
            max_grad_norm=0.5,
            device=device,
            use_wandb=True,  # Enable wandb for monitoring
            wandb_project=args.wandb_project,
            exp_name=f"PPO-BikeRacer-GPU{args.gpu}-Seed{args.seed}",
            exp_dir=exp_dir,
            logger=logger,
            max_steps=args.max_steps  # Pass max_steps to train_ppo
        )
        logger.info("Training complete!")
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        # Print stack trace for debugging
        import traceback

        logger.error(traceback.format_exc())

        # Try to close wandb if it's running
        try:
            if wandb.run is not None:
                wandb.finish()
        except:
            pass
        sys.exit(1)
