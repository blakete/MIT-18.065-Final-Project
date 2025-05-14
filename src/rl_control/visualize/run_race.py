#!/usr/bin/env python3
import argparse
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from PIL import Image, ImageDraw

from rl_control.train.torch_env import VectorizedRaceBikeDojo
from rl_control.train.ppo_agent import PPOAgent


def run_race_course(
        model_path: str,
        waypoint_csv: str,
        device: torch.device,
        max_steps: int = 2000,
        generate_gif: bool = True,
        state_output_path: str = "states.npy",
        completed_race_pause_frames: int = 500
):
    # 1) load agent checkpoint & infer hidden size
    ckpt = torch.load(model_path, map_location=device)
    hidden_dim = ckpt['actor_mean.0.weight'].shape[0]
    agent = PPOAgent(obs_dim=4, hidden_dim=hidden_dim, action_dim=2, device=device)
    agent.load_state_dict(ckpt)
    agent.to(device).eval()

    # 2) load your CSV of waypoints
    df = pd.read_csv(waypoint_csv)
    waypoints = df[['x', 'y']].values.astype(np.float32)  # (N,2)
    num_wps = len(waypoints)
    wp_idx = 0  # start at first waypoint

    # 3) make env & override its reset so it never zeroes out your bike
    env = VectorizedRaceBikeDojo(num_envs=1, device=device, max_steps=max_steps)
    # disable env's automatic reset when waypoint captured:
    env.reset_if_done = lambda done_mask: None
    global_state = env.reset()  # we won't use this obs 

    # 4) set first target waypoint in env
    env.waypoints = torch.from_numpy(waypoints[0]).to(env.device).unsqueeze(0)

    frames = []
    if generate_gif:
        # 5) prepare plotting—auto‐scale to cover entire track plus 10%
        xmin, xmax = waypoints[:, 0].min(), waypoints[:, 0].max()
        ymin, ymax = waypoints[:, 1].min(), waypoints[:, 1].max()
        spanx, spany = xmax - xmin, ymax - ymin
        m = max(spanx, spany) * 0.5  # margin
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_xlim(xmin - m, xmax + m)
        ax.set_ylim(ymin - m, ymax + m)
        ax.set_aspect("equal")
        ax.grid(True)
        ax.set_title("Race Course")

        # draw all waypoints as blue dots
        ax.scatter(waypoints[:, 0], waypoints[:, 1], c='blue', s=20, label='waypoints')

        # dynamic capture‐circle
        wp_circle = Circle(
            (waypoints[0, 0], waypoints[0, 1]),
            env.capture_threshold,
            fc="green", alpha=0.4
        )
        ax.add_patch(wp_circle)

        # bike, heading & speed
        bike = Circle((0, 0), 0.05, fc="red", alpha=0.8)
        ax.add_patch(bike)
        heading_line, = ax.plot([], [], lw=2, color="black")
        speed_line, = ax.plot([], [], lw=4, color="orange")

        # trail
        trail_x, trail_y = [], []
        trail_line, = ax.plot([], [], "r-", lw=1, alpha=0.5)

        # stats text
        stats = ax.text(
            0.02, 0.95, "", transform=ax.transAxes,
            va="top", ha="left",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
        )

    total_reward = 0.0
    captured = 0

    # helper to _both_ advance index _and_ rebuild the obs
    def advance_waypoint_and_obs():
        nonlocal wp_idx, captured, obs
        captured += 1
        wp_idx += 1
        nx, ny = waypoints[wp_idx]
        # update the green circle
        if generate_gif:
            wp_circle.center = (nx, ny)
        # tell env about the new target
        env.waypoints = torch.tensor([nx, ny], device=env.device).unsqueeze(0)
        # build a fresh "obs" relative to the new origin:
        #   take current global state and subtract new waypoint origin
        gx, gy, gth, gsp = env.state[0].cpu().numpy()
        dx = nx - gx
        dy = ny - gy
        # Relative heading:
        ang = np.arctan2(dy, dx) - gth
        ang = np.arctan2(np.sin(ang), np.cos(ang))
        rel_obs = np.array([dx, dy, ang, gsp], dtype=np.float32)
        obs = torch.from_numpy(rel_obs).unsqueeze(0).to(device)

    # 6) initial obs: we compute manually so it's relative to waypoint
    gx, gy, gth, gsp = env.state[0].cpu().numpy()
    dx0 = waypoints[0, 0] - gx
    dy0 = waypoints[0, 1] - gy
    ang0 = np.arctan2(dy0, dx0) - gth
    ang0 = np.arctan2(np.sin(ang0), np.cos(ang0))
    obs = torch.from_numpy(np.array([dx0, dy0, ang0, gsp], np.float32)).unsqueeze(0).to(device)

    # in case we start _right_ on the waypoint
    if np.hypot(dx0, dy0) < env.capture_threshold and wp_idx < num_wps - 1:
        advance_waypoint_and_obs()

    all_states = np.nan * np.ones((max_steps, 4), dtype=np.float32)
    # 7) main loop
    for step in range(max_steps):
        # action from obs
        with torch.no_grad():
            scaled_action, _, _, _, _ = agent.get_action_and_value(obs)

        # step the environment (global frame)
        obs_global, reward, dones, _, state = env.step(scaled_action)
        total_reward += reward.item()

        tmp = state.cpu().numpy()[0]
        all_states[step, :] = tmp

        # read global state
        gx, gy, gth, gsp = env.state[0].cpu().numpy()
        # compute relative to current target
        tgtx, tgty = waypoints[wp_idx]
        dx = tgtx - gx
        dy = tgty - gy
        ang = np.arctan2(dy, dx) - gth
        ang = np.arctan2(np.sin(ang), np.cos(ang))
        obs = torch.from_numpy(np.array([dx, dy, ang, gsp], np.float32)).unsqueeze(0).to(device)

        # if within threshold, advance
        if np.hypot(dx, dy) < env.capture_threshold:
            if wp_idx < num_wps - 1:
                advance_waypoint_and_obs()
            else:
                # Last waypoint captured - end early
                captured += 1
                if generate_gif:
                    print(f"Last waypoint captured at step {step + 1}/{max_steps}! Ending simulation.")

                    # Add completion message to stats text
                    stats.set_text(
                        f"RACE COMPLETED!\n"
                        f"Step:      {step + 1}\n"
                        f"Reward:    {total_reward:.2f}\n"
                        f"Captured:  {captured}/{num_wps}"
                    )

                    # Make text size larger and make it red for completion
                    stats.set_fontsize(14)
                    stats.set_color('red')

                    # Draw one more frame with updated stats
                    fig.canvas.draw()
                    completion_img = Image.frombytes(
                        "RGB",
                        fig.canvas.get_width_height(),
                        fig.canvas.tostring_rgb()
                    )
                    frames.append(completion_img)

                    # # Save the completion frame directly to verify it's being created
                    # completion_img.save(state_output_path + ".completion_frame.png")

                    # Add pause frames - make each one visibly different to ensure they're preserved
                    for i in range(completed_race_pause_frames):
                        # Create a copy of the completion image
                        pause_frame = completion_img.copy()
                        # Add a timestamp at the bottom to make each frame visibly unique
                        # This prevents PIL from optimizing away duplicate frames
                        draw = ImageDraw.Draw(pause_frame)
                        # # Make a visible but subtle frame counter in the corner
                        # draw.text((5, 5), f"Frame {i+1}/{completed_race_pause_frames}", 
                        #           fill=(255, 0, 0))
                        frames.append(pause_frame)

                    print(f"Added {completed_race_pause_frames} pause frames with visible counters")

                # Create a properly sized state array with only the actual steps
                actual_steps = step + 1
                final_states = all_states[:actual_steps].copy()
                all_states = final_states
                print(f"Race completed in {actual_steps} steps out of maximum {max_steps}")
                break

        # draw
        if generate_gif:
            trail_x.append(gx);
            trail_y.append(gy)
            bike.center = (gx, gy)
            heading_line.set_data(
                [gx, gx + 0.1 * np.cos(gth)],
                [gy, gy + 0.1 * np.sin(gth)]
            )
            speed_line.set_data(
                [gx, gx + gsp * np.cos(gth)],
                [gy, gy + gsp * np.sin(gth)]
            )
            trail_line.set_data(trail_x, trail_y)
            stats.set_text(
                f"Step:      {step + 1}\n"
                f"Reward:    {total_reward:.2f}\n"
                f"Captured:  {captured}/{num_wps}"
            )

            # record frame
            fig.canvas.draw()
            img = Image.frombytes(
                "RGB",
                fig.canvas.get_width_height(),
                fig.canvas.tostring_rgb()
            )
            frames.append(img)

    if generate_gif:
        plt.close(fig)
        # Debug: Check that the frames list contains the expected number of frames
        total_frames = len(frames)
        print(f"run_race_course is returning {total_frames} frames")
        if total_frames > 10:
            print(f"First few frames dimensions: {frames[0].size}")
            print(f"Last few frames dimensions: {frames[-1].size}")
            # # Save the very last frame as another verification
            # frames[-1].save(state_output_path + ".last_frame_verification.png")

    return frames, all_states


def save_gif(frames, output_path, fps=20):
    # Debug information about frames
    print(f"Total frames to save: {len(frames)}")

    # Create a list of durations - one per frame
    # This allows for variable frame timing if needed
    duration = int(1000 / fps)  # Convert fps to milliseconds
    durations = [duration] * len(frames)

    # For the last 50 frames (or less if there aren't that many), make them stay longer
    # This ensures the completion message is visible
    longer_duration = duration * 3
    for i in range(max(0, len(frames) - 50), len(frames)):
        durations[i] = longer_duration

    # Save the gif with the specified durations
    try:
        # # First, save the final frame as a separate image for verification
        # if len(frames) > 0:
        #     frames[-1].save(output_path + ".final_frame.png")
        #     print(f"Saved final frame as {output_path}.final_frame.png for verification")

        # Try saving the GIF with different options
        frames[0].save(
            output_path,
            save_all=True,
            append_images=frames[1:],
            loop=0,
            duration=durations,
            disposal=2,  # Each frame completely replaces the previous frame
            optimize=False  # Disable optimization to preserve all frames
        )
        print("Successfully saved GIF with all frames")
    except Exception as e:
        print(f"Error saving GIF: {e}")

        # Try alternative method if the first fails
        try:
            # Fallback to simpler GIF saving
            frames[0].save(
                output_path + ".backup.gif",
                format='GIF',
                save_all=True,
                append_images=frames[1:],
                duration=duration,
                loop=0
            )
            print(f"Created backup GIF at {output_path}.backup.gif")
        except Exception as backup_error:
            print(f"Backup GIF also failed: {backup_error}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="path to .pt checkpoint")
    p.add_argument("--waypoints", required=True,
                   help="CSV of waypoints: columns waypoint_id,x,y")
    p.add_argument("--gif_output_path", default="race_course.gif", help="where to write the GIF")
    p.add_argument("--steps", type=int, default=200, help="total inference steps")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="inference device")
    p.add_argument("--generate_gif", type=lambda x: x.lower() == "true", default=True, help="generate gif (true/false)")
    p.add_argument("--state_output_path", type=str, default="states.npy", help="path to save states")
    p.add_argument("--pause_frames", type=int, default=200, help="number of frames to pause at end of completed race")
    args = p.parse_args()

    device = torch.device(args.device)
    print("Running race course...")
    frames, states = run_race_course(
        args.model,
        args.waypoints,
        device,
        generate_gif=args.generate_gif,
        max_steps=args.steps,
        state_output_path=args.state_output_path,
        completed_race_pause_frames=args.pause_frames
    )

    print(f"Saving states to {args.state_output_path}")
    np.save(args.state_output_path, states)
    print(f"💾 Wrote states to {args.state_output_path}")

    if args.generate_gif:
        print("Saving gif...")
        save_gif(frames, args.gif_output_path)
        print(f"💾 Wrote race-course visualization to {args.gif_output_path}")
