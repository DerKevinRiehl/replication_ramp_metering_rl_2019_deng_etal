"""
PPO Training Script for Ramp Metering.
Trains the PPO agent as described in Section II.D and III.C of Deng et al. (2019).

Usage:
    python train_ppo.py --scenario flat_peak --episodes 1000
    
As per Section III.C.1, PPO is trained on the flat_peak demand profile.
Training takes approximately 6 hours on CPU for 1000 episodes.
Progress can be monitored via TensorBoard.
"""

import argparse
import os
import sys
from pathlib import Path
from datetime import datetime
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for headless plotting
import matplotlib.pyplot as plt

from config import (
    SCENARIOS, TRAINING_EPISODES, TRAINING_TIMESTEPS,
    LOGS_DIR, get_model_path, RANDOM_SEED, NUM_CONTROL_PERIODS,
    PPO_GAMMA, VIS_DATA_DIR
)
from sumo_env import SumoRampMeteringEnv
from controllers.ppo_controller import PPOController
from utils import set_random_seed, print_progress

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import (
        BaseCallback, CheckpointCallback, EvalCallback, CallbackList
    )
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
except ImportError:
    print("Error: stable-baselines3 is required for training.")
    print("Install with: pip install stable-baselines3")
    sys.exit(1)


class DebugTrainingCallback(BaseCallback):
    """
    Custom callback that prints episode-level and rollout-level training
    diagnostics so you can verify the agent is learning.
    """

    def __init__(self, print_freq_episodes: int = 5, verbose: int = 1):
        super().__init__(verbose)
        self.print_freq = print_freq_episodes
        # episode tracking
        self.episode_count = 0
        self.ep_returns = []
        self.ep_lengths = []
        self.ep_speeds = []   # avg bottleneck speed per episode (km/h)
        self.ep_queues = []   # avg ramp queue per episode (m)
        # step-level action tracking within current episode
        self._ep_actions = []
        # rollout tracking
        self.rollout_count = 0

    # ------------------------------------------------------------------
    # Called every env step
    # ------------------------------------------------------------------
    def _on_step(self) -> bool:
        # Record action taken this step (clipped to [0,1] as the env sees it)
        action = self.locals.get("actions")
        if action is not None:
            clipped = float(np.clip(action.flatten()[0], 0.0, 1.0))
            self._ep_actions.append(clipped)

        # Check for episode end via Monitor wrapper info
        infos = self.locals.get("infos", [])
        for info in infos:
            maybe_ep = info.get("episode")
            if maybe_ep is not None:
                self.episode_count += 1
                ep_ret = maybe_ep["r"]
                ep_len = maybe_ep["l"]
                self.ep_returns.append(ep_ret)
                self.ep_lengths.append(ep_len)

                # Grab env-level stats if available
                avg_speed = info.get("avg_speed", None)
                avg_queue = info.get("avg_queue", None)
                # Store numeric values (fallback to NaN if missing)
                self.ep_speeds.append(float(avg_speed) if avg_speed is not None else float('nan'))
                self.ep_queues.append(float(avg_queue) if avg_queue is not None else float('nan'))

                # Action stats for the episode
                actions = np.array(self._ep_actions) if self._ep_actions else np.zeros(1)
                act_mean = actions.mean()
                act_std  = actions.std()
                act_min  = actions.min()
                act_max  = actions.max()
                mr_mean = 60.0 + act_mean * 1140.0  # MR_MIN + a*(MR_MAX-MR_MIN)

                if self.episode_count % self.print_freq == 0 or self.episode_count <= 10:
                    print(f"\n{'='*70}")
                    print(f"  EPISODE {self.episode_count}  "
                          f"(timestep {self.num_timesteps})")
                    print(f"{'='*70}")
                    print(f"  Return:    {ep_ret:>10.2f}")
                    print(f"  Length:    {ep_len:>10d} steps")
                    print(f"  Avg speed: {avg_speed if avg_speed is None else f'{avg_speed:>10.2f}'} km/h")
                    print(f"  Avg queue: {avg_queue if avg_queue is None else f'{avg_queue:>10.2f}'} m")
                    print(f"  Actions:   mean={act_mean:.4f}  std={act_std:.4f}  "
                          f"range=[{act_min:.4f}, {act_max:.4f}]")
                    print(f"  MR(mean):  {mr_mean:.1f} veh/h")

                    # Rolling averages over last N episodes
                    for window in [10, 50]:
                        if len(self.ep_returns) >= window:
                            recent = self.ep_returns[-window:]
                            print(f"  Last-{window} avg return: {np.mean(recent):.2f}  "
                                  f"(std {np.std(recent):.2f})")
                    print()

                # Reset per-episode action buffer
                self._ep_actions = []

        return True  # continue training

    # ------------------------------------------------------------------
    # Called after each PPO rollout (every n_steps timesteps)
    # ------------------------------------------------------------------
    def _on_rollout_end(self) -> None:
        self.rollout_count += 1
        # SB3 stores training losses in self.model.logger
        log = self.logger.name_to_value  # dict of recent logged scalars
        policy_loss = log.get("train/policy_gradient_loss", None)
        value_loss  = log.get("train/value_loss", None)
        entropy     = log.get("train/entropy_loss", None)
        clip_frac   = log.get("train/clip_fraction", None)
        approx_kl   = log.get("train/approx_kl", None)
        explained_var = log.get("train/explained_variance", None)

        if policy_loss is not None:
            print(f"  [Rollout {self.rollout_count}] "
                  f"policy_loss={policy_loss:.6f}  "
                  f"value_loss={value_loss:.4f}  "
                  f"entropy={entropy:.4f}  "
                  f"clip_frac={clip_frac:.4f}  "
                  f"approx_kl={approx_kl:.6f}  "
                  f"expl_var={explained_var:.4f}")

    def on_training_end(self) -> None:
        print(f"\n{'#'*70}")
        print(f"  TRAINING COMPLETE: {self.episode_count} episodes")
        print(f"{'#'*70}")
        if self.ep_returns:
            returns = np.array(self.ep_returns)
            print(f"  Overall return:  mean={returns.mean():.2f}  "
                  f"std={returns.std():.2f}  "
                  f"min={returns.min():.2f}  max={returns.max():.2f}")
            # First 10 vs last 10
            if len(returns) >= 20:
                first10 = returns[:10].mean()
                last10  = returns[-10:].mean()
                print(f"  First-10 avg return: {first10:.2f}")
                print(f"  Last-10  avg return: {last10:.2f}")
                print(f"  Improvement:         {last10 - first10:+.2f}  "
                      f"({(last10 - first10) / abs(first10) * 100:+.1f}%)")
        print()


def plot_training_process(
    callback: 'DebugTrainingCallback',
    scenario: str,
    output_dir: Path = None,
):
    """
    Generate the 3-panel training process figure (cf. paper Fig. 7).

    Produces one combined PNG with:
      (a) Episode Return        — red
      (b) Avg Bottleneck Speed  — green
      (c) Avg Ramp Queue Length — blue

    A smoothed (EMA) curve is overlaid on top of the raw data.
    """
    if output_dir is None:
        output_dir = VIS_DATA_DIR / "training_curves"
    output_dir.mkdir(parents=True, exist_ok=True)

    episodes = np.arange(1, len(callback.ep_returns) + 1)
    returns  = np.array(callback.ep_returns, dtype=float)
    speeds   = np.array(callback.ep_speeds, dtype=float)
    queues   = np.array(callback.ep_queues, dtype=float)

    def ema(values, alpha=0.95):
        """Exponential moving average (ignores NaNs)."""
        out = np.empty_like(values)
        last = values[np.isfinite(values)][0] if np.any(np.isfinite(values)) else 0.0
        for i, v in enumerate(values):
            if np.isfinite(v):
                last = alpha * last + (1 - alpha) * v
            out[i] = last
        return out

    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    fig.suptitle(
        f'PPO Training Process — {scenario.replace("_", " ").title()}\n'
        f'(cf. Deng et al. 2019, Fig. 7)',
        fontsize=15, fontweight='bold', y=0.98,
    )

    # ---- (a) Return per episode — RED ----
    ax = axes[0]
    ax.plot(episodes, returns, color='#e74c3c', alpha=0.25, linewidth=0.8, label='Raw')
    ax.plot(episodes, ema(returns), color='#c0392b', linewidth=2.0, label='Smoothed (EMA)')
    ax.set_ylabel('Episode Return', fontsize=12, fontweight='bold')
    ax.set_title('(a) Episode Return', fontsize=13)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)

    # ---- (b) Avg speed per episode — GREEN ----
    ax = axes[1]
    valid_speed = np.isfinite(speeds)
    if valid_speed.any():
        ax.plot(episodes[valid_speed], speeds[valid_speed],
                color='#27ae60', alpha=0.25, linewidth=0.8, label='Raw')
        ax.plot(episodes, ema(speeds), color='#1e8449', linewidth=2.0, label='Smoothed (EMA)')
    else:
        ax.text(0.5, 0.5, 'Speed data not available',
                ha='center', va='center', transform=ax.transAxes, fontsize=12)
    ax.set_ylabel('Avg Speed (km/h)', fontsize=12, fontweight='bold')
    ax.set_title('(b) Average Bottleneck Speed', fontsize=13)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)

    # ---- (c) Avg queue per episode — BLUE ----
    ax = axes[2]
    valid_queue = np.isfinite(queues)
    if valid_queue.any():
        ax.plot(episodes[valid_queue], queues[valid_queue],
                color='#2980b9', alpha=0.25, linewidth=0.8, label='Raw')
        ax.plot(episodes, ema(queues), color='#1a5276', linewidth=2.0, label='Smoothed (EMA)')
    else:
        ax.text(0.5, 0.5, 'Queue data not available',
                ha='center', va='center', transform=ax.transAxes, fontsize=12)
    ax.set_ylabel('Avg Queue Length (m)', fontsize=12, fontweight='bold')
    ax.set_title('(c) Average Ramp Queue Length', fontsize=13)
    ax.set_xlabel('Episode', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = output_dir / f'{scenario}_training_process.png'
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"\nTraining process figure saved to: {out_path}")

    # Also save the raw data as CSV for later analysis
    csv_path = output_dir / f'{scenario}_training_log.csv'
    import csv
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['episode', 'return', 'avg_speed_kmh', 'avg_queue_m'])
        for i in range(len(episodes)):
            writer.writerow([int(episodes[i]), returns[i], speeds[i], queues[i]])
    print(f"Training log CSV saved to: {csv_path}")


def make_env(scenario: str, rank: int = 0, seed: int = 0, add_noise: bool = True):
    """
    Create a wrapped environment for training.
    
    Args:
        scenario: Traffic demand scenario
        rank: Environment rank (for parallel training)
        seed: Random seed
        add_noise: Whether to add demand noise (for generalization)
    """
    def _init():
        env = SumoRampMeteringEnv(
            scenario=scenario,
            use_gui=False,
            add_noise=add_noise,
            noise_std=0.05,  # Small noise for training
        )
        env = Monitor(env)
        return env
    
    return _init


def train(
    scenario: str = "flat_peak",
    total_timesteps: int = None,
    num_envs: int = 1,
    seed: int = RANDOM_SEED,
    checkpoint_freq: int = 10000,
    eval_freq: int = 5000,
    save_path: str = None,
    resume_from: str = None,
):
    """
    Train PPO agent for ramp metering.
    
    Args:
        scenario: Traffic scenario for training
        total_timesteps: Total training timesteps (default: 1000 episodes worth)
        num_envs: Number of parallel environments
        seed: Random seed
        checkpoint_freq: Steps between checkpoints
        eval_freq: Steps between evaluations
        save_path: Path to save final model
        resume_from: Path to resume training from
    """
    print("=" * 60)
    print(f"PPO Training for Ramp Metering - Scenario: {scenario}")
    print("=" * 60)
    print(f"Paper reference: Deng et al. (2019) ITSC")
    print(f"Training should take ~6 hours on CPU (1000 episodes)")
    print("=" * 60)
    
    # Set random seed
    set_random_seed(seed)
    
    # Calculate total timesteps
    if total_timesteps is None:
        total_timesteps = TRAINING_TIMESTEPS
    
    print(f"\nConfiguration:")
    print(f"  Scenario: {scenario}")
    print(f"  Total timesteps: {total_timesteps:,}")
    print(f"  Episodes (approx): {total_timesteps // NUM_CONTROL_PERIODS:,}")
    print(f"  Parallel envs: {num_envs}")
    print(f"  Random seed: {seed}")
    
    # Create directories
    log_dir = LOGS_DIR / "tensorboard" / f"{scenario}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    checkpoint_dir = LOGS_DIR / "checkpoints" / scenario
    log_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"  TensorBoard logs: {log_dir}")
    print(f"  Checkpoints: {checkpoint_dir}")
    
    # Create environment(s)
    print("\nCreating training environment...")
    if num_envs > 1:
        env = SubprocVecEnv([
            make_env(scenario, i, seed + i) 
            for i in range(num_envs)
        ])
    else:
        env = DummyVecEnv([make_env(scenario, 0, seed)])
    
    # ── CRITICAL FIX: Reward normalisation ──────────────────────────────
    # Without normalisation the returns are ~5000, which is far too large
    # for the value network to fit → explained_variance stays at ~0.
    # VecNormalize divides rewards by the running std of discounted returns
    # so the value targets become O(1), letting the critic converge.
    # norm_obs=False because occupancies are already in [0, 1].
    env = VecNormalize(
        env,
        norm_obs=False,
        norm_reward=True,
        gamma=PPO_GAMMA,
        clip_reward=10.0,
    )
    
    # Create evaluation environment (no noise, with reward normalisation)
    eval_env = DummyVecEnv([make_env(scenario, 0, seed, add_noise=False)])
    eval_env = VecNormalize(
        eval_env,
        norm_obs=False,
        norm_reward=True,
        gamma=PPO_GAMMA,
        clip_reward=10.0,
        training=False,  # Important: do not update stats during eval
    )
    
    # Create or load model
    if resume_from and Path(resume_from).exists():
        print(f"\nResuming training from: {resume_from}")
        model = PPO.load(resume_from, env=env)
    else:
        print("\nCreating new PPO model...")
        # Try tensorboard logging, fallback to no logging if not available
        try:
            import tensorboard
            tensorboard_log = str(log_dir)
        except ImportError:
            print("TensorBoard not available, using console logging only")
            tensorboard_log = None
            
        model = PPOController.create_model(
            env=env,
            tensorboard_log=tensorboard_log
        )
    
    # Setup callbacks
    callbacks = []

    # Debug training callback (prints per-episode diagnostics)
    debug_callback = DebugTrainingCallback(print_freq_episodes=5)
    callbacks.append(debug_callback)
    
    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_freq // num_envs,
        save_path=str(checkpoint_dir),
        name_prefix=f"ppo_{scenario}",
        save_replay_buffer=False,
        save_vecnormalize=True,   # save VecNormalize running stats
    )
    callbacks.append(checkpoint_callback)
    
    # Evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(checkpoint_dir),
        log_path=str(log_dir),
        eval_freq=eval_freq // num_envs,
        n_eval_episodes=3,
        deterministic=True,
    )
    callbacks.append(eval_callback)
    
    callback_list = CallbackList(callbacks)
    
    # Train
    print("\n" + "=" * 60)
    print("Starting training...")
    print("Monitor progress with: tensorboard --logdir", log_dir.parent)
    print("=" * 60 + "\n")
    
    try:
        # Try with progress bar, fallback without if libraries missing
        try:
            model.learn(
                total_timesteps=total_timesteps,
                callback=callback_list,
                progress_bar=True,
            )
        except ImportError:
            print("Progress bar libraries not available, training without progress bar...")
            model.learn(
                total_timesteps=total_timesteps,
                callback=callback_list,
                progress_bar=False,
            )
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
    finally:
        # Save final model
        if save_path is None:
            save_path = get_model_path(scenario, TRAINING_EPISODES)
        
        save_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(save_path))
        print(f"\nModel saved to: {save_path}")
        
        # Save VecNormalize running statistics (needed if norm_obs were True)
        vec_norm_path = str(save_path).replace('.zip', '_vecnormalize.pkl')
        env.save(vec_norm_path)
        print(f"VecNormalize stats saved to: {vec_norm_path}")
        
        # ── Generate training process plots (Paper Fig. 7) ──────────
        if debug_callback.ep_returns:
            print("\nGenerating training process plots (Paper Fig. 7)...")
            try:
                plot_training_process(debug_callback, scenario)
            except Exception as exc:
                print(f"Warning: could not generate training plots: {exc}")
        
        # Cleanup
        env.close()
        eval_env.close()
    
    print("\nTraining complete!")
    return model


def main():
    parser = argparse.ArgumentParser(
        description="Train PPO agent for ramp metering (Deng et al. 2019)"
    )
    parser.add_argument(
        "--scenario", "-s",
        type=str,
        default="flat_peak",
        choices=SCENARIOS,
        help="Traffic demand scenario (paper trains on flat_peak)"
    )
    parser.add_argument(
        "--episodes", "-e",
        type=int,
        default=TRAINING_EPISODES,
        help="Number of training episodes"
    )
    parser.add_argument(
        "--timesteps", "-t",
        type=int,
        default=None,
        help="Total training timesteps (overrides --episodes)"
    )
    parser.add_argument(
        "--num-envs", "-n",
        type=int,
        default=1,
        help="Number of parallel environments"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help="Random seed"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to model to resume training from"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output path for trained model"
    )
    
    args = parser.parse_args()
    
    # Calculate timesteps from episodes if not specified
    timesteps = args.timesteps
    if timesteps is None:
        timesteps = args.episodes * NUM_CONTROL_PERIODS
    
    # Convert output path if specified
    output_path = Path(args.output) if args.output else None
    
    train(
        scenario=args.scenario,
        total_timesteps=timesteps,
        num_envs=args.num_envs,
        seed=args.seed,
        save_path=output_path,
        resume_from=args.resume,
    )


if __name__ == "__main__":
    main()
