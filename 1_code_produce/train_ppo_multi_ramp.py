"""
Train PPO for the 2-ramp flat_peak scenario.

This uses a centralized policy with 8-D observation (both ramps) and 2-D action
(one metering action per ramp).
"""

import argparse
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from multi_ramp_env import MultiRampMeteringEnv


def parse_args():
    parser = argparse.ArgumentParser(description="Train PPO on multi-ramp flat_peak")
    parser.add_argument("--timesteps", type=int, default=300_000, help="Total PPO timesteps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--use-gui", action="store_true", help="Use SUMO GUI")
    parser.add_argument(
        "--model-name",
        type=str,
        default="ppo_multi_flat_peak",
        help="Model filename stem",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    models_dir = Path(__file__).parent.parent / "2_data_produce" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    def make_env():
        return MultiRampMeteringEnv(use_gui=args.use_gui)

    env = DummyVecEnv([make_env])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        seed=args.seed,
        n_steps=1024,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        learning_rate=3e-4,
        ent_coef=0.0,
        vf_coef=0.5,
        clip_range=0.2,
        gae_lambda=0.95,
        tensorboard_log=str(Path(__file__).parent.parent / "2_data_produce" / "logs" / "tensorboard"),
    )

    model.learn(total_timesteps=args.timesteps, progress_bar=True)

    model_path = models_dir / f"{args.model_name}.zip"
    vecnorm_path = models_dir / f"{args.model_name}_vecnormalize.pkl"

    model.save(model_path)
    env.save(str(vecnorm_path))
    env.close()

    print(f"Saved model: {model_path}")
    print(f"Saved vecnormalize: {vecnorm_path}")


if __name__ == "__main__":
    main()
