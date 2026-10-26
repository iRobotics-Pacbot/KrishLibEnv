from gymnasium.envs.registration import register
import gymnasium as gym
# from stable_baselines3 import PPO
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
import numpy as np
import csv
import os
import sys

ENV_ID = "MotionProfilePacman-v1"
MODEL_NAME = "dqn_pacbot"
NUM_EPISODES = 10
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "bin/results")
DEVICE = "mps"

# registers custom environment MotionProfilePacman from env.py 
register (
    id=ENV_ID,
    entry_point="env:MotionProfilePacman",
    max_episode_steps=10_000, # time limit per episode
)

REWARD_COMPONENTS = [
    "reward_score",
    "reward_large_score_bonus",
    "reward_score_event_bonus",
    "reward_step",
    "reward_life_loss",
    "reward_wall_hit",
    "reward_ghost_proximity",
    "reward_pellet_proximity",
]

# helper function for evaluate model
def collect_episode_metrics(info, episode):
    stats = info.get("episode_stats", {})

    metrics = {
        "model": MODEL_NAME,
        "episode": episode,
        "score": info.get("final_score", stats.get("score", 0)),
        "success": stats.get("successful", False),
        "steps": stats.get("steps", 0),
        "wall_hits": stats.get("wall_hits", 0),
        "lives_lost": stats.get("lives_lost", 0),
        "total_reward": stats.get("total_reward", 0),
    }

    for comp in REWARD_COMPONENTS:
        metrics[comp] = stats.get(comp, 0)

    return metrics

# save results of model in action sequences and per episode results
def evaluate_model(model, env):
    episode_results = []
    action_sequences = []
    current_actions = []

    obs = env.reset()

    while len(episode_results) < NUM_EPISODES:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)

        current_actions.append(int(action[0]))

        if done[0]:
            episode_number = len(episode_results) + 1
            stats = collect_episode_metrics(info[0], episode_results)

            episode_results.append(stats)
            action_sequences.append(current_actions.copy())

            print(
                f"Episode {episode_number}/{NUM_EPISODES}: "
                f"score={stats['score']}, "
                f"steps={stats['steps']}, "
                f"success={stats['success']}, "
                f"wall_hits={stats['wall_hits']}, "
                f"lives_lost={stats['lives_lost']}, "
                f"reward={stats['total_reward']:.2f}"
            )

            current_actions = []

    return episode_results, action_sequences

# combines all results in a unified summary
def summarize_results(episode_results):
    scores = [row["score"] for row in episode_results]
    rewards = [row["total_reward"] for row in episode_results]

    summary = {
        "model": MODEL_NAME,
        "num_episodes": len(episode_results),
        "mean_score": np.mean(scores),
        "median_score": np.median(scores),
        "std_score": np.std(scores),
        "min_score": np.min(scores),
        "max_score": np.max(scores),
        "success_rate": np.mean([row["success"] for row in episode_results]),
        "mean_total_reward": np.mean(rewards),
        "mean_wall_hits": np.mean([row["wall_hits"] for row in episode_results]),
        "mean_lives_lost": np.mean([row["lives_lost"] for row in episode_results]),
    }

    for comp in REWARD_COMPONENTS:
        summary[f"mean_{comp}"] =  np.mean([row[comp] for row in episode_results])

    return summary

# save to CSV
def save_csv(path, rows):
    print(os.getcwd())
    with open(path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

# printing summary of means and other stats
def print_summary(summary, action_sequences):
    print("\n" + "="*40)
    print("=== EVALUATION RESULTS ===")
    print("="*40+"\n")

    print(f"Mean score: {summary['mean_score']:.2f}")
    print(f"Median score: {summary['median_score']:.2f}")
    print(f"Std score: {summary['std_score']:.2f}")
    print(f"Min score: {summary['min_score']:.2f}")
    print(f"Max score: {summary['max_score']:.2f}")
    print(f"Success rate: {summary['success_rate']:.1%}")
    print(f"Mean total reward: {summary['mean_total_reward']:.2f}")
    print(f"Mean wall hits: {summary['mean_wall_hits']:.2f}")
    print(f"Mean lives lost: {summary['mean_lives_lost']:.2f}")

    print("\nMean reward components:")
    for comp in REWARD_COMPONENTS:
        print(f"{comp}: {summary[f'mean_{comp}']:.2f}")

    identical = all(
        sequence == action_sequences[0]
        for sequence in action_sequences
    )

    print(f"\nAll action sequences identical: {identical}")

# fully evaluate summary
def main(render):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    env = make_vec_env(
        ENV_ID,
        n_envs=1,
        env_kwargs={"render_mode": render},
        vec_env_cls=DummyVecEnv,
    )
    env = VecFrameStack(env, n_stack=4)

    try:
        # model = PPO.load(MODEL_NAME, env, device=DEVICE)
        model = DQN.load(MODEL_NAME, env, device=DEVICE)

        print(f"Evaluating {MODEL_NAME} over {NUM_EPISODES} episodes...\n")

        episode_results, action_sequences = evaluate_model(model, env)
        summary = summarize_results(episode_results)

        print_summary(summary, action_sequences)

        save_csv(
            os.path.join(OUTPUT_DIR, f"{MODEL_NAME}_episodes.csv"),
            episode_results,
        )
        save_csv(
            os.path.join(OUTPUT_DIR, f"{MODEL_NAME}_summary.csv"),
            [summary],
        )

        print(f"\nEpisode and summary metrics saved in {OUTPUT_DIR}/")

    finally:
        env.close()

if __name__ == "__main__":
    # if render is passed in shell args than the game will render
    render = None
    if len(sys.argv) > 1 and sys.argv[1] == "render":
        render = "human"

    main(render)
