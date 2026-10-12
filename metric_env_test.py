import gymnasium as gym
from gymnasium.envs.registration import register

register(
    id="MotionProfilePacman-v1",
    entry_point="env:MotionProfilePacman",
    max_episode_steps=1_000,
)

env = gym.make("MotionProfilePacman-v1", render_mode=None)

obs, info = env.reset()
terminated = False
truncated = False

while not (terminated or truncated):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

print("Episode Finished")
print("Episode stats:", info.get("episode_stats", "MISSING"))

env.close()