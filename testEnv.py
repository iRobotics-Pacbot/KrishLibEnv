from gymnasium.envs.registration import register
import gymnasium as gym
import random
import time

try:
    register(
        id="MotionProfilePacman-v1",
        entry_point="env:MotionProfilePacman",
        max_episode_steps=300,
    )
except:
    pass

env = gym.make("MotionProfilePacman-v1", render_mode="human")

env.reset()

for i in range(100):
    # Sample a random action from the action space
    action = env.action_space.sample()
    print(f"Step {i}: Action {action}")
    observation, reward, terminated, truncated, info = env.step(action)
    time.sleep(0.25)
    env.render()
    if terminated or truncated:
        env.reset()

env.close()