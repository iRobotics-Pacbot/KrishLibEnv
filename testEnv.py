from gymnasium.envs.registration import register
import gymnasium as gym
from stable_baselines3.

register(
    id='MotionProfilePacman-v1',
    entry_point='env:MotionProfilePacman',
    max_episode_steps=300,
)

env = gym.make('MotionProfilePacman-v1', render_mode="human")

env.reset()

is_done = False
while not is_done:
    new_space, reward, is_done, state, info = env.step(action)
    env.render()
env.close()
