from gymnasium.envs.registration import register
import gymnasium as gym

register(
    id='MotionProfilePacman-v1',
    entry_point='env:MotionProfilePacman',
    max_episode_steps=300,
)

env = gym.make('MotionProfilePacman-v1', render_mode="human")

env.reset()

is_done = False
test_actions = [(23,21),(1,21)]
for action in test_actions:
    new_space, reward, is_done, state, info = env.step(action)
    env.render()
env.close()