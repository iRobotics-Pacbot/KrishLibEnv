from gymnasium.envs.registration import register
import gymnasium as gym

register(
    id="MotionProfilePacman-v1",
    entry_point="env:MotionProfilePacman",
    max_episode_steps=300,
)

env = gym.make("MotionProfilePacman-v1", render_mode="human")

env.reset()

action_list = []
test_actions = [(25,2),(25,0)]
for action in test_actions:
    new_space, reward, is_done, state, info = env.step(action)
    env.render()
env.close()
