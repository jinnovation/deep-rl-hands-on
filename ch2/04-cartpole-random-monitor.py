import random
from typing import TypeVar

import gym

Action = TypeVar("Action")


class RandomActionWrapper(gym.ActionWrapper):
    epsilon: float

    def __init__(self, env, epsilon: float = 0.1):
        super().__init__(env)
        self.epsilon = epsilon

    def action(self, action: Action) -> Action:
        if random.random() < self.epsilon:
            print("Random!")
            return self.env.action_space.sample()
        return action


if __name__ == "__main__":
    env = RandomActionWrapper(
        gym.wrappers.Monitor(gym.make("CartPole-v0"), "recordings", force=True)
    )
    obs = env.reset()
    total_reward = 0.0
    while True:
        obs, reward, done, _ = env.step(0)
        total_reward += reward
        if done:
            break
    print("Reward got: %.2f" % total_reward)
