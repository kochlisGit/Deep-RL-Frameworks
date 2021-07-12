from tf_agents.environments.py_environment import PyEnvironment
from tf_agents.environments import utils
from tf_agents.specs.array_spec import BoundedArraySpec
from tf_agents.trajectories import time_step
import numpy as np
import gym


def validate_enviroment(env, episodes):
    utils.validate_py_environment(environment=env, episodes=episodes)


class LunarLanderDiscrete(PyEnvironment):
    def __init__(self, render=False):
        super().__init__()
        self._render = render

        # Loading the environment.
        self.env = gym.make('LunarLander-v2')

        # Creating the observation &  action space.
        self._action_spec = BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=self.env.action_space.n-1, name='action'
        )
        self._observation_spec = BoundedArraySpec(
            shape=self.env.observation_space.shape, dtype=np.float32,
            minimum=self.env.observation_space.low, maximum=self.env.observation_space.high, name='observation'
        )
        self._done = False
        self.discount_rate = 0.99

    # Returns the _action_spec.
    def action_spec(self):
        return self._action_spec

    # Returns the _observation_spec.
    def observation_spec(self):
        return self._observation_spec

    # Resets the environment & returns the initial agent's state.
    def _reset(self):
        self._done = False
        observation = self.env.reset()
        return time_step.restart(observation=observation)

    # Executes a step in the environment.
    def _step(self, action_spec):
        if self._done:
            return self._reset()

        action = action_spec.item()
        observation, reward, self._done, _ = self.env.step(action)

        if self._render:
            self.env.render()

        if self._done:
            return time_step.termination(observation=observation, reward=reward)
        else:
            return time_step.transition(observation=observation, reward=reward, discount=self.discount_rate)


class LunarLanderContinuous(PyEnvironment):
    pass
