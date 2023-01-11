import gym
import numpy as np


class KnapsackEnv(gym.Env):
    def __init__(self, env_config: dict):
        self._max_weight = 15.0
        self._item_weights = np.float32([1, 1, 2, 3, 3, 4, 5, 8, 10, 12])
        self._item_values = np.float32([1, 2, 3, 2, 6, 5, 10, 5, 6, 4])
        self._N = self._item_weights.shape[0]

        self.action_space = gym.spaces.Discrete(n=10)

        obs_space = gym.spaces.Dict({
            'observation_weights': gym.spaces.Box(low=1.0, high=self._max_weight, shape=(self._N,), dtype=np.float32),
            'observation_values': gym.spaces.Box(low=1.0, high=np.inf, shape=(self._N,), dtype=np.float32),
            'observation_knapsack': gym.spaces.Box(low=0.0, high=np.inf, shape=(1,), dtype=np.float32)
        })
        self.observation_space = gym.spaces.Dict({
            'action_mask': gym.spaces.Box(low=0.0, high=1.0, shape=(self._N,), dtype=np.float32),
            'observations': obs_space
        })

        self._action_mask = None
        self._knapsack_weight = np.float32([0.0])

    def _update_action_mask(self):
        for i, weight in enumerate(self._item_weights):
            if self._knapsack_weight[0] + weight > self._max_weight:
                self._action_mask[i] = 0.0

    def _get_state(self) -> dict:
        return {
            'action_mask': self._action_mask,
            'observations': {
                'observation_weights': self._item_weights,
                'observation_values': self._item_values,
                'observation_knapsack': self._knapsack_weight
            }
        }

    def reset(self) -> dict:
        self._action_mask = np.ones(shape=(self._N,), dtype=np.float32)
        self._knapsack_weight[0] = 0.0
        return self._get_state()

    def step(self, action: int) -> (dict, float, bool, dict):
        assert self._action_mask[action] == 1.0, f'AssertionError: Illegal Action "{action}"'

        self._knapsack_weight[0] += self._item_weights[action]
        self._action_mask[action] = 0.0
        reward = self._item_values[action]

        self._update_action_mask()
        done = self._action_mask.sum() == 0.0

        next_state = self._get_state()
        return next_state, reward, done, {}

    def render(self, mode=None):
        for observation_name, observation in self._get_state().items():
            print(f'{observation_name}: {observation}')
