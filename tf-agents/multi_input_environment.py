import numpy as np


class TwoInputEnvironment:
    def __init__(self):
        self.left_observation_shape = (5,)
        self.right_observation_shape = (10,)
        self.observation_dtype = np.int32
        self.observation_low = 0
        self.observation_high = 20
        self.action_space = {0, 1}

        self._left_obs = None
        self._right_obs = None

    def _get_observation(self):
        return {'left_observation': self._left_obs, 'right_observation': self._right_obs}

    def reset(self):
        self._left_obs = np.random.randint(
            low=self.observation_low,
            high=self.observation_high,
            size=self.left_observation_shape,
            dtype=self.observation_dtype
        )
        self._right_obs = np.random.randint(
            low=self.observation_low,
            high=self.observation_high,
            size=self.right_observation_shape,
            dtype=self.observation_dtype
        )
        return self._get_observation()

    def step(self, action):
        left_sum = np.sum(self._left_obs)
        right_sum = np.sum(self._right_obs)

        if action == 0 and left_sum >= right_sum:
            reward = 1
        elif action == 1 and left_sum < right_sum:
            reward = 1
        else:
            reward = -1
        done = True
        return self._get_observation(), reward, done

    def render(self):
        print('Left Observation: {}\tRight Observation: {}'.format(self._left_obs, self._right_obs))
