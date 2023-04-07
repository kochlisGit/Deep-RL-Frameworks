import gym
import numpy as np


class CartPoleEnv(gym.Env):
    def __init__(self, env_config: dict):
        self._verbose = False if 'verbose' not in env_config else env_config['verbose']

        self._env = gym.make('CartPole-v1')
        self.action_space = gym.spaces.Discrete(self._env.action_space.n)
        self.observation_space = gym.spaces.Box(
            self._env.observation_space.low,
            self._env.observation_space.high,
            self._env.observation_space.shape,
            self._env.observation_space.dtype
        )

        self._step = 0

    def reset(self) -> np.ndarray:
        observation = self._env.reset()
        return observation

    def step(self, action: int) -> (np.ndarray, float, bool, dict):
        state, reward, done, info = self._env.step(action)

        if self._verbose:
            self._step += 1

            if done:
                print(f'Episode done at step {self._step}')

        return state, reward, done, info

    def render(self, mode: str or None = None):
        self._env.render()


class MaskedCartPoleEnv(gym.Env):
    def __init__(self, env_config: dict):
        self._verbose = False if 'verbose' not in env_config else env_config['verbose']
        self._num_allowed_consecutive_actions = 2 if 'num_allowed_consecutive_actions' not in env_config \
            else env_config['num_allowed_consecutive_actions']

        self._env = gym.make('CartPole-v1')
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Dict({
            'action_mask': gym.spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32),
            'observations': gym.spaces.Dict({
                'cartpole_obs': gym.spaces.Box(
                    self._env.observation_space.low,
                    self._env.observation_space.high,
                    self._env.observation_space.shape,
                    self._env.observation_space.dtype
                )
            })
        })

        self._step = 0
        self._action = -1
        self._consecutive_action_counter = 0

    def _get_state(self, cartpole_obs: np.ndarray) -> dict:
        action_mask = np.float32([1, 1])

        if self._consecutive_action_counter == self._num_allowed_consecutive_actions:
            action_mask[self._action] = 0

        return {
            'action_mask': action_mask,
            'observations': {'cartpole_obs': cartpole_obs}
        }

    def reset(self) -> dict:
        self._step = 0
        self._action = -1
        self._consecutive_action_counter = 0

        cartpole_obs = self._env.reset()
        return self._get_state(cartpole_obs=cartpole_obs)

    def step(self, action: int) -> (dict, float, bool, dict):
        if action == self._action:
            self._consecutive_action_counter += 1
        else:
            self._consecutive_action_counter = 1
            self._action = action

        assert self._consecutive_action_counter <= self._num_allowed_consecutive_actions, f'Illegal Action: {action}'

        cartpole_obs, reward, done, info = self._env.step(action)
        state = self._get_state(cartpole_obs=cartpole_obs)

        if self._verbose:
            self._step += 1

            if done:
                print(f'Episode done at step {self._step}')

        return state, reward, done, info

    def render(self, mode: str or None = None):
        self._env.render()
