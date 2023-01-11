import gym


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

    def reset(self):
        observation = self._env.reset()
        return observation

    def step(self, action: int):
        state, reward, done, info = self._env.step(action)

        if self._verbose:
            self._step += 1

            if done:
                print(f'Episode done at step {self._step}')

        return state, reward, done, info

    def render(self, mode: str or None = None):
        self._env.render()
