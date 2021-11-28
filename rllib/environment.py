import gym


class LunarLander(gym.Env):
    def __init__(self, env_config=None):
        self._env_config = env_config
        self._env = gym.make('LunarLander-v2')
        self.action_space = gym.spaces.Discrete(self._env.action_space.n)
        self.observation_space = gym.spaces.Box(
            self._env.observation_space.low,
            self._env.observation_space.high,
            self._env.observation_space.shape,
            self._env.observation_space.dtype
        )

    def reset(self):
        observation = self._env.reset()
        return observation

    def step(self, action):
        state, reward, done, info = self._env.step(action)
        return state, reward, done, info

    def render(self, mode=None):
        self._env.render()
