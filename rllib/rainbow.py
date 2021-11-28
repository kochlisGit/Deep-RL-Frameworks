import ray
from ray.rllib.agents.dqn.dqn import DQNTrainer, DEFAULT_CONFIG
from environment import LunarLander

ray.shutdown()
ray.init(ignore_reinit_error=True, log_to_driver=False)


rainbow_config = DEFAULT_CONFIG
rainbow_config['hiddens'] = [256, 256]
rainbow_config['n_steps'] = 3
rainbow_config['double'] = True
rainbow_config['dueling'] = True
rainbow_config['num_atoms'] = 51
rainbow_config['v_min'] = -200
rainbow_config['v_max'] = 200
rainbow_config['noisy'] = True
rainbow_config['sigma0'] = 0.5
rainbow_config['prioritized_replay'] = True
rainbow_config['prioritized_replay_alpha'] = 0.6
rainbow_config['prioritized_replay_beta'] = 0.4

rainbow_agent = DQNTrainer(rainbow_config, LunarLander)


def train(episodes):
    s = '\nEpisode: {}\nMean Reward: {}\nMin Reward: {}\nMax Reward: {}\nMean Episode Length: {}\n'
    mean_returns = []
    mean_episode_lengths = []

    for i in range(episodes):
        metrics = rainbow_agent.train()
        mean_returns.append(metrics['episode_reward_mean'])
        mean_episode_lengths.append(metrics['episode_len_mean'])

        if i % 10 == 0:
            print(s.format(
                i,
                metrics['episode_reward_mean'],
                metrics['episode_reward_min'],
                metrics['episode_reward_max'],
                metrics['episode_len_mean']
            ))
    return mean_returns, mean_episode_lengths


def evaluate(env, episodes):
    sum_returns = 0

    for i in range(episodes):
        done = False
        state = env.reset()
        env.render()

        while not done:
            action = rainbow_agent.compute_action(state)
            state, reward, done, _ = env.step(action)
            sum_returns += reward
            env.render()
    mean_returns = int(sum_returns / episodes)
    return mean_returns


train(episodes=1000)
evaluate(LunarLander(), episodes=10)
