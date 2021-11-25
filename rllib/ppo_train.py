import ray
from ray.rllib.agents.ppo.ppo import PPOTrainer
from ray.rllib.agents.ppo.ppo import DEFAULT_CONFIG
import matplotlib.pyplot as plt

print('Loading config...')

config = DEFAULT_CONFIG.copy()
config['model']['fcnet_activation'] = 'relu'
config['lambda'] = 0.95
config['entropy_coeff'] = 0.01
config['batch_mode'] = 'complete_episodes'

print('Initializing ray...')

ray.shutdown()
ray.init(ignore_reinit_error=True)

print('Training...')

appo_agent = PPOTrainer(config, 'LunarLander-v2')

average_returns = []
total_iterations = 600
checkpoint_dir = 'saved_policy/ppo'

s = '\nEpisode: {}\nMean Reward: {}\nMin Reward: {}\nMax Reward: {}\nMean Episode Length: {}\n'

best_average_return = -2000

for i in range(total_iterations):
    metrics = appo_agent.train()
    average_return = metrics['episode_reward_mean']
    average_returns.append(average_return)

    if i % 10 == 0:
        print(s.format(
            i,
            metrics['episode_reward_mean'],
            metrics['episode_reward_min'],
            metrics['episode_reward_max'],
            metrics['episode_len_mean']
        ))

    if average_return > best_average_return:
        best_average_return = average_return
        appo_agent.save(checkpoint_dir)

# Plotting average returns.
plt.plot(average_returns)
plt.ylabel('Average Returns')
plt.xlabel('Iterations')
plt.show()
