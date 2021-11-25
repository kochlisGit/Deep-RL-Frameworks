import ray
from ray.rllib.agents.ppo.appo import APPOTrainer
from ray.rllib.agents.ppo.appo import DEFAULT_CONFIG
import matplotlib.pyplot as plt

print('Loading config...')

config = DEFAULT_CONFIG.copy()
config['model']['fcnet_activation'] = 'relu'
config['num_sgd_iter'] = 30


print('Initializing ray...')

ray.shutdown()
ray.init(ignore_reinit_error=True)

print('Training...')

appo_agent = APPOTrainer(config, 'LunarLander-v2')

average_returns = []
total_iterations = 600
checkpoint_dir = 'saved_policy/appo'

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
