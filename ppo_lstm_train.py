from environment import LunarLanderDiscrete
from tf_agents.environments.tf_py_environment import TFPyEnvironment
from tf_agents.networks.actor_distribution_rnn_network import ActorDistributionRnnNetwork
from tf_agents.networks.value_rnn_network import ValueRnnNetwork
from tensorflow_addons.optimizers import Yogi
from tf_agents.agents.ppo.ppo_agent import PPOAgent
from tf_agents.replay_buffers.tf_uniform_replay_buffer import TFUniformReplayBuffer
from tf_agents.drivers.dynamic_episode_driver import DynamicEpisodeDriver
from tf_agents.utils.common import Checkpointer, function
import tensorflow as tf
import matplotlib.pyplot as plt

# Actor - Critic network parameters.
lstm_layers = (64,)

# PPO Agent parameters.
train_step = tf.Variable(initial_value=0)
optimizer = Yogi(learning_rate=0.00025)

# Replay Buffer parameters.
memory_size = 10000
feed_batch_size = 64
train_sequence_length = 4

# Training parameters.
checkpoint_dir = 'saved_policies/ppo_lstm'
total_train_iterations = 500
train_episodes_per_iteration = 5
eval_episodes_per_iteration = 10

# 1. Creating tf-environments (Train: for training, Eval: For testing)
train_env = TFPyEnvironment(environment=LunarLanderDiscrete())
eval_env = TFPyEnvironment(environment=LunarLanderDiscrete())

# 2. Constructing the Networks: Actor & Value.
actor_network = ActorDistributionRnnNetwork(
    input_tensor_spec=train_env.observation_spec(),
    output_tensor_spec=train_env.action_spec(),
    lstm_size=lstm_layers,
    activation_fn='gelu'
)

value_network = ValueRnnNetwork(
    input_tensor_spec=train_env.observation_spec(),
    activation_fn='gelu'
)

# 3. Constructing the PPO Agent with Clipping, KL-Penalty, GAE, Regularization.
ppo_agent = PPOAgent(
    time_step_spec=train_env.time_step_spec(),
    action_spec=train_env.action_spec(),
    optimizer=optimizer,
    actor_net=actor_network,
    value_net=value_network,
    train_step_counter=train_step
)

ppo_agent.initialize()
ppo_agent.train = function(ppo_agent.train)

# 4. Constructing a Replay Buffer to store episode experiences.
replay_buffer = TFUniformReplayBuffer(
    data_spec=ppo_agent.collect_data_spec,
    batch_size=train_env.batch_size,
    max_length=memory_size
)
replay_buffer_observer = replay_buffer.add_batch

dataset = replay_buffer.as_dataset(
    sample_batch_size=feed_batch_size,
    num_steps=train_sequence_length+1,
    num_parallel_calls=train_sequence_length).prefetch(train_sequence_length)
dataset_iter = iter(dataset)

# 5. Constructing an episode-collection driver that collects experiences from episodes.
collect_driver = DynamicEpisodeDriver(
    train_env,
    ppo_agent.collect_policy,
    observers=[replay_buffer_observer],
    num_episodes=train_episodes_per_iteration)

collect_driver.run = function(collect_driver.run)
ppo_agent.train = function(ppo_agent.train)


# 6. Initializing a Train Checkpointer.
train_checkpointer = Checkpointer(
    ckpt_dir=checkpoint_dir,
    max_to_keep=1,
    agent=ppo_agent,
    policy=ppo_agent.policy
)
train_checkpointer.initialize_or_restore()


# 7. Training the agent.
def compute_avg_return(eval_environment, policy):
    total_return = 0.0

    for _ in range(eval_episodes_per_iteration):
        policy_state = policy.get_initial_state(train_env.batch_size)
        time_step = eval_environment.reset()
        episode_return = 0.0

        while not time_step.is_last():
            action, policy_state, info = policy.action(time_step, policy_state)
            time_step = eval_environment.step(action)
            episode_return += time_step.reward
        total_return += episode_return

    avg_return = total_return / eval_episodes_per_iteration
    return avg_return.numpy()[0]


def train(train_environment, checkpointer):
    average_returns = []

    train_loss = 0
    time_step = None
    policy_state = ppo_agent.collect_policy.get_initial_state(train_environment.batch_size)

    for iteration in range(total_train_iterations + 1):
        time_step, policy_state = collect_driver.run(
            time_step=time_step,
            policy_state=policy_state
        )

        for _ in range(train_episodes_per_iteration):
            experiences, _ = next(dataset_iter)
            train_loss = ppo_agent.train(experiences)
        replay_buffer.clear()

        if iteration % 10 == 0:
            avg_return = compute_avg_return(eval_env, ppo_agent.policy)
            average_returns.append(avg_return)
            print('\nIteration = {} = loss = {}\nAverage Return = {}'.format(iteration, train_loss.loss, avg_return))

            if iteration % 100 == 0:
                checkpointer.save(iteration)
    return average_returns


all_average_returns = train(train_env, train_checkpointer)
train_env.close()
eval_env.close()


average_returns = train(total_train_iterations)
plt.plot(average_returns)
plt.ylabel('Average Return')
plt.xlabel('Step')
plt.show()