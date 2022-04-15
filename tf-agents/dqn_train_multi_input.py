from wrapper import MultiInputPyEnvironment
from tensorflow_addons.optimizers import Yogi
from tensorflow.keras.optimizers.schedules import PolynomialDecay
from tensorflow.keras.losses import Huber
from tensorflow.keras.activations import gelu
from tf_agents.environments.tf_py_environment import TFPyEnvironment
from tf_agents.networks.q_network import QNetwork
from tf_agents.agents.dqn.dqn_agent import DqnAgent
from tf_agents.replay_buffers.tf_uniform_replay_buffer import TFUniformReplayBuffer
from tf_agents.metrics.tf_metrics import AverageReturnMetric, AverageEpisodeLengthMetric
from tf_agents.drivers.dynamic_step_driver import DynamicStepDriver
from tf_agents.policies import random_tf_policy
from tf_agents.utils.common import Checkpointer
import tensorflow as tf
import matplotlib.pyplot as plt
import timeit


# 1. Creating tf-environments (Train: for training, Eval: For testing)
train_env = TFPyEnvironment(environment=MultiInputPyEnvironment())
eval_env = TFPyEnvironment(environment=MultiInputPyEnvironment())

# 2. Constructing the QNetwork. The target network is automatically created.
conv_layers = None
fc_layers = [128]

preprocessing_layers = {
    'left_observation': tf.keras.layers.Dense(units=128),
    'right_observation': tf.keras.layers.Dense(units=128)
}

online_q_net = QNetwork(
    input_tensor_spec=train_env.observation_spec(),
    action_spec=train_env.action_spec(),
    conv_layer_params=conv_layers,
    fc_layer_params=fc_layers,
    activation_fn=gelu,
    preprocessing_layers=preprocessing_layers,
    preprocessing_combiner=tf.keras.layers.Concatenate(axis=-1)
)

target_q_net = QNetwork(
    input_tensor_spec=train_env.observation_spec(),
    action_spec=train_env.action_spec(),
    conv_layer_params=conv_layers,
    fc_layer_params=fc_layers,
    activation_fn=gelu,
    preprocessing_layers=preprocessing_layers,
    preprocessing_combiner=tf.keras.layers.Concatenate(axis=-1)
)

# 3. Defining the exploration policy.
train_step = tf.Variable(initial_value=0)
total_training_steps = 5000
epsilon_max = 0.5
epsilon_min = 0.001

decay_epsilon_greedy = PolynomialDecay(
    initial_learning_rate=epsilon_max,
    decay_steps=total_training_steps,
    end_learning_rate=epsilon_min
)

# 4. Constructing the DQN Agent.
optimizer = Yogi(learning_rate=0.00025)
n_steps = 1
target_update_steps = 1
tau = 0.001
huber_loss = Huber()
gamma = 0.99

dqn_agent = DqnAgent(
    time_step_spec=train_env.time_step_spec(),
    action_spec=train_env.action_spec(),
    q_network=online_q_net,
    target_q_network=target_q_net,
    optimizer=optimizer,
    epsilon_greedy=lambda: decay_epsilon_greedy(train_step),
    n_step_update=n_steps,
    target_update_tau=tau,
    target_update_period=target_update_steps,
    td_errors_loss_fn=huber_loss,
    gamma=gamma,
    train_step_counter=train_step
)
dqn_agent.initialize()

# 5. Constructing the Replay Memory.
memory_size = 5000

replay_buffer = TFUniformReplayBuffer(
    data_spec=dqn_agent.collect_data_spec,
    batch_size=train_env.batch_size,
    max_length=memory_size
)
dataset = replay_buffer.as_dataset(sample_batch_size=64, num_steps=n_steps+1, num_parallel_calls=1).prefetch(1)

# 6. Collecting initial random experiences.
initial_collect_steps = 500
initial_collect_policy = random_tf_policy.RandomTFPolicy(train_env.time_step_spec(), train_env.action_spec())
replay_buffer_observer = replay_buffer.add_batch

init_driver = DynamicStepDriver(
    env=train_env,
    policy=initial_collect_policy,
    observers=[replay_buffer_observer],
    num_steps=initial_collect_steps
)
print('Collecting initial experiences...\n')
init_driver.run()

# 7. Defining a Policy Saver & Checkpointer (Training Saver).
checkpoint_dir = 'saved_policies/dqn'

train_checkpointer = Checkpointer(
    ckpt_dir=checkpoint_dir,
    max_to_keep=1,
    agent=dqn_agent,
    policy=dqn_agent.policy
)
train_checkpointer.initialize_or_restore()

# 8. Defining a collect driver that interacts with the environment.
train_metrics = [AverageReturnMetric(), AverageEpisodeLengthMetric()]

collect_driver = DynamicStepDriver(
    env=train_env,
    policy=dqn_agent.collect_policy,
    observers=[replay_buffer_observer] + train_metrics,
    num_steps=n_steps+1
)

# 9. Training the agent.
all_train_loss = []
all_metrics = []

policy_state = dqn_agent.collect_policy.get_initial_state(train_env.batch_size)
dataset_iter = iter(dataset)

print('Training the agent...\n')
start = timeit.default_timer()
for step in range(1, total_training_steps+1):
    time_step = None
    current_metrics = []

    time_step, policy_state = collect_driver.run(time_step, policy_state)
    trajectories, buffer_info = next(dataset_iter)

    dqn_agent.train(trajectories)

    for i in range(len(train_metrics)):
        current_metrics.append(train_metrics[i].result().numpy())

    all_metrics.append(current_metrics)

    if step % 500 == 0:
        print('\nIteration: {}'.format(step))
        for i in range(len(train_metrics)):
            print('{}: {}'.format(train_metrics[i].name, train_metrics[i].result().numpy()))

    if step % 500 == 0:
        train_checkpointer.save(step)
        print('Training has been saved.')
end = timeit.default_timer()

print('\nTotal training time = {}'.format(end-start))

# 8. Plotting metrics and results.
average_return = [ metric[0] for metric in all_metrics ]
plt.plot(average_return)
plt.show()

episode_length = [ metric[1] for metric in all_metrics ]
plt.plot(episode_length)
plt.show()

# Terminating client's connection and closing the training environment.
train_env.close()

# 9. Validating the training.
validation_episodes = 10
total_return = 0.0

for _ in range(validation_episodes):
    time_step = eval_env.reset()
    episode_return = 0.0
    step = 0
    print("Step: 0")
    while not time_step.is_last():
        step += 1
        print("---\nStep: {}".format(step))
        action_step = dqn_agent.policy.action(time_step)
        print("Action taken: {}".format(action_step.action))
        time_step = eval_env.step(action_step.action)
        episode_return += time_step.reward
        print("Reward: {} \n".format(episode_return))

    total_return += episode_return

avg_return = total_return / validation_episodes

print('Average return = {}'.format(avg_return))
