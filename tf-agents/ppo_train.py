from environment import LunarLanderDiscrete
import tensorflow as tf
from tf_agents.environments.tf_py_environment import TFPyEnvironment
from tf_agents.networks.actor_distribution_network import ActorDistributionNetwork
from tensorflow.keras.activations import gelu
from tensorflow_addons.optimizers import Yogi
from tf_agents.networks.value_network import ValueNetwork
from tf_agents.agents.ppo.ppo_agent import PPOAgent
from tf_agents.replay_buffers.tf_uniform_replay_buffer import TFUniformReplayBuffer
from tf_agents.trajectories import trajectory
from tf_agents.utils.common import Checkpointer, function
import matplotlib.pyplot as plt


# 1. Creating tf-environments (Train: for training, Eval: For testing)
train_env = TFPyEnvironment(environment=LunarLanderDiscrete())
eval_env = TFPyEnvironment(environment=LunarLanderDiscrete())

# 2. Constructing the Networks: Critic & Value.
conv_layers = None
fc_layers = [256, 256]

actor_network = ActorDistributionNetwork(
    input_tensor_spec=train_env.observation_spec(),
    output_tensor_spec=train_env.action_spec(),
    conv_layer_params=conv_layers,
    fc_layer_params=fc_layers,
    activation_fn=gelu
)

value_network = ValueNetwork(
    input_tensor_spec=train_env.observation_spec(),
    conv_layer_params=conv_layers,
    fc_layer_params=fc_layers,
    activation_fn=gelu
)

# 3. Constructing the PPO Agent with Clipping, KL-Penalty, GAE, Regularization.
train_step = tf.Variable(initial_value=0)
optimizer = Yogi(learning_rate=0.00025)
#clipping_epsilon = 0.1
td_lambda = 0.95
gamma = 0.99
num_epochs = 10
#use_gae = True
#entropy_coef = 0.005

ppo_agent = PPOAgent(
    time_step_spec=train_env.time_step_spec(),
    action_spec=train_env.action_spec(),
    optimizer=optimizer,
    actor_net=actor_network,
    value_net=value_network,
#    importance_ratio_clipping=clipping_epsilon,
    lambda_value=td_lambda,
    discount_factor=gamma,
#    entropy_regularization=entropy_coef,
#    use_gae=use_gae,
    num_epochs=num_epochs,
    train_step_counter=train_step
)

ppo_agent.initialize()

# 5. Constructing a Replay Buffer.
memory_size = 2500

replay_buffer = TFUniformReplayBuffer(
    data_spec=ppo_agent.collect_data_spec,
    batch_size=train_env.batch_size,
    max_length=memory_size
)

# 6. Initializing a Checkpointer.
checkpoint_dir = 'saved_policies/ppo'

train_checkpointer = Checkpointer(
    ckpt_dir=checkpoint_dir,
    max_to_keep=1,
    agent=ppo_agent,
    policy=ppo_agent.policy
)
train_checkpointer.initialize_or_restore()


# 7. Defining Metrics
def compute_avg_return(environment, policy, num_episodes=10):
    total_return = 0.0
    for _ in range(num_episodes):

        time_step = environment.reset()
        episode_return = 0.0

        while not time_step.is_last():
            action_step = policy.action(time_step)
            time_step = environment.step(action_step.action)
            episode_return += time_step.reward
        total_return += episode_return

    avg_return = total_return / num_episodes
    return avg_return.numpy()[0]


# 8. Defining a collect driver.
def collect_episode(environment, policy, num_episodes):
    global replay_buffer

    episode_counter = 0
    environment.reset()

    while episode_counter < num_episodes:
        time_step = environment.current_time_step()
        action_step = policy.action(time_step)
        next_time_step = environment.step(action_step.action)
        traj = trajectory.from_transition(time_step, action_step, next_time_step)

        # Add trajectory to the replay buffer
        replay_buffer.add_batch(traj)

        if traj.is_boundary():
            episode_counter += 1


# 9. Training the agent.
num_iterations = 800
episodes_per_iteration = 5
evaL_episodes = 10

# Adding train function to graph.
ppo_agent.train = function(ppo_agent.train)

initial_avg_return = compute_avg_return(eval_env, ppo_agent.policy, evaL_episodes)
average_returns = [initial_avg_return]

for iteration in range(num_iterations+1):
    collect_episode(train_env, ppo_agent.collect_policy, episodes_per_iteration)

    experiences = replay_buffer.gather_all()
    train_loss = ppo_agent.train(experiences)
    replay_buffer.clear()

    if iteration % 10 == 0:
        avg_return = compute_avg_return(eval_env, ppo_agent.policy, evaL_episodes)
        average_returns.append(avg_return)

        print('\nIteration = {}: loss = {}'.format(iteration, train_loss.loss))
        print('Average Return = {}'.format(avg_return))

    if iteration % 100 == 0:
        train_checkpointer.save(iteration)

# 10. Visualizing the results.
plt.plot(average_returns)
plt.ylabel('Average Return')
plt.xlabel('Step')
plt.show()
