import matplotlib.pyplot as plt


def evaluate(agent, eval_env, eval_iterations, render):
    total_returns = 0.0

    for _ in range(eval_iterations):
        done = False
        state = eval_env.reset()

        if render:
            eval_env.render()

        while not done:
            action = agent.compute_single_action(observation=state)
            state, reward, done, _ = eval_env.step(action)

            total_returns += reward

            if render:
                eval_env.render()
    return total_returns / eval_iterations


def train(
        agent,
        eval_env,
        train_iterations=50,
        iterations_per_eval=1,
        eval_iterations=5,
        plot_training=True,
        algo_name='Agent'
):
    average_returns = []

    for i in range(train_iterations):
        agent.train()

        if i % iterations_per_eval == 0:
            average_return = evaluate(agent, eval_env, eval_iterations=eval_iterations, render=False)
            average_returns.append(average_return)

            print(f'Iteration: {i}, Average Returns: {average_return}')

    if plot_training:
        plt.plot(average_returns)
        plt.title(f'{algo_name} Training Progress on CartPole')
        plt.xlabel('Iterations')
        plt.ylabel('Average Return')
        plt.show()
    return average_returns
