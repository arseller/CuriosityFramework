import gym
import torch
import numpy as np

from ppo_agent import Agent
from utils import plot_learning_curve

if __name__ == '__main__':

    game = 'CartPole-v1'
    env = gym.make(game)
    train_mode = False

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if device == torch.device('cpu'):
        print('if')
        print(device)
        device_name = 'none'
    else:
        print('else')
        print(device)
        device_name = torch.cuda.get_device_name(0)

    print('Device: {}'.format(device))
    print('Device name: {}'.format(device_name))
    print('Env: {}'.format(game))
    print('Observation space: {}'.format(env.observation_space))
    print('Number of actions: {}'.format(env.action_space.n))

    N = 20
    batch_size = 5
    n_epochs = 4
    alpha = 0.0003
    agent = Agent(n_actions=env.action_space.n, batch_size=batch_size,
                  alpha=alpha, n_epochs=n_epochs,
                  input_dims=env.observation_space.shape,
                  game=game)

    n_games = 300

    figure_file = 'plots/{}.png'.format(game)

    best_score = env.reward_range[0]
    score_history = []

    learn_iters = 0
    avg_score = 0
    n_steps = 0

    if train_mode:
        for i in range(n_games):
            state = env.reset()
            done = False
            score = 0
            while not done:
                action, prob, val = agent.choose_action(state)
                state_, reward, done, info = env.step(action)
                n_steps += 1
                score += reward
                agent.remember(state, action, prob, val, reward, done)
                if n_steps % N == 0:
                    agent.learn()
                    learn_iters += 1
                state = state_
            score_history.append(score)
            avg_score = np.mean(score_history[-100:])

            if avg_score > best_score:
                best_score = avg_score
                agent.save_models()

            print('episode', i, 'score %.1f' % score, 'avg score %.1f' % avg_score,
                  'time_steps', n_steps, 'learning_steps', learn_iters)
        x = [i+1 for i in range(len(score_history))]
        plot_learning_curve(x, score_history, figure_file)
    else:
        agent.actor.load_state_dict(torch.load('./models/actor_{}.pt'.format(game), map_location=device))
        agent.critic.load_state_dict(torch.load('./models/critic_{}.pt'.format(game), map_location=device))

        for i in range(n_games):
            state = env.reset()
            done = False
            score = 0
            while not done:
                # env.render(mode='human')
                env.render()
                action, _, _ = agent.choose_action(state)
                state_, reward, done, info = env.step(action)
                score += reward
                state = state_

            print('episode', i, 'score %.1f' % score)
