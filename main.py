import wandb

import torch
import numpy as np

from ppo_agent import Agent
from wrappers import make_env
from utils import plot_learning_curve

if __name__ == '__main__':

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    game = 'SpaceInvadersNoFrameskip-v4'
    env = make_env(env_name=game, shape=(42, 42, 1), repeat=4)
    train_mode = True
    WANDB = True

    print()
    print('Device: {}'.format(device))
    print('Env: {}'.format(game))
    print('Observation space: {}'.format(env.observation_space))
    print('Number of actions: {}'.format(env.action_space.n))
    print()

    N = 200
    batch_size = 50
    n_epochs = 5
    alpha = 0.0003
    agent = Agent(n_actions=env.action_space.n, batch_size=batch_size,
                  alpha=alpha, n_epochs=n_epochs,
                  input_dims=env.observation_space.shape,
                  game=game,
                  device=device,
                  WANDB=WANDB)

    figure_file = 'plots/{}.png'.format(game)

    best_score = env.reward_range[0]
    score_history = []

    n_games = 300
    max_steps = 1e8
    learn_iters = 0
    avg_score = 0
    n_steps = 0

    settings = {}  # TODO

    if WANDB and train_mode:
        wandb.init(project='curiosity-learning', name=game, config=settings)
        wandb.watch(agent.actor)
        wandb.watch(agent.critic)

    if train_mode:
        while n_steps < max_steps:
            state = env.reset()
            done = False
            score = 0
            h_a = agent.actor.init_hidden(h_dim=256)
            h_c = agent.actor.init_hidden(h_dim=256)
            while not done:
                action, prob, val, h_a_, h_c_ = agent.choose_action(state, h_a, h_c)
                state_, reward, done, info = env.step(action)
                n_steps += 1
                score += reward
                agent.remember(state, action, prob, val, reward, done,
                               h_a.detach().cpu().numpy(), h_c.detach().cpu().numpy())
                if n_steps % N == 0:
                    agent.learn()
                    learn_iters += 1
                state = state_
                h_a = h_a_
                h_c = h_c_
            score_history.append(score)
            avg_score = np.mean(score_history[-100:])

            if avg_score > best_score:
                best_score = avg_score
                agent.save_models()

            print('time_steps', n_steps, 'score %.1f' % score, 'avg score %.1f' % avg_score,
                  'learning_steps', learn_iters)

            if WANDB:
                wandb.log({
                    'Scores/Reward': score,
                    'Scores/AVG Reward': avg_score
                })

        x = [i+1 for i in range(len(score_history))]
        plot_learning_curve(x, score_history, figure_file)

    else:
        agent.actor.load_state_dict(torch.load('./models/actor_{}.pt'.format(game)))
        agent.critic.load_state_dict(torch.load('./models/critic_{}.pt'.format(game)))

        for i in range(n_games):
            state = env.reset()
            done = False
            score = 0
            h_a = agent.actor.init_hidden(h_dim=256)
            h_c = agent.actor.init_hidden(h_dim=256)
            while not done:
                # env.render(mode='human')
                env.render()
                action, _, _, h_a_, h_c_ = agent.choose_action(state, h_a, h_c)
                state_, reward, done, info = env.step(action)
                score += reward
                state = state_
                h_a = h_a_
                h_c = h_c_
            print('episode', i, 'score %.1f' % score)
