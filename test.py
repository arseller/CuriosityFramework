import gym
import torch
from ppo_agent import Agent

if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    batch_size = 5
    n_epochs = 4
    alpha = 0.0003
    agent = Agent(n_actions=env.action_space.n, batch_size=batch_size,
                    alpha=alpha, n_epochs=n_epochs,
                    input_dims=env.observation_space.shape)

    n_games = 300
    game_folder = 'cartpole'

    agent.actor.load_state_dict(torch.load('./models/{}/actor_torch_ppo.pt'.format(game_folder)))
    agent.critic.load_state_dict(torch.load('./models/{}/critic_torch_ppo.pt'.format(game_folder)))

    for i in range(n_games):
        observation = env.reset()
        done = False
        score = 0
        while not done:
            # env.render(mode='human')
            env.render()
            action, _, _ = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            score += reward
            observation = observation_

        print('episode', i, 'score %.1f' % score)
