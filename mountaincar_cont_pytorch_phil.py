import numpy as np 
import gym
from actor_critic_cont_pytorch_phil import Agent
import matplotlib.pylab as plt

if __name__ == '__main__':
    agent = Agent(alpha=0.000005, beta=0.00001, input_dims=[2], gamma=0.99, layer1_size=256, layer2_size=256)

    env = gym.make('MountainCarContinuous-v0')
    score_history = []
    num_episodes = 100
    for i in range(num_episodes):
        done = False
        score = 0
        observation = env.reset()
        while not done:
            action = np.array(agent.choose_action(observation)).reshape((1,))
            observation_ , reward, done, info = env.step(action)
            agent.learn(observation, reward, observation_, done)
            observation = observation_
            score += reward
        score_history.append(score)
        print('episode', i, 'score %.2f' % score)

    plt.plot(score_history)
    plt.show()
