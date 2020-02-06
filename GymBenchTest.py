import matplotlib.pylab as plt
import pandas as pd
import numpy as np
import gym
import os
from DeepQKerasSR import DeepQKerasSR
from AC_keras_phil import ACKeras

env = gym.make('CartPole-v0')

Agent = DeepQKerasSR(24,2)
# Agent = ACKeras(0.00001,0.00005,gamma=0.99,n_actions=2,
    # layer1_size=1024,layer2_size=512,input_dims=4)

num_episodes = 3000
tot_reward_list = []

for i in range(num_episodes):
    observation = env.reset()
    reward_score = 0
    done = False
    while not done:
        action = Agent.Action(observation)
        new_observation, reward, done, info = env.step(action)
        Agent.Train(action, observation, new_observation, reward, done)
        
        reward_score += reward
        observation = new_observation 

    tot_reward_list.append(reward_score)
    print('episode: {0:.0f}, reward score: {1:.2f}'.format(i, reward_score))

dataframe_csv = {'Reward': tot_reward_list}
df = pd.DataFrame(dataframe_csv)
os.chdir('Cartpole_data')
df.to_csv('Cartpole-DQKSR-24-3000-second_layer-applied_lr3.csv', index= False)
os.chdir('..')

plt.plot(tot_reward_list)
plt.xlabel('Number of episodes')
plt.ylabel('Total reward')
plt.title('BenchMarkTest AXP 2004, DQK-SR')
plt.show()
    