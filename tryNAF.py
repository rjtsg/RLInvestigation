import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input, Concatenate, InputLayer
import matplotlib.pylab as plt
import pandas as pd 
import gym

env = gym.make('Pendulum-v0')
print(env.observation_space.shape)
print(env.action_space.shape[0])
print(env.reset())

observation_space = 3
nb_actions = 1

V_model = Sequential()
V_model.add(InputLayer(batch_input_shape=(1, observation_space)))
V_model.add(Dense(30,activation='relu'))
V_model.add(Dense(1,activation='linear'))
V_model.compile(loss='mse', optimizer='adam', metrics=['mae'])
# print(V_model.summary())

mu_model = Sequential()
mu_model.add(InputLayer(batch_input_shape=(1, observation_space)))
mu_model.add(Dense(30,activation='relu'))
mu_model.add(Dense(nb_actions,activation='linear'))
mu_model.compile(loss='mse', optimizer='adam', metrics=['mae'])
# print(mu_model.summary())

random_action = np.random.uniform(-2,2)
print(random_action)
episodes = 10
iteration = 2
ReplayBuffer = np.array([np.random.rand(3),[[np.random.uniform(-2,2)]],np.random.uniform(-10,10), np.random.rand(3)])
for k in range(100):
  ReplayBuffer = np.vstack((ReplayBuffer,np.array([np.random.rand(3),[[np.random.uniform(-2,2)]],np.random.uniform(-10,10), np.random.rand(3)])))

# print(ReplayBuffer)
# print('************')
# print(ReplayBuffer[2])
r_avg_list = []
for M in range(episodes):
  observation = env.reset()
  done = False
  r_sum = 0
  print('Start episode {}**************************'.format(M))
  while not done:
    action = mu_model.predict(np.array([observation])) + np.random.uniform(-2,2)
    # print(action)
    env.render()
    new_observation, reward, done, _ = env.step(action[0])
    ReplayBuffer = np.vstack((ReplayBuffer,np.array([observation,action,reward, new_observation])))
    ReplayBuffer = np.delete(ReplayBuffer,0,axis=0)
    observation = new_observation
    for I in range(iteration):
      for i in np.random.randint(100,size=10):
        reward_i = ReplayBuffer[i,2]
        # print(reward_i)
        obs_ip1 = ReplayBuffer[i,3]
        # print(obs_ip1)
        y = (ReplayBuffer[i,2]+0.99*V_model.predict(np.array([ReplayBuffer[i,3]])))
        # print(V_model.predict(np.array([ReplayBuffer[i,3]])))
        target = y
        target_vec = V_model.predict(np.array([ReplayBuffer[i,0]]))[0]
        # print(target_vec)
        # target_vec[a] = target
        V_model.fit(np.array([ReplayBuffer[i,0]]), target_vec, epochs=1, verbose=0)
    r_sum += reward
  r_avg_list.append(r_sum)

plt.plot(r_avg_list)
plt.show()

print(len(ReplayBuffer))

  



