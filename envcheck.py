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

Q_model = Sequential()
Q_model.add(InputLayer(batch_input_shape=(1,4)))
Q_model.add(Dense(50,activation='relu'))
Q_model.add(Dense(1,activation='linear'))
Q_model.compile(loss='mse', optimizer='adam', metrics=['mae'])

V_model = Sequential()
V_model.add(InputLayer(batch_input_shape=(1,4)))
V_model.add(Dense(50,activation='relu'))
V_model.add(Dense(3,activation='linear'))
V_model.compile(loss='mse', optimizer='adam', metrics=['mae'])

action_model = Sequential()
action_model.add(InputLayer(batch_input_shape=(1,3)))
action_model.add(Dense(50,activation='relu'))
# action_model.add(Dense(50,activation='sigmoid'))
action_model.add(Dense(5,activation='linear'))
action_model.compile(loss='mse', optimizer='adam', metrics=['mae'])

done = False
r = []
Qr = []
v1 = []
vm1 = [] 
v2 = []
vm2 = []
v3 = []
vm3 = []

r_sum = 0
episodes = 1000
eps = 0.5
decay_factor = 0.9999
a1_sum = []
a2_sum = []
a3_sum =[]
a4_sum = []
a5_sum = []
a_sum = []

for i in range(episodes):
    observation = env.reset()
    done = False
    r_sum = 0
    Qr_sum = 0
    next_Q_reward = np.ones((5,5))
    
    while not done:
        # action = np.random.uniform(-2,2)
        if np.random.random() < eps:
            action = np.random.uniform(-2, 2,(1,5))
        else:
            action = action_model.predict(np.array([observation]))
        eps *= decay_factor 
        a1_sum.append(action[0][0])
        a2_sum.append(action[0][1])
        a3_sum.append(action[0][2])
        a4_sum.append(action[0][3])
        a5_sum.append(action[0][4])
        Q_reward = []
        for j in range(len(action[0])):
            observation1 = np.append(observation,action[0][j])
            Q_reward.append(Q_model.predict(np.array([observation1])))
            next_obs_V = V_model.predict(np.array([observation1]))
            # print(next_obs_V)
            # print('+------+')
            action2 = action_model.predict(next_obs_V)
            for k in range(len(action2[0])):
                observation2 = np.append(next_obs_V,action2[0][k])
                next_Q_reward[j,k] = Q_reward[j] + 0.95 * Q_model.predict(np.array([observation2]))
                # print(j)


        action = action[0][np.unravel_index(np.argmax(next_Q_reward, axis=None), next_Q_reward.shape)[0]]
        if action > 2 or action < -2:
            punish = -100
        else:
            punish = 0
        print(action)
        observation = np.append(observation,action)

        a_sum.append(action)
        if i % 10 == 0:
            env.render(action)


        # V_reward = V_model.predict(np.array([observation]))
        # reward = reward +0.95*next_reward
        # action2 = np.random.uniform(-2,2)
        # observation2 = np.append(new_observation,action2)
        # future_reward = reward + 0.95*Q_model.predict(np.array([observation2]))[0]
        new_observation, reward, done, _ = env.step(np.array([action]))
        print(observation,reward)
        print('+-----+')
        reward = reward + punish
        Q_model.fit(np.array([observation]),np.array([reward]), epochs=1, verbose=0)
        print('observation')
        print(np.ones((1,5))*action)
        action_model.fit(np.array([observation[0:3]]),np.ones((1,5))*action, epochs=1, verbose=0)
        V_model.fit(np.array([observation]),np.array([new_observation]), epochs=1, verbose=0)
        observation = new_observation
        # env.render()
        r_sum +=reward
        Qr_sum += np.amax(Q_reward[0])
        # v1_sum += observation[0]
        # vm1_sum += V_reward[0][0]
        # v2_sum += observation[1]
        # vm2_sum += V_reward[0][1]
        # v3_sum += observation[2]
        # vm3_sum += V_reward[0][2]

    r.append(r_sum)
    Qr.append(Qr_sum)
    # v1.append(v1_sum)
    # vm1.append(vm1_sum)
    # v2.append(v2_sum)
    # vm2.append(vm2_sum)
    # v3.append(v3_sum)
    # vm3.append(vm3_sum)
    # print(Q_reward)
plt.plot(r)
plt.plot(Qr)
plt.show()

plt.plot(a_sum)
plt.plot(a1_sum)
plt.plot(a2_sum)
plt.plot(a3_sum)
plt.plot(a4_sum)
plt.plot(a5_sum)
plt.legend(['chosen','1','2','3','4','5'])
plt.show()

# plt.plot(v1)
# plt.plot(vm1)
# plt.show()

# plt.plot(v2)
# plt.plot(vm2)
# plt.show()

# plt.plot(v3)
# plt.plot(vm3)
# plt.show()

