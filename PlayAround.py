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

mu_model = Sequential()
mu_model.add(InputLayer(batch_input_shape=(1,3)))
mu_model.add(Dense(50,activation='relu'))
# mu_model.add(Dense(30,activation='relu'))
mu_model.add(Dense(5,activation='linear'))
mu_model.compile(loss='mse', optimizer='adam', metrics=['mae'])

episodes = 100
done = False
r_avg_list = []
real_reward = []
AI_reward = []
diff = []
eps = 0.5
decay_factor = 0.999

for M in range(episodes):
    observation = env.reset()
    done = False
    r_sum = 0
    eps *= decay_factor
    if M % 10 == 0:
            print("Episode {} of {}".format(M + 1, episodes))
    while not done:
        # action = np.random.uniform(-2,2)
        # action = mu_model.predict(np.array([observation])) 
        if np.random.random() < eps:
                action = np.random.uniform(-2,2,(1,5))
        else:
                action = mu_model.predict(np.array([observation])) 
        # print(action)
        QPREDICTION = []
        for i in range(len(action[0])):
                actionI = action[0][i]
                q_input = np.append(observation,actionI)
                # print(q_input)
                predicted_reward = Q_model.predict(np.array([q_input]))
                QPREDICTION.append(predicted_reward)
        # for i in range(len(action[0])):
        #         actionI = action[0][i]
        #         new_observation, reward, new_done, _ = env.step(actionI)
        #         Q_model.fit(observation,reward, epochs=1, verbose=0))
        # print(np.argmax(QPREDICTION))
        ACTION = action[0][np.argmax(QPREDICTION)]
        if ACTION > 2 or ACTION < -2:
                punish = -100
        else:
                punish = 0

        # print(ACTION)
        mu_model.fit(np.array([observation]),(np.ones((1,5))*ACTION),epochs=1, verbose=0)
        new_observation, reward, done, _ = env.step(np.array([ACTION]))
        reward  = reward + punish
        if M % 10 == 0:
                env.render()
                print(action,ACTION,reward,np.amax(QPREDICTION))
                # print(np.amax(QP REDICTION)-reward)
        q_input = np.append(observation,ACTION)
        # print(reward)
        action2 = mu_model.predict(np.array([new_observation])) 
        QPREDICTION2 = []
        for i in range(len(action2[0])):
                actionI = action2[0][i]
                q_input = np.append(observation,actionI)
                # print(q_input)
                predicted_reward = Q_model.predict(np.array([q_input]))
                QPREDICTION2.append(predicted_reward)
        ACTION2 = action[0][np.argmax(QPREDICTION2)]
        q_input2 = np.append(new_observation,ACTION2)
        target = reward + 0.95*Q_model.predict(np.array([q_input2]))
        # print(target)
        Q_model.fit(np.array([q_input]),target, epochs=1, verbose=0)
        observation = new_observation
        r_sum += reward
    r_avg_list.append(r_sum)
plt.plot(r_avg_list)
plt.show()
