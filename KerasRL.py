import numpy as np
import gym

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

from rl.agents import SARSAAgent
from rl.policy import BoltzmannQPolicy


class Sinus:
    def __init__(self):
        self.x = 0
        self.y = 0
        self.done = False
        self.stop = 2*np.pi + np.pi/10
        self.state = None
        self.cash = 10
        self.stock = 0
        self.reward = 0
        
    def step(self,action):
        self.y = np.sin(self.x) + 1
        self.x += np.pi/10
        #self.state += 1
        if action == 0 and self.cash > 2: #buying
            self.cash -= self.y
            self.stock += 1
            # self.reward += (self.cash + self.stock*StockWorth)
        elif action == 1 and self.stock > 0: #selling
            self.stock -= 1
            self.cash += self.y
            # self.reward += (self.cash + self.stock*StockWorth)
        elif action == 2: #Do nothing
            pass
        else:   
            pass
        
        if self.x >= self.stop:
            self.done = True
        # observation = [self.x, self.y,self.cash, self.stock]
        # info = None
        self.reward = self.cash + self.y*self.stock
        self.state = (self.x, self.y,self.cash, self.stock)
        return np.array(self.state), self.reward, self.done, {}
    def reset(self):
        self.state = (0,1,10,0)
        self.done = False
        self.x = 0
        self.y = 0
        self.cash = 10
        self.stock = 0
        return np.array(self.state)

class nchain:
    def __init__(self):
        self.state = 0
        self.reward = 0
        self.stop = 100
        self.counter = 0
        self.done = False
    def step(self,action):
        if action == 1 and self.state <4:
            self.state += 1
            self.reward =0
        elif action == 1 and self.state == 4:
            self.state = 4
            self.reward = 10
            #print('here')
        elif action == 0:
            self.state = 1
            self.reward =2
        else:
            print('error')
        if self.counter == self.stop:
            self.done = True
            #print('done')
        self.counter += 1
        return np.array((self.state,0)), self.reward, self.done, {}
    def reset(self):
        self.state = 0
        self.reward = 0
        self.stop = 100
        self.counter = 0
        self.done = False
        return np.array((self.state,0))

ENV_NAME = 'CartPole-v0'

# Get the environment and extract the number of actions.
env1 = gym.make(ENV_NAME)
env = Sinus()
env2 = nchain()
env.reset()
env1.reset()
nb_actions = 2

# Next, we build a very simple model.
model = Sequential()
model.add(Flatten(input_shape=(1,2) ))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(nb_actions))
model.add(Activation('linear'))
print(model.summary())

# SARSA does not require a memory.
policy = BoltzmannQPolicy()
sarsa = SARSAAgent(model=model, nb_actions=nb_actions, nb_steps_warmup=10, policy=policy)
sarsa.compile(Adam(lr=1e-3), metrics=['mae'])

# Okay, now it's time to learn something! We visualize the training here for show, but this
# slows down training quite a lot. You can always safely abort the training prematurely using
# Ctrl + C.
sarsa.fit(env2, nb_steps=10000, visualize=False, verbose=2)

# After training is done, we save the final weights.
sarsa.save_weights('sarsa_{}_weights.h5f'.format(ENV_NAME), overwrite=True)

# Finally, evaluate our algorithm for 5 episodes.
sarsa.test(env2, nb_episodes=5, visualize=False)



