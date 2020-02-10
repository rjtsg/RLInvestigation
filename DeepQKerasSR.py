import numpy as np
from keras.models import Sequential
from keras.layers import Dense, InputLayer
from keras.optimizers import Adam

class DeepQKerasSR:
    def __init__(self, alpha, gamma=0.99, n_actions=3, 
                layer1_size=100, layer2_size=100, input_dims=4):
        self.model = Sequential()
        self.model.add(InputLayer(batch_input_shape=(1, input_dims)))
        self.model.add(Dense(layer1_size, activation='sigmoid'))
        if layer2_size != 0:
            self.model.add(Dense(layer2_size, activation='sigmoid')) #second layer
        self.model.add(Dense(n_actions, activation='linear'))
        # self.model.compile(loss='mse', optimizer='adam', metrics=['mae'])
        self.model.compile(loss='mse', optimizer=Adam(lr=alpha), metrics=['mae'])
        self.y = gamma
        self.eps = 0.5
        self.decay_factor = 0.999
        self.n_actions = n_actions

    def Action(self,observation):
        if np.random.random() < self.eps:
            action = np.random.randint(0, self.n_actions)
        else:
            action = np.argmax(self.model.predict(np.array([observation])))
        return action

    def Train(self, action, observation, new_observation, reward, done):
        target = reward + self.y * np.amax(self.model.predict(np.array([new_observation])))*(1-int(done))
        target_vec = self.model.predict(np.array([observation]))[0]
        target_vec[action] = target
        self.model.fit(np.array([observation]), target_vec.reshape(-1, self.n_actions), epochs=1, verbose=0)
        if done == True:
            self.eps *= self.decay_factor