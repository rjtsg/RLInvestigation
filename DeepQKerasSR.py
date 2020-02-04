import numpy as np
from keras.models import Sequential
from keras.layers import Dense, InputLayer

class DeepQKerasSR:
    def __init__(self):
        self.model = Sequential()
        self.model.add(InputLayer(batch_input_shape=(1, 4)))
        self.model.add(Dense(50, activation='sigmoid'))
        self.model.add(Dense(3, activation='linear'))
        self.model.compile(loss='mse', optimizer='adam', metrics=['mae'])
        self.y = 0.95
        self.eps = 0.5
        self.decay_factor = 0.999

    def Action(self,observation):
        if np.random.random() < self.eps:
            action = np.random.randint(0, 3)
        else:
            action = np.argmax(self.model.predict(np.array([observation])))
        return action

    def Train(self, action,observation, new_observation, reward, done):
        target = reward + self.y * np.amax(self.model.predict(np.array([new_observation])))*(1-int(done))
        target_vec = self.model.predict(np.array([observation]))[0]
        target_vec[action] = target
        self.model.fit(np.array([observation]), target_vec.reshape(-1, 3), epochs=1, verbose=0)
        if done == True:
            self.eps *= self.decay_factor