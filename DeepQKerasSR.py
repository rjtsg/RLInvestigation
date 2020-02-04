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