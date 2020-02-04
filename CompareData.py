# import numpy as np
# from keras.models import Sequential
# from keras.layers import Dense, InputLayer
import matplotlib.pylab as plt
# import pandas as pd 

from Create_train_test_data import CreateTestTrainData
from TradeEnvironments import TradingEnvironment

DataCreation = CreateTestTrainData()
train_data, x = DataCreation.TrainData()
env = TradingEnvironment(train_data)
observation, reward, done = env.reset()


day_list = [observation[0]]
price_list = [observation[1]]
while not done:
    observation, reward, done = env.step(2)
    day_list.append(observation[0])
    price_list.append(observation[1])

plt.plot(x, train_data)
plt.plot(day_list,price_list,'--')
plt.legend(['excel data', 'environment data'])
plt.xlabel('trading days')
plt.ylabel('stock price ($)')
plt.title('AXP 2004')
plt.show()