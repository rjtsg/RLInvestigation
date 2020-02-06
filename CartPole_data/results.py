import matplotlib.pylab as plt
import pandas as pd

data1 = pd.read_csv('Cartpole-DQKSR-5-2000.csv')
data2 = pd.read_csv('Cartpole-DQKSR-10-2000.csv')
data3 = pd.read_csv('Cartpole-DQKSR-15-2000.csv')

plt.plot(data1['Reward'])
plt.plot(data2['Reward'])
plt.plot(data3['Reward'])
plt.show()
