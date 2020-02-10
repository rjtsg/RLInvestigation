import matplotlib.pylab as plt
import pandas as pd
import numpy as np
import os

data = []
data_smooth = []
N = 25

data.append(pd.read_csv('Cartpole-DQKSR-5-2000.csv'))
data.append(pd.read_csv('Cartpole-DQKSR-10-2000.csv'))
data.append(pd.read_csv('Cartpole-DQKSR-15-2000.csv'))
data.append(pd.read_csv('Cartpole-DQKSR-25-2000.csv'))
data.append(pd.read_csv('Cartpole-DQKSR-50-2000.csv'))
data.append(pd.read_csv('Cartpole-DQKSR-100-2000.csv'))

for i in range(len(data)):
    data_smooth.append(np.convolve(data[i]['Reward'], np.ones((N,))/N, mode='valid'))
    # plt.plot(data[i]['Reward'])
    plt.plot(data_smooth[i])




# plt.legend(['no filter', 'filter'])
plt.legend(['5 nodes','10 nodes', '15 nodes', '25 nodes',
            '50 nodes', '100 nodes'])

plt.title('Cartpole DQKSR comparison filtered')
plt.xlabel('Number of episodes')
plt.ylabel('Reward')
# os.chdir('CartPole Figures')
# plt.savefig('CartPole-DQKSR-2000-filtered.png')
# os.chdir('..')
plt.show()
