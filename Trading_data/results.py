import matplotlib.pylab as plt
import pandas as pd
import numpy as np
import os

data = []
data_smooth = []
N = 50
save_file = True
save_name = 'CartPole-DQKSR-10_nodes-5000-multiple-filtered.png'
plt.title('Cartpole DQKSR comparison filtered')
plt.xlabel('Number of episodes')
plt.ylabel('Reward')
LEGEND = ['Run 1', 'Run 2', 'Run 3', 'Run 4', 'Run 5']

data.append(pd.read_csv('Cartpole-DQKSR-10-5000-1.csv'))
data.append(pd.read_csv('Cartpole-DQKSR-10-5000-2.csv'))
data.append(pd.read_csv('Cartpole-DQKSR-10-5000-3.csv'))
data.append(pd.read_csv('Cartpole-DQKSR-10-5000-4.csv'))
data.append(pd.read_csv('Cartpole-DQKSR-10-5000-5.csv'))

for i in range(len(data)):
    data_smooth.append(np.convolve(data[i]['Reward'], np.ones((N,))/N, mode='valid'))
    plt.plot(data_smooth[i])

plt.legend(LEGEND)


if save_file == True:
    os.chdir('CartPole Figures')
    duplicate = os.path.exists(save_name)
    if duplicate == True:
        print('The file name is already in use, it will not be saved')
    else:
        print('Saving figure as:', save_name)
        plt.savefig(save_name)
    os.chdir('..')
plt.show()
