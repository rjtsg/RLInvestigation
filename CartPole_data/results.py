import matplotlib.pylab as plt
import pandas as pd
import numpy as np
import os

data = []
data_smooth = []
N = 100
save_file = False
save_name = 'CartPole-DQKSR-5000-filtered.png'
plt.title('Cartpole DQKSR comparison filtered')
plt.xlabel('Number of episodes')
plt.ylabel('Reward')
LEGEND = ['5 nodes','10 nodes', '15 nodes']

data.append(pd.read_csv('Cartpole-DQKSR-5-5000.csv'))
data.append(pd.read_csv('Cartpole-DQKSR-10-5000.csv'))
data.append(pd.read_csv('Cartpole-DQKSR-15-5000.csv'))


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
