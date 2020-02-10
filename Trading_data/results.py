import matplotlib.pylab as plt
import pandas as pd
import numpy as np
import os

data = []
data_smooth = []
N = 50
save_file = False
save_name = 'Trading-DQKSR-xx_nodes-2000-lr_0.0001-quiter-filtered.png'
plt.title('Trading DQKSR comparison filtered\n Environment: Quiter\n LR: 0.0001')
plt.xlabel('Number of episodes')
plt.ylabel('Reward')
LEGEND = ['24 nodes', '48 nodes', '100 nodes']

data.append(pd.read_csv('Trading-DQKSR-24-2000-quiter-second_layer-applied_lr_0.0001-V1.csv'))
data.append(pd.read_csv('Trading-DQKSR-48-2000-quiter-second_layer-applied_lr_0.0001-V1.csv'))
data.append(pd.read_csv('Trading-DQKSR-100-2000-quiter-second_layer-applied_lr_0.0001-V1.csv'))



for i in range(len(data)):
    data_smooth.append(np.convolve(data[i]['Reward'], np.ones((N,))/N, mode='valid'))
    plt.plot(data_smooth[i])

plt.legend(LEGEND)


if save_file == True:
    os.chdir('Trading Figures')
    duplicate = os.path.exists(save_name)
    if duplicate == True:
        print('The file name is already in use, it will not be saved')
    else:
        print('Saving figure as:', save_name)
        plt.savefig(save_name)
    os.chdir('..')
plt.show()
