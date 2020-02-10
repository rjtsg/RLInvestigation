import matplotlib.pylab as plt
import pandas as pd
import numpy as np
import os

data = []
data_smooth = []
N = 5
save_file = True
save_name = 'Trading-ACKeras-1024_512_nodes-100-lr_xx_xx-normal-filtered-V2.png'
plt.title('Trading ACKeras comparison filtered\n Environment: Normal\n Nodes: 1024-512')
plt.xlabel('Number of episodes')
plt.ylabel('Reward')
LEGEND = ['lr 0.001-0.005', 'lr 0.0001-0.0005']

data.append(pd.read_csv('Trading-ACKeras-1024_512-100-lr-0.001_0.005.csv'))
data.append(pd.read_csv('Trading-ACKeras-1024_512-100-lr-0.0001_0.0005.csv'))



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
