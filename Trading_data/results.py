import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import numpy as np
import os

data = []
data_smooth = []
N = 10
save_file = False
save_name = 'Trading-ACKeras-1024_512_nodes-100-lr_xx_xx-normal-filtered-V2.png'
plt.title('Trading ACKeras comparison filtered\n Environment: Normal\n Nodes: 1024-512')
plt.xlabel('Number of episodes')
plt.ylabel('Reward')
LEGEND = ['lr 0.001-0.005', 'lr 0.0001-0.0005']

data.append(pd.read_csv('Trading-2014AXP-normal-DQKSR-100_100-0.0001-0.99-500.csv'))
data.append(pd.read_csv('Trading-2014AXP-normal-DQKSR-100_50-0.0001-0.99-500.csv'))



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

## make surface plots
layer_sizes = np.array([5,10,25,50,100])
data2 = []
for i in layer_sizes:
    for j in layer_sizes:
        load_name = 'Trading-2014AXP-normal-DQKSR-{}_{}-0.0001-0.99-500.csv'.format(i,j)
        data2.append(pd.read_csv(load_name))
# print(data2)
last_100_average = []
for k in range(len(data2)):
    last_100_average.append(np.mean(data2[k]['Reward'][-100:]))
matrix_100_average = np.reshape(last_100_average,(5,5))
print(matrix_100_average)
XX, YY = np.meshgrid(layer_sizes,layer_sizes)
fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(XX,YY,matrix_100_average, cmap = 'viridis')
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.xlabel('Fisrt layer nodes')
plt.ylabel('Second layer nodes')
ax.set_zlabel('Average reward of last 100 values')
plt.title('DQKSR\nEnv: normal, eps: 500\ndiscout: 0.99, lr: 0.0001')
plt.show()