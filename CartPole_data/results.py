import matplotlib.pylab as plt
import pandas as pd
import os

data1 = pd.read_csv('Cartpole-DQKSR-5-2000.csv')
data2 = pd.read_csv('Cartpole-DQKSR-10-2000.csv')
data3 = pd.read_csv('Cartpole-DQKSR-15-2000.csv')
data4 = pd.read_csv('Cartpole-DQKSR-25-2000.csv')
data5 = pd.read_csv('Cartpole-DQKSR-50-2000.csv')
data6 = pd.read_csv('Cartpole-DQKSR-100-2000.csv')


plt.plot(data1['Reward'])
plt.plot(data2['Reward'])
plt.plot(data3['Reward'])
plt.plot(data4['Reward'])
plt.plot(data5['Reward'])
plt.plot(data6['Reward'])

plt.legend(['5 nodes','10 nodes', '15 nodes', '25 nodes',
            '50 nodes', '100 nodes'])

plt.title('Cartpole DQKSR comparison')
plt.xlabel('Number of episodes')
plt.ylabel('Reward')
os.chdir('CartPole Figures')
plt.savefig('CartPole-DQKSR-2000-2.png')
os.chdir('..')
plt.show()
