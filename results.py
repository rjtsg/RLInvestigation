import matplotlib.pylab as plt
import pandas as pd

data1 = pd.read_csv('DQKSR-10-2000.csv')
data2 = pd.read_csv('DQKSR-100-2000.csv')
data3 = pd.read_csv('DQKSR-1000-2000.csv')

plt.plot(data1['Net worth end game'])
plt.plot(data2['Net worth end game'])
plt.plot(data3['Net worth end game'])
plt.show()