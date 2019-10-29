import numpy as np
from keras.models import Sequential
from keras.layers import Dense, InputLayer
import matplotlib.pylab as plt

class Sinus:
    def __init__(self):
        self.x = 0
        self.y = 0
        self.done = False
        self.stop = 2*np.pi + np.pi/10
    def DO(self):
        self.y = np.sin(self.x)
        self.x += np.pi/10
        print(self.x)
        if self.x >= self.stop:
            self.done = True
        return self.x, self.y, self.done
    def reset(self):
        self.__init__()

done = False
env = Sinus()
X = []
Y = []
while not done:
    x,y,done = env.DO()
    X.append(x)
    Y.append(y)

plt.plot(X,Y)
plt.show()