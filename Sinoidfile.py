import numpy as np
from keras.models import Sequential
from keras.layers import Dense, InputLayer
import matplotlib.pylab as plt

class sinoid:
    def __init__(self):
        self.state = 0
        self.reward = 0
        self.stop = 48
        self.counter = 0
        self.cash = 500
        self.stock = 0
        self.t = np.arange(0,50)
        self.done = False
        self.NetWorthOld = self.cash
    def DO(self,action):
        StockWorth = 30*np.sin(4*np.pi*self.t[self.state+1]/self.t[-1]) + 30
        if action == 0 and self.cash > StockWorth: #buying
            self.cash -= StockWorth
            self.stock += 1
            self.reward += (self.cash + self.stock*StockWorth) - self.NetWorthOld
        elif action == 1 and self.stock > 0: #selling
            self.stock -= 1
            self.cash += StockWorth
            self.reward += (self.cash + self.stock*StockWorth) - self.NetWorthOld 
        elif action == 2: #Do nothing
            pass
        else:   
            pass
        self.state +=1
        if self.counter == self.stop:
            self.done = True
            #print('done')
        self.counter += 1
        self.NetWorthOld = self.cash + self.stock*StockWorth
        # print(self.counter)
        # print(action, self.reward, self.cash, self.stock,self.t[self.state])
        return self.state, self.reward, self.done
    def reset(self):
        self.__init__()
        return self.state
    def results(self):
        print('The agents cash at the end of the game is: {}'.format(self.cash))
        print('The agents stock at the end of the game is: {}'.format(self.stock))
        print('The agents NetWorth at the end of the game is: {}'.format(self.NetWorthOld))


def q_learning_keras(env, num_episodes=1000):
    # create the keras model
    states = 50
    actions = 3
    model = Sequential()
    model.add(InputLayer(batch_input_shape=(1, states)))
    model.add(Dense(200, activation='sigmoid'))
    model.add(Dense(200, activation='sigmoid'))
    model.add(Dense(actions, activation='linear'))
    model.compile(loss='mse', optimizer='adam', metrics=['mae'])
    # now execute the q learning
    y = 0.95
    eps = 0.5
    decay_factor = 0.999
    r_avg_list = []
    for i in range(num_episodes):
        s = env.reset()
        eps *= decay_factor
        if i % 10 == 0:
            print("Episode {} of {}".format(i + 1, num_episodes))
        done = False
        r_sum = 0
        while not done:
            if np.random.random() < eps:
                a = np.random.randint(0, actions)
            else:
                a = np.argmax(model.predict(np.identity(states)[s:s + 1]))
            new_s, r, done = env.DO(a)
            #print(np.identity(5)[new_s:new_s + 1],a)
            target = r + y * np.amax(model.predict(np.identity(states)[new_s:new_s + 1]))
            target_vec = model.predict(np.identity(states)[s:s + 1])[0]
            target_vec[a] = target
            model.fit(np.identity(states)[s:s + 1], target_vec.reshape(-1, actions), epochs=1, verbose=0)
            s = new_s
            r_sum += r
        r_avg_list.append(r_sum)
    plt.plot(r_avg_list)
    plt.ylabel('Average reward per game')
    plt.xlabel('Number of games')
    plt.show()
    for i in range(states):
        print("State {} - action {}".format(i, model.predict(np.identity(states)[i:i + 1])))
    return r_sum

t = np.arange(0,50)
plt.plot(t,30*np.sin(4*np.pi*t/t[-1])+30)
plt.show()

env = sinoid()
r = q_learning_keras(env,2000)
env.results()
