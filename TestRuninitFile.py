import numpy as np
from keras.models import Sequential
from keras.layers import Dense, InputLayer
import matplotlib.pylab as plt
import pandas as pd 
from DefinedStates import moving_average, indicator2

class nchain2:

    def __init__(self):
        self.state = 0
        self.done = False
        self.y = 1
        self.state_method = 0
        self.cash = 5
        self.NetWorth = self.cash
        self.stock = 0
        self.reward2 = 0
    def DO(self,action,state_method):
        def EvalFunc(x):
            # return (np.sin(x/20*2*np.pi)+1)
            return np.cos(2*np.pi*x/40) + np.cos(2*2*np.pi*x/40)+ 3 + np.cos(4*2*np.pi*x/40)
        punish = 0
        NetWorthOld = self.cash + self.stock*EvalFunc(self.state)
        self.y = EvalFunc(self.state)
        self.state_method = state_method(self.state)
        if action == 0 and self.cash > EvalFunc(self.state): #buy
            self.stock += 1
            self.cash -= EvalFunc(self.state)
        elif action == 1 and self.stock > 0: #sell
            self.stock -= 1
            self.cash += EvalFunc(self.state)
        elif action == 2 or self.cash <= EvalFunc(self.state) or self.stock <= 0:
            self.reward1 = 0
            if action != 2:
                punish = -1
        else:
            print('error')
        self.state += 1
        self.reward2 = ((self.cash + self.stock*EvalFunc(self.state)) - NetWorthOld) + punish
        if self.state == 41:
            self.done = True
        
        #print(state_method(self.state))
        return np.array([state_method(self.state), self.y, self.cash, self.stock]) , self.reward2, self.done
        #return np.array([state_method(self.state), 0,self.cash,self.stock]) , self.reward2, self.done
        
    def reset(self,state_method):
        self.state = 0
        self.done = False
        self.reward2 = 0
        self.y = 1 
        self.state_method = 0
        self.cash = 5
        self.stock = 0
        return np.array([state_method(self.state), self.y, self.cash, self.stock])
        #return np.array([state_method(self.state),0,self.cash,self.stock])
    def result(self):
        def EvalFunc(x):
            # return (np.sin(x/20*2*np.pi)+1)
            return np.sin(2*np.pi*x/40) + np.sin(2*2*np.pi*x/40) + 3 + np.sin(4*2*np.pi*x/40)
            
        return self.cash + self.stock*EvalFunc(self.state-1)

def q_learning_keras(env, num_episodes=2000,state_method='moving_average'):
    # create the keras model
    model = Sequential()
    model.add(InputLayer(batch_input_shape=(1, 4)))
    model.add(Dense(50, activation='sigmoid'))
    model.add(Dense(3, activation='linear'))
    model.compile(loss='mse', optimizer='adam', metrics=['mae'])
    # now execute the q learning
    y = 0.95
    eps = 0.5
    decay_factor = 0.999
    r_avg_list = []
    for i in range(num_episodes):
        s = env.reset(state_method)
        eps *= decay_factor
        if i % 100 == 0:
            print("Episode {} of {}".format(i + 1, num_episodes))
        done = False
        r_sum = 0
        while not done:
            if np.random.random() < eps:
                a = np.random.randint(0, 3)
            else:
                a = np.argmax(model.predict(np.array([s])))
            new_s, r, done = env.DO(a,state_method)
            target = r + y * np.amax(model.predict(np.array([new_s])))
            target_vec = model.predict(np.array([s]))[0]
            target_vec[a] = target
            model.fit(np.array([s]), target_vec.reshape(-1, 3), epochs=1, verbose=0)
            s = new_s
            r_sum += r
        r_avg_list.append(r_sum)
    plt.plot(r_avg_list)
    plt.ylabel('Average reward per game')
    plt.xlabel('Number of games')
    plt.show()
    return model

#Select a method!!!
statistical_method = indicator2
#Creating the environment
env = nchain2()
#Start the learning of the model
model = q_learning_keras(env,500,statistical_method)
#Using the model to show how it works:
CASH = []
STOCK = []
NETWORTH = []
REWARD = []
ACTION = []
done = False
s = env.reset(statistical_method)
while not done:
    ACTION.append(np.argmax(model.predict(np.array([s]))))
    s, r, done = env.DO(np.argmax(model.predict(np.array([s]))),statistical_method)
    NETWORTH.append(env.result())
    CASH.append(s[2])
    STOCK.append(s[3])
    REWARD.append(r)
print(env.result())
plt.plot(CASH)
plt.plot(REWARD)
plt.plot(NETWORTH)
plt.plot(ACTION)
plt.legend(['cash','reward','net worth','action'])
plt.show()

