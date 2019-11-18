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
        self.state = 0
        
    def DO(self,action):
        self.y = np.sin(self.x)
        self.x += np.pi/10
        self.state += 1
        reward = 1 - np.abs(self.y - action)/self.y
        if self.x >= self.stop:
            self.done = True
        return reward, self.x, self.y, self.done, self.state
    def reset(self):
        self.__init__()

def q_learning_keras(env, num_episodes=100):
    # create the keras model
    states = 21
    actions = 1
    model = Sequential()
    model.add(InputLayer(batch_input_shape=(1, states)))
    model.add(Dense(40, activation='sigmoid'))
    model.add(Dense(actions, activation='linear'))
    model.compile(loss='mse', optimizer='adam', metrics=['mae'])
    # now execute the q learning
    gamma = 0.95
    eps = 0.5
    decay_factor = 0.999
    r_avg_list = []
    for i in range(num_episodes):
        s = env.reset()
        s = 0
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
            r, x,y,done, new_s = env.DO(a)
            #print(np.identity(5)[new_s:new_s + 1],a)
            target = r + gamma * np.amax(model.predict(np.identity(states)[new_s:new_s + 1])) - np.amax(model.predict(np.identity(states)[s:s+1]))
            target_vec = model.predict(np.identity(states)[s:s + 1])[0]
            target_vec[a] = target
            # print(target_vec.reshape(-1,actions))
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
    return r_sum, model


done = False
env = Sinus()
X = []
Y = []
while not done:
    reward, x,y,done, state = env.DO(1)
    X.append(x)
    Y.append(y)

plt.plot(X,Y)
plt.show()
env = Sinus()
q_learning_keras(env,100)