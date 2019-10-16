import numpy as np
from keras.models import Sequential
from keras.layers import Dense, InputLayer
import matplotlib.pylab as plt
class nchain:
    def __init__(self):
        self.state = 0
        self.reward = 0
        self.stop = 100
        self.counter = 0
        self.done = False
    def DO(self,action):
        if action == 1 and self.state <4:
            self.state += 1
            self.reward +=0
        elif action == 1 and self.state == 4:
            self.state = 4
            self.reward += 10
            #print('here')
        elif action == 0:
            self.state = 1
            self.reward +=2
        else:
            print('error')
        if self.counter == self.stop:
            self.done = True
            #print('done')
        self.counter += 1
        return self.state, self.reward, self.done
    def reset(self):
        self.state = 0
        self.reward = 0
        self.stop = 100
        self.counter = 0
        self.done = False
        return self.state

def q_learning_keras(env, num_episodes=1000):
    # create the keras model
    model = Sequential()
    model.add(InputLayer(batch_input_shape=(1, 5)))
    model.add(Dense(10, activation='sigmoid'))
    model.add(Dense(2, activation='linear'))
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
                a = np.random.randint(0, 2)
            else:
                a = np.argmax(model.predict(np.identity(5)[s:s + 1]))
            new_s, r, done = env.DO(a)
            #print(np.identity(5)[new_s:new_s + 1],a)
            target = r + y * np.amax(model.predict(np.identity(5)[new_s:new_s + 1]))
            target_vec = model.predict(np.identity(5)[s:s + 1])[0]
            target_vec[a] = target
            model.fit(np.identity(5)[s:s + 1], target_vec.reshape(-1, 2), epochs=1, verbose=0)
            s = new_s
            r_sum += r
        r_avg_list.append(r_sum / 1000)
    plt.plot(r_avg_list)
    plt.ylabel('Average reward per game')
    plt.xlabel('Number of games')
    plt.show()
    for i in range(5):
        print("State {} - action {}".format(i, model.predict(np.identity(5)[i:i + 1])))

env = nchain()
q_learning_keras(env)

