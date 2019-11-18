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
        self.reward1 = 0
    

    def DO(self,action):
        def EvalFunc(x):
            # return (x-3)**4-6*(x-3)**2-12*(x-3)+35
            return (np.sin(x/20*2*np.pi)+1)
        remember = EvalFunc(self.state)
        if action == 0: #buy
            self.reward1 = EvalFunc((self.state+1)) - EvalFunc(self.state)
        elif action == 1: #sell
            self.reward1 = EvalFunc(self.state) - EvalFunc((self.state+1))
        elif action == 2:
            self.reward1 = 0
        else:
            print('error')
        # print(self.reward1)
        # self.reward1 = self.reward1*2
        self.state += 1
        if self.state == 21:
            self.done = True
            
            # if self.reward1 >= :
            #     self.reward = 10
            # else:
            #     self.reward = 0

        
        # return self.state, self.reward, self.done, remember
        return self.state, self.reward1, self.done
    def reset(self):
        self.state = 0
        self.reward = 0
        self.stop = 100
        self.counter = 0
        self.done = False
        self.reward1 = 0
        return self.state

def q_learning_keras(env, num_episodes=1000):
    # create the keras model
    model = Sequential()
    model.add(InputLayer(batch_input_shape=(1, 22)))
    model.add(Dense(30, activation='sigmoid'))
    model.add(Dense(3, activation='linear'))
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
                a = np.random.randint(0, 3)
            else:
                a = np.argmax(model.predict(np.identity(22)[s:s + 1]))
            new_s, r, done = env.DO(a)
            #print(np.identity(5)[new_s:new_s + 1],a)
            target = r + y * np.amax(model.predict(np.identity(22)[new_s:new_s + 1]))
            target_vec = model.predict(np.identity(22)[s:s + 1])[0]
            target_vec[a] = target
            print(target_vec)
            print(np.identity(22)[s:s + 1])
            model.fit(np.identity(22)[s:s + 1], target_vec.reshape(-1, 3), epochs=1, verbose=0)
            s = new_s
            r_sum += r
        r_avg_list.append(r_sum)
    plt.plot(r_avg_list)
    plt.ylabel('Average reward per game')
    plt.xlabel('Number of games')
    plt.show()
    for i in range(22):
        print("State {} - action {}".format(i, model.predict(np.identity(22)[i:i + 1])))

env = nchain()
q_learning_keras(env,2)

# testList = []
# done = False
# while not done:
#     new_s, r, done, remember = env.DO(0)
#     testList.append(remember)
# print(testList)
# plt.plot(testList)
# plt.show()


