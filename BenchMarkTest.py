import matplotlib.pylab as plt
import pandas as pd
import os
from Create_train_test_data import CreateTestTrainData
from TradeEnvironments import TradingEnvironment
from DeepQKerasSR import DeepQKerasSR
from AC_keras_phil import ACKeras

DataCreation = CreateTestTrainData()
train_data, x = DataCreation.TrainData()
env = TradingEnvironment(train_data)
Agent = DeepQKerasSR(24,3)
# Agent = ACKeras(0.00001,0.00005,gamma=0.99,n_actions=3,
#     layer1_size=1024,layer2_size=512,input_dims=4)

num_episodes = 2000
tot_reward_list = []
no_stock_list = []
no_money_list = []
NetWorth_list = []
stop_day_list = []

for i in range(num_episodes):
    observation, reward, done = env.reset()
    reward_score = 0
    while not done:
        action = Agent.Action(observation)
        # print(action)
        new_observation, reward, done, info = env.step(action)
        
        Agent.Train(action, observation, new_observation, reward, done)
        
        reward_score += reward
        observation = new_observation

    tot_reward_list.append(reward_score)
    NetWorth_list.append(info[2])
    no_stock_list.append(info[1])
    no_money_list.append(info[0])
    stop_day_list.append(observation[0])
    print('episode: {0:.0f}, reward score: {1:.2f}'.format(i, reward_score))

dataframe_csv = {'Reward': tot_reward_list,
                 'No cash trade': no_money_list,
                 'No stock trade': no_stock_list,
                 'Net worth end game': NetWorth_list,
                 'Stop day': stop_day_list}


df = pd.DataFrame(dataframe_csv)
os.chdir('Trading_data')
df.to_csv('Trading-DQKSR-24-2000-quiter-second_layer-applied_lr_0.0001-V1.csv', index= False)
os.chdir('..')

plt.plot(tot_reward_list)
plt.xlabel('Number of episodes')
plt.ylabel('Total reward')
plt.title('BenchMarkTest AXP 2004, DQK-SR')
plt.show()

plt.plot(no_money_list)
plt.xlabel('Number of episodes')
plt.ylabel('Times traded with no money')
plt.title('BenchMarkTest AXP 2004, DQK-SR')
plt.show()

plt.plot(no_stock_list)
plt.xlabel('Number of episodes')
plt.ylabel('Times traded with no stock')
plt.title('BenchMarkTest AXP 2004, DQK-SR')
plt.show()

plt.plot(NetWorth_list)
plt.xlabel('Number of episodes')
plt.ylabel('Net worth')
plt.title('BenchMarkTest AXP 2004, DQK-SR')
plt.show()
