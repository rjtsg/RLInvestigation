import matplotlib.pylab as plt
from Create_train_test_data import CreateTestTrainData
from TradeEnvironments import TradingEnvironment
from DeepQKerasSR import DeepQKerasSR

DataCreation = CreateTestTrainData()
train_data, x = DataCreation.TrainData()
env = TradingEnvironment(train_data)
Agent = DeepQKerasSR()

num_episodes = 1000
tot_reward_list = []

for i in range(num_episodes):
    observation, reward, done = env.reset()
    reward_score = 0
    while not done:
        action = Agent.Action(observation)
        
        new_observation, reward, done = env.step(action)
        
        Agent.Train(action, observation, new_observation, reward, done)
        
        reward_score += reward
        observation = new_observation
    tot_reward_list.append(reward_score)
    print('episode: {0:.0f}, reward score: {1:.2f}'.format(i, reward_score))

plt.plot(tot_reward_list)
plt.xlabel('Number of episodes')
plt.ylabel('Total reward')
plt.title('BenchMarkTest AXP 2004, DQK-SR')
plt.show()

