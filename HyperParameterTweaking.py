import matplotlib.pylab as plt
import pandas as pd
import numpy as np
import os
import re
import gym
from Create_train_test_data import CreateTestTrainData
from TradeEnvironments import TradingEnvironment
from DeepQKerasSR import DeepQKerasSR
from AC_keras_phil import ACKeras

class SetupStudyParameters:
    def __init__(self, Environment_type, Agent_type, layer_size, 
                 learning_rate, discount, num_episodes):
        self.Environment_type = Environment_type
        self.Agent_type = Agent_type
        self.layer_size = layer_size
        self.learning_rate = learning_rate
        self.discount = discount
        self.num_episodes = num_episodes
        #Use regex to decompose Environment type into: 
        #Type, Year, Stock ticker, (if needed type of env (quiter))
        ####Building enviroments####
        EnvRex = re.compile(r'\w*')
        GROUPS = EnvRex.findall(self.Environment_type)
        if GROUPS[0] == 'Trading':
            #Build trading env
            self.env_type = GROUPS[0]
            self.DataCreation = CreateTestTrainData() #Right now it only creates AXP stocks
            TrainYearRex = re.compile(r'(\d\d\d\d)(\w*)') #Regex to get training year and ticker data
            self.TrainingYear = TrainYearRex.search(self.Environment_type).group(1)
            self.StockTicker = TrainYearRex.search(self.Environment_type).group(2)
            self.train_data, x = self.DataCreation.TrainData(self.TrainingYear)
            self.env_mode = GROUPS[-2] #gets the type of Trading environment that needs to be created.
            self.env = TradingEnvironment(self.train_data, self.env_mode)
            self.n_actions = 3 #This is not ideal
            self.input_dims = 4 #this is not ideal
            pass
        elif GROUPS[0] == 'CartPole':
            self.env_type = GROUPS[0]
            self.env = gym.make('CartPole-v0')
            self.n_actions = 2
            self.input_dims = 4
        else:
            print('Environment_type not understoond, check name')
            raise NameError 
        #Create and error here... something like above?
        
        ####CREATE AGENT####
        if Agent_type == 'DQKSR':
            self.Agent = DeepQKerasSR(self.learning_rate[0], self.discount[0], self.n_actions, 
                self.layer_size[0], self.layer_size[1], self.input_dims)
        elif Agent_type == 'ACKeras':
            self.Agent = ACKeras(self.learning_rate[0],self.learning_rate[1],self.discount[0],
                self.n_actions, self.layer_size[0],self.layer_size[1], self.input_dims)

    def StartTraining(self):
        self.tot_reward_list = []
        for i in range(self.num_episodes):
            observation, reward, done = self.env.reset()
            reward_score = 0
            while not done:
                action = self.Agent.Action(observation)
                new_observation, reward, done, info = self.env.step(action)
                # print(reward.type())
                self.Agent.Train(action, observation, new_observation, reward, done)
                reward_score += reward
                observation = new_observation
            self.tot_reward_list.append(reward_score)
            print('episode: {0:.0f}, average reward score: {1:.2f}'.format(i, np.mean(self.tot_reward_list[-100:])))
    
    def CreateSaveFile(self):
        #Save the total rewards in a csv file with appropiate naming
        dataframe_csv = {'Reward': self.tot_reward_list}
        save_name = self.Environment_type + '-' + self.Agent_type + '-'
        save_name = save_name + '{}_{}'.format(self.layer_size[0], self.layer_size[1])
        if self.Agent_type == 'DQKSR':
            save_name = save_name + '-{}-'.format(self.learning_rate[0])
        elif self.Agent_type == 'ACKeras':
            save_name = save_name + '-{}_{}-'.format(self.learning_rate[0], self.learning_rate[1])
        save_name = save_name + '{}-'.format(self.discount[0])
        save_name = save_name + '{}.csv'.format(self.num_episodes)
        df = pd.DataFrame(dataframe_csv)
        if self.env_type == 'Trading':
            os.chdir('Trading_data')
            df.to_csv(save_name , index= False)
            os.chdir('..')
        elif self.env_type == 'CartPole':
            os.chdir('CartPole_data')
            df.to_csv(save_name , index= False)
            os.chdir('..')
        print('Saved file: ', save_name)

        
                    
# Run = SetupStudyParameters('Trading-2014AXP-quiter', 'DQKSR', [10,10],[0.0001],
#                             [0.99],10)
# Run.StartTraining()
# Run.CreateSaveFile()