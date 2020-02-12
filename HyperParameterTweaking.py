import matplotlib.pylab as plt
import pandas as pd
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
                EnvRex = re.compile(r'\w*')
                GROUPS = EnvRex.findall(self.Environment_type)
                if GROUPS[0] == 'Trading':
                    #Build trading env
                    self.DataCreation = CreateTestTrainData() #Right now it only creates AXP stocks
                    TrainYearRex = re.compile(r'(\d\d\d\d)(\w*)') #Regex to get training year and ticker data
                    self.TrainingYear = TrainYearRex.search(self.Environment_type).group(1)
                    self.StockTicker = TrainYearRex.search(self.Environment_type).group(2)
                    self.train_data, x = self.DataCreation.TrainData(self.TrainingYear)
                    self.env_mode = GROUPS[-2] #gets the type of Trading environment that needs to be created.
                    self.env = TradingEnvironment(self.train_data, self.env_mode)
                    pass
                elif GROUPS[0] == 'CartPole':
                    self.env = gym.make('CartPole-v0')

Run = SetupStudyParameters('Trading-2014AXP-normal', 'DQKSR', [10,10],[0.0001],
                            [0.99],1000)