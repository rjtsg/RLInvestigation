import matplotlib.pylab as plt
import pandas as pd
import os
import re
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
                print(GROUPS)

