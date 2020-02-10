import pandas as pd 
import numpy as np

class TradingEnvironment:
    def __init__(self, stock_data, env_type='normal'):
        self.stock_data = stock_data
        self.env_type = env_type
        self.observation = 0
        self.done = False
        self.start = 0
        self.day = self.start
        self.end = len(self.stock_data)
        self.stock_price = self.stock_data[self.start]
        self.cash = 200
        self.stock = 0
        self.no_money = 0
        self.no_stock = 0

        types = ['normal', 'quiter', 'punish'] #all possible environment types
        assert self.env_type in types, 'Environment type does not exist!'

    def step(self, action):
        NetWorthOld = self.cash + self.stock*self.stock_price
        if self.env_type == 'punish':
            punish = 0
        if action == 0 and self.cash > self.stock_price: #buy
            self.stock += 1
            self.cash -= self.stock_price
        elif action == 1 and self.stock > 0: #sell
            self.stock -= 1
            self.cash += self.stock_price
        elif action == 2 or self.cash <= self.stock_price or self.stock <= 0:
            if self.env_type == 'punish':
                punish = -5
            if action == 0:
                self.no_money += 1
                if self.env_type == 'quiter':
                    self.done = True
            elif action == 1:
                self.no_stock += 1
                if self.env_type == 'quiter':
                    self.done = True
                
        #Update day
        self.day += 1
        self.stock_price = self.stock_data[self.day]
        NetWorthNew = self.cash + self.stock*self.stock_price
        if self.env_type == 'punish':
            self.reward = NetWorthNew - NetWorthOld + punish
        else:
            self.reward = NetWorthNew - NetWorthOld

        if self.day == self.end-1:
            self.done = True
        
        self.info = [self.no_money, self.no_stock, NetWorthNew]

        return np.array([self.day, self.stock_price, self.cash, self.stock]), self.reward, self.done, self.info
    
    def reset(self):
        self.start = 0
        self.day = self.start
        self.stock_price = self.stock_data[self.start]
        self.cash = 200
        self.stock = 0
        self.no_money = 0
        self.no_stock = 0
        self.done = False
        self.reward = 0
        self.info = [0, 0, 0]
        return np.array([self.day, self.stock_price, self.cash, self.stock]), self.reward, self.done
    