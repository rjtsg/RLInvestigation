import pandas as pd 
import numpy as np

class TradingEnvironment:
    def __init__(self, stock_data):
        self.stock_data = stock_data
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

    def step(self, action):
        NetWorthOld = self.cash + self.stock*self.stock_price
        if action == 0 and self.cash > self.stock_price: #buy
            self.stock += 1
            self.cash -= self.stock_price
        elif action == 1 and self.stock > 0: #sell
            self.stock -= 1
            self.cash += self.stock_price
        elif action == 2 or self.cash <= self.stock_price or self.stock <= 0:
            self.reward1 = 0
            if action != 2:
                self.no_money += 1
                self.no_stock += 1
        #Update day
        self.day += 1
        self.stock_price = self.stock_data[self.day]
        NetWorthNew = self.cash + self.stock*self.stock_price
        self.reward = NetWorthNew - NetWorthOld
        if self.day == self.end-1:
            self.done = True
        
        return np.array([self.day, self.stock_price, self.cash, self.stock]), self.reward, self.done
    
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
        return np.array([self.day, self.stock_price, self.cash, self.stock]), self.reward, self.done
    