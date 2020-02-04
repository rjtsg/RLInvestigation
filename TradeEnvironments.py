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

    def step(self, action):
        
