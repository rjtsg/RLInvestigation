import numpy as np
import matplotlib.pylab as plt
import pandas as pd 

class CreateTestTrainData:
    def __init__(self, TrainYear='2004', TestYear='2005'):
        self.df = pd.read_excel('AXPData.xlsx')
        self.test_data = pd.DataFrame(data=None, columns=self.df.columns)
        self.train_data = pd.DataFrame(data=None, columns=self.df.columns)
        
    def TrainData(self,TrainYear='2004'):
        for i in range(len(self.df)):
            datecheck = str(self.df.Date[i])
            if datecheck[0:4] == TrainYear:
                self.train_data.loc[datecheck] = self.df.iloc[i]
        self.train_data = self.train_data.iloc[::-1]
        self.xaxis_train = range(0,len(self.train_data))
        return self.train_data['Close']
        
    
    def TestData(self,TestYear='2005'):
        for i in range(len(self.df)):
            datecheck = str(self.df.Date[i])
            if datecheck[0:4] == TestYear:
                self.test_data.loc[datecheck] = self.df.iloc[i]
        self.test_data = self.test_data.iloc[::-1]
        self.xaxis_test = range(0,len(self.test_data))
        return self.test_data['Close']
        
    
    def PLOTTER(self):
        plt.plot(self.xaxis_train,self.train_data['Close'])
        plt.title('train year')
        plt.show()
        plt.plot(self.xaxis_test,self.test_data['Close'])
        plt.title('test year')
        plt.show()
        


Test = CreateTestTrainData()
training = Test.TrainData('2004')
testing = Test.TestData('2005')
Test.PLOTTER()
print(training)