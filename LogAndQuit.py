import os

class OperativeState:
    def __init__(self):
        self.name = 'RUN.txt'
        self.run_message = 'RUN'
        self.exist = os.path.exists(self.name)
        if self.exist == True:
            print('This file already exists')
            raise EnvironmentError
            
        txt_file = open(self.name,'w')
        txt_file.write(self.run_message)
        txt_file.close()
        
    def CheckAndQuit(self):
        self.exist = os.path.exists(self.name)
        if self.exist == False:
            print('Cannot find RUN.txt file')
            raise EnvironmentError
        txt_file = open(self.name,'r')
        if txt_file.read() != 'RUN':
            txt_file.close()
            os.remove(self.name)
            raise EnvironmentError



# OPS = OperativeState()
# while True:
#     OPS.CheckAndQuit()
