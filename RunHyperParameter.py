from HyperParameterTweaking import SetupStudyParameters
from LogAndQuit import OperativeState

#inputs:
Environment = ['CartPole'] #Trading or CartPole
environment_mode = ['normal'] #normal or quiter or punish
Trading_year = ['2014']
Trading_ticker = ['AXP']
Agent_type = ['DDQN']
layer_size = [[1024], [512]]
learning_rate = [[0.00001], [0.00005]] #with DQKSR only the first one matters
discount = [0.99]
num_episodes = [5]

Environment_types = []

OPS = OperativeState()
for i in Environment:
    if i == 'Trading':
        for j in Trading_year:
            Environment_type = i + '-' + j
            for k in Trading_ticker:
                Environment_type = Environment_type + k
                for l in environment_mode:
                    Environment_type = Environment_type + '-' + l
                    Environment_types.append(Environment_type)
    else:
        Environment_types.append(i)
for i in Environment_types:
    for m in Agent_type:
        for ly1 in layer_size[0]:
            for ly2 in layer_size[1]:
                layers = [ly1,ly2]
                for lr1 in learning_rate[0]:
                    lrs = [lr1]
                    if m == 'ACKeras':
                        for lr2 in learning_rate[1]:
                            lrs.append(lr2)
                    for y in discount:
                        y = [y]
                        for eps in num_episodes:
                            OPS.CheckAndQuit()

                            Run = SetupStudyParameters(i,m,layers,
                                                lrs, y, eps)
                            # Run.LoadAgent()
                            Run.StartTraining()
                            # Run.CreateSaveFile()
                            # Run.SaveAgent()

OPS.OperationDone()
