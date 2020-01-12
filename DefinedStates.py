import numpy as np
def EvalFunc(x):
    # return (np.sin(x/20*2*np.pi)+1)
    return np.cos(2*np.pi*x/40) + np.cos(2*2*np.pi*x/40)+ 3 + np.cos(4*2*np.pi*x/40)

def moving_average(state):
    # return (EvalFunc(state)+EvalFunc(state-1)+EvalFunc(state-2))/3
    return EvalFunc(state) - EvalFunc(state-1)

def indicator2(state):
    return state

