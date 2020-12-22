import numpy as np

'''
线性整流函数
去除负数，作为神经元的激活函数
'''

def ReLU(x):
    return np.maximum(0, x)