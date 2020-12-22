import numpy as np
from scipy import signal

def Conv(x, W):
    (wrow, wcol, numFilters) = W.shape
    (xrow, xcol)             = x.shape

    # 基于wrow*wcol的神经元，需要计算得到的特征图结果为(xrow-wrow+1)*(xcol-wcol+1)
    
    yrow = xrow - wrow + 1
    ycol = xcol - wcol + 1 

    # 通过numFilters进行多层卷积
    y = np.zeros((yrow, ycol, numFilters))
    
    for k in range(numFilters):
        filter = W[:, :, k] # (wrow*wcol) 卷积核
        filter = np.rot90(np.squeeze(filter), 2) # 去维，反向（旋转90*2）
        y[:, :, k] = signal.convolve2d(x, filter, 'valid')
    
    return y