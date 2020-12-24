import numpy as np
from scipy import signal
from Softmax import *
from ReLU import *
from Conv import *
from Pool import *


def PIEConv(W1, W5, Wo, X, D, labels_sort, filter_num):
    alpha = 0.01
    beta = 0.95

    momentum1 = np.zeros_like(W1)
    momentum5 = np.zeros_like(W5)
    momentumo = np.zeros_like(Wo)

    # Labels's amount
    N = len(D)

    step_size = 100
    # 0,100,200,...,N
    blist = np.arange(0, N, step_size)

    for batch in range(len(blist)):
        dW1 = np.zeros_like(W1)
        dW5 = np.zeros_like(W5)
        dWo = np.zeros_like(Wo)

        begin = blist[batch]

        # k within step_size
        for k in range(begin, min(begin + step_size, N)):
            # Forward pass = inference
            # Images[k]
            x = X[k, :, :]
            y1 = Conv(x, W1)
            y2 = ReLU(y1)
            y3 = Pool(y2)
            y4 = np.reshape(y3, (-1, 1))
            v5 = np.matmul(W5, y4)
            y5 = ReLU(v5)
            v = np.matmul(Wo, y5)
            y = Softmax(v)

            # one-hot encoding
            d = np.zeros((labels_sort, 1)) # 68: 分类总数量
            d[D[k][0]][0] = 1

            # Backpropagation
            e = d - y
            delta = e

            e5 = np.matmul(Wo.T, delta)  # Hidden(ReLU)
            delta5 = (y5 > 0) * e5

            e4 = np.matmul(W5.T, delta5)  # Pooling layer

            e3 = np.reshape(e4, y3.shape)

            e2 = np.zeros_like(y2)  # pooling
            W3 = np.ones_like(y2) / (2 * 2)
            for c in range(filter_num):
                e2[:, :, c] = np.kron(e3[:, :, c], np.ones((2, 2))) * W3[:, :, c]

            delta2 = (y2 > 0) * e2

            delta1_x = np.zeros_like(W1)
            for c in range(filter_num):
                delta1_x[:, :, c] = signal.convolve2d(x[:, :], np.rot90(delta2[:, :, c], 2), 'valid')

            dW1 = dW1 + delta1_x
            dW5 = dW5 + np.matmul(delta5, y4.T)
            dWo = dWo + np.matmul(delta, y5.T)

        dW1 = dW1 / step_size
        dW5 = dW5 / step_size
        dWo = dWo / step_size

        momentum1 = alpha * dW1 + beta * momentum1
        W1 = W1 + momentum1

        momentum5 = alpha * dW5 + beta * momentum5
        W5 = W5 + momentum5

        momentumo = alpha * dWo + beta * momentumo
        Wo = Wo + momentumo

    return W1, W5, Wo