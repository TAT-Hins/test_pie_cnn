import numpy as np
from scipy import signal
from LoadPIEData import *
from Softmax import *
from ReLU import *
from Conv import *
from Pool import *
from PIEConv import *

# Learn
#
Images, Labels = LoadPIEData('PIE_32x32.mat')
Images = Images[0:10000]
Labels = Labels[0:10000]
Images = np.divide(Images, 255) # 255 mean rate of RGB

train_scale = (int) (0.8 * Images.shape[0])
test_scale = Images.shape[0] - train_scale

# 卷积核 9*9
W1 = 1e-2 * np.random.randn(9, 9, 20)

'''
卷积层权重初始化：Kaiming/He/MSRA Initialization
2880 = 12 * 12 * 20
12 = (32 - 9 + 1) / 2
'''
W5 = np.random.uniform(-1, 1, (100, 2880)) * np.sqrt(6) / np.sqrt(520 + 2880)

'''
全连接层权重初始化：Kaiming/He/MSRA Initialization
'''
Wo = np.random.uniform(-1, 1, (68, 100)) * np.sqrt(6) / np.sqrt(68 + 100)

X = Images[0:train_scale, :, :]
D = Labels[0:train_scale]

for _epoch in range(3):
    print(_epoch)
    W1, W5, Wo = PIEConv(W1, W5, Wo, X, D)

print("Train Finish")

# Test
#
X = Images[train_scale + 1:Images.shape[0], :, :]
D = Labels[train_scale + 1:Images.shape[0]]

acc = 0
N = len(D)
for k in range(N):
    x = X[k, :, :]

    y1 = Conv(x, W1)
    y2 = ReLU(y1)
    y3 = Pool(y2)
    y4 = np.reshape(y3, (-1, 1))
    v5 = np.matmul(W5, y4)
    y5 = ReLU(v5)
    v = np.matmul(Wo, y5)
    y = Softmax(v)

    i = np.argmax(y)
    if i == D[k][0]:
        acc = acc + 1

acc = acc / N
print("Accuracy is : ", acc)


