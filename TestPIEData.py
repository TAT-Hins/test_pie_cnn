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
Images = np.divide(Images, 255) # 255 mean rate of RGB

train_scale = (int) (0.8 * Images.shape[0])
test_scale = Images.shape[0] - train_scale

img_h = img_w = 32
krnl_h = krnl_w = 9
filter_num = 20
labels_sort = 68
pool_volume = (int)((img_h - krnl_h + 1) / 2) * (int)((img_w - krnl_w + 1) / 2) * filter_num
allconnect_volume = 100
relu_a = 0.4

# 卷积核 9*9
W1 = 1e-2 * np.random.randn(krnl_h, krnl_w, filter_num)

'''
卷积层权重初始化：Kaiming/He/MSRA Initialization
2880 = 12 * 12 * 20
12 = (32 - 9 + 1) / 2
'''
W5 = np.random.uniform(-1, 1, (allconnect_volume, pool_volume)) * np.sqrt(6) / np.sqrt(pool_volume * (1 + relu_a * relu_a))

'''
全连接层权重初始化：Kaiming/He/MSRA Initialization
'''
Wo = np.random.uniform(-1, 1, (labels_sort, allconnect_volume)) * np.sqrt(6) / np.sqrt(allconnect_volume * (1 + relu_a * relu_a))

X = Images[0:train_scale, :, :]
D = Labels[0:train_scale]

for _epoch in range(3):
    print(_epoch)
    W1, W5, Wo = PIEConv(W1, W5, Wo, X, D, labels_sort, filter_num)

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


