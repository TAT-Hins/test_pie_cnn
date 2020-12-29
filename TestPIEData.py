import numpy as np
import os.path
import math
from scipy import signal
from LoadPIEData import *
from Softmax import *
from ReLU import *
from Conv import *
from Pool import *
from PIEConv import *

repeat_train_times = 10
train_percent = 0.8
# 卷积核 9*9
krnl_h = krnl_w = 9
filter_num = 20
labels_times = 5
relu_a = 0.4
forceTrain = True
shouldAppendTrainData = False

ori_data_path = 'PIE_32x32.mat'
train_data_name = "train_data_" +
train_data_path = train_data_name + ".npz"

# Learn
#
# Configurations
Images, Labels = LoadPIEData(ori_data_path)
Images = np.divide(Images, 255) # 255 mean rate of RGB
data_size = Images.shape[0]

train_size = (int)(data_size * train_percent)
# make train_size % 100 = 0
if (train_size != 0):
    train_size = 100 * (int)(train_size / math.pow(100, math.floor(math.log(train_size, 100))))

img_h = Images.shape[1]
img_w = Images.shape[2]
labels_sort = np.max(Labels)
Labels = Labels - 1
allconnect_sample_size = labels_sort * labels_times

pool_volume = (int)((img_h - krnl_h + 1) / 2) * (int)((img_w - krnl_w + 1) / 2) * filter_num

if os.path.exists(train_data_path):
    print("loading saved train data...")
    train_data = np.load(train_data_path)
    W1 = train_data['w1']
    W5 = train_data['w5']
    Wo = train_data['wo']
if forceTrain:
    if not (shouldAppendTrainData and os.path.exists(train_data_path)):
        print("reset configurations")
        '''
        Deep Learning properties
        '''
        W1 = 1e-2 * np.random.randn(krnl_h, krnl_w, filter_num)
        # 卷积层权重初始化：Kaiming/He/MSRA Initialization
        W5 = np.random.uniform(-1, 1, (allconnect_sample_size, pool_volume)) * np.sqrt(6) / np.sqrt(
            pool_volume * (1 + relu_a * relu_a)
        )
        # 全连接层权重初始化：Kaiming/He/MSRA Initialization
        Wo = np.random.uniform(-1, 1, (labels_sort, allconnect_sample_size)) * np.sqrt(6) / np.sqrt(
            # allconnect_sample_size * (1 + relu_a * relu_a)
            labels_sort + allconnect_sample_size
        )

    for _epoch in range(repeat_train_times):
        print(_epoch)
        X = Images[0:train_size, :, :]
        D = Labels[0:train_size]
        W1, W5, Wo = PIEConv(W1, W5, Wo, X, D, labels_sort, filter_num)

    np.savez(train_data_name, w1=W1, w5=W5, wo=Wo)
    print("Train Finish, train data save on local")
else:
    print("No need to re-train")

# Test
#
acc = 0
X = Images[train_size:, :, :]
D = Labels[train_size:]
N = X.shape[0]
for k in range(N):
    if D[k] == 0:
        continue

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


