import numpy as np
import os.path
from scipy import signal
from LoadPIEData import *
from Softmax import *
from ReLU import *
from Conv import *
from Pool import *
from PIEConv import *

repeat_train_times = 3
train_percent = 0.8
forceTrain = True

labels_sort = 68
ori_data_path = 'PIE_32x32.mat'
train_data_name = "train_data"
train_data_path = train_data_name + ".npz"

img_h = img_w = 32
krnl_h = krnl_w = 9 # 卷积核 9*9
filter_num = 20
allconnect_volume = 100
relu_a = 0.4
pool_volume = (int)((img_h - krnl_h + 1) / 2) * (int)((img_w - krnl_w + 1) / 2) * filter_num

# Learn
#
# Configurations
Images, Labels, group_max_size = LoadPIEData(ori_data_path, labels_sort)
Images = np.divide(Images, 255) # 255 mean rate of RGB
train_scale = (int) (train_percent * group_max_size)

if os.path.exists(train_data_path):
    train_data = np.load(train_data_path)
    W1 = train_data['w1']
    W5 = train_data['w5']
    Wo = train_data['wo']
    print("already get train data, no need to retrain")
if forceTrain:
    '''
    Deep Learning properties
    '''
    if not os.path.exists(train_data_path):
        W1 = 1e-2 * np.random.randn(krnl_h, krnl_w, filter_num)
        # 卷积层权重初始化：Kaiming/He/MSRA Initialization
        W5 = np.random.uniform(-1, 1, (allconnect_volume, pool_volume)) * np.sqrt(6) / np.sqrt(
            pool_volume * (1 + relu_a * relu_a))
        # 全连接层权重初始化：Kaiming/He/MSRA Initialization
        Wo = np.random.uniform(-1, 1, (labels_sort, allconnect_volume)) * np.sqrt(6) / np.sqrt(
            allconnect_volume * (1 + relu_a * relu_a))

    for _epoch in range(repeat_train_times):
        print(_epoch)
        X = Images[:, 0:train_scale, :, :]
        D = Labels[:, 0:train_scale]
        W1, W5, Wo = PIEConv(W1, W5, Wo, X, D, labels_sort, filter_num)

    np.savez(train_data_name, w1=W1, w5=W5, wo=Wo)
    print("Train Finish, train data save on local")

# Test
#
acc = 0
count = 0
for lb in range(labels_sort):
    X = Images[lb, train_scale:, :, :]
    D = Labels[lb, train_scale:]

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
        if i == D[k]:
            acc = acc + 1
        count += 1

acc = acc / count
print("Accuracy is : ", acc)


