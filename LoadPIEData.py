from struct import unpack
import gzip
from numpy import uint8, zeros, float32
import scipy.io as sio
import numpy as np


# Read input images and labels(0-9).
# Return it as list of tuples.
#
def LoadPIEData(file):

    data = sio.loadmat(file)

    # Open the images with gzip in read binary mode
    labels = data['gnd']
    images = data['fea']
    images = np.reshape(images, (-1, 32, 32))
    arr = np.arange(labels.shape[0])
    np.random.shuffle(arr)
    images = images[arr]
    labels = labels[arr]
    return (images, labels)