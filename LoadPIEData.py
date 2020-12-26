from struct import unpack
import gzip
from numpy import uint8, zeros, float32
import scipy.io as sio
import numpy as np


# Read input images and labels(0-9).
# Return it as list of tuples.
#
def LoadPIEData(file, labels_sort):

    data = sio.loadmat(file)

    # Open the images with gzip in read binary mode
    labels = data['gnd']
    images = data['fea']

    # Read the binary data
    # We have to get big endian unsigned int. So we need '>I'

    # Get metadata for images
    (number_of_images, volume) = images.shape
    rows = cols = (int) (np.sqrt(volume))

    # Get metadata for labels
    N = labels.shape[0]

    if number_of_images != N:
        raise Exception('number of labels did not match the number of images')

    # nums = np.zeros(labels_sort, int)
    # for i in range(N):
    #     nums[labels[i, 0]] += 1
    # max = np.max(nums)

    nums = [0] * labels_sort
    for i in range(N):
        nums[labels[i, 0] - 1] += 1
    max_size = max(nums)

    # Get the data
    x = zeros((labels_sort, max_size, rows, cols), dtype=float32)  # Initialize numpy array
    y = zeros((labels_sort, max_size), dtype=uint8)  # Initialize numpy array

    gb_idx = 0
    for l in range(labels_sort):
        for i in range(nums[l]):
            if gb_idx % 1000 == 0:
                print("i: %i" % gb_idx)

            for row in range(rows):
                for col in range(cols):
                    x[l][i][row][col] = images[gb_idx, row * cols + col]

            test = labels[gb_idx, 0]
            y[l][i] = test
            gb_idx += 1

    return (x, y, max_size)