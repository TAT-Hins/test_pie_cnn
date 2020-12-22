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

    # Read the binary data
    # We have to get big endian unsigned int. So we need '>I'

    # Get metadata for images
    (number_of_images, volume) = images.shape
    rows = cols = (int) (np.sqrt(volume))

    # Get metadata for labels
    N = labels.shape[0]

    if number_of_images != N:
        raise Exception('number of labels did not match the number of images')

    # Get the data
    x = zeros((N, rows, cols), dtype=float32)  # Initialize numpy array
    y = zeros((N, 1), dtype=uint8)  # Initialize numpy array
    for i in range(N):
        if i % 1000 == 0:
            print("i: %i" % i)

        for row in range(rows):
            for col in range(cols):
                idx = row * cols + col
                x[i][row][col] = images[i, idx]

        y[i] = labels[i, 0]

    return (x, y)
    return number_of_images