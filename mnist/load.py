import sys
import os
import gzip
import random
import numpy as np
from time import time
from collections import Counter
from matplotlib import pyplot as plt
from urllib import urlretrieve

from lib.data_utils import shuffle
from lib.config import data_dir


sys.path.append('..')


def download(filename, source='http://yann.lecun.com/exdb/mnist/'):
    print("Downloading %s" % (source + filename))
    urlretrieve(source + filename, os.path.join(data_dir,filename))


def load_mnist(filename):
    if not os.path.exists(os.path.join(data_dir,filename)):
        download(filename)
    # Read the inputs in Yann LeCun's binary format.
    f = gzip.open(os.path.join(data_dir,filename), 'rb')
    return np.frombuffer(f.read(), np.uint8, offset=0)


def mnist():
    loaded = load_mnist('train-images-idx3-ubyte.gz')
    trX = loaded[16:].reshape((60000,28*28)).astype(float)
    trX = trX[:600,:]

    loaded = load_mnist('train-labels-idx1-ubyte.gz')
    trY = loaded[8:].reshape((60000))
    trY = trY[:600]

    loaded = load_mnist('t10k-images-idx3-ubyte.gz')
    teX = loaded[16:].reshape((10000,28*28)).astype(float)
    teX = teX[:100,:]

    loaded = load_mnist('t10k-labels-idx1-ubyte.gz')
    teY = loaded[8:].reshape((10000))
    teY = teY[:100]

    trY = np.asarray(trY)
    teY = np.asarray(teY)

    return trX, teX, trY, teY

def mnist_with_valid_set():
    trX, teX, trY, teY = mnist()

    trX, trY = shuffle(trX, trY)
    vaX = trX[50000:]
    vaY = trY[50000:]
    trX = trX[:50000]
    trY = trY[:50000]

    return trX, vaX, teX, trY, vaY, teY
