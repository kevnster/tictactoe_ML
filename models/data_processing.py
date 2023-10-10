import numpy as np

def load_file(filename):
    data = np.loadtxt(filename)
    X = data[:,:9]
    y = data[:,9:]
    return X, y