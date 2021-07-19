import numpy as np

def softmax(x):
    """Transform a 1D array to probability distribution"""
    return np.exp(x-np.max(x)) / np.sum(np.exp(x-np.max(x)))
