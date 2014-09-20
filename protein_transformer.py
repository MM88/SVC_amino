__author__ = 'miky'
import pickle
import numpy as np

def transformer(pickle_path,X):

    pkl_file = open(pickle_path, 'rb')
    model = pickle.load(pkl_file)
    W = model.get_weights(borrow=True)
    X_new = np.dot(X,W)
    return X_new