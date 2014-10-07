__author__ = 'miky'

from pylearn2.config import yaml_parse
import pattern_generator
import pickle
import numpy as np
from sklearn import svm
from sklearn.cross_validation import train_test_split


def train_dae():

    f = open('./dae.pkl','wb')
    yaml_file = open ("./denoisingAutoencoder.yaml")
    train = yaml_parse.load(yaml_file)
    train.main_loop()
    f.close()
    yaml_file.close()


def train_rbm():

    f = open('./rbm_11.pkl','wb')
    yaml_file = open ("./rbm.yaml")
    train = yaml_parse.load(yaml_file)
    train.main_loop()
    f.close()
    yaml_file.close()
    pkl_file = open('./rbm_11.pkl', 'rb')
    model = pickle.load(pkl_file)
    return model.get_weights(borrow=True)

def train_dbm():

    f = open('./dbm.pkl','wb')
    yaml_file = open ("./dbm.yaml")
    train = yaml_parse.load(yaml_file)
    train.main_loop()
    f.close()
    yaml_file.close()

if __name__ == '__main__':

    train_dae()
    # train_dbm()
    #train_rbm()

    X,y = pattern_generator.load_dataset("binary", True,"86")

    pkl_file = open('./dae.pkl', 'rb')
    model = pickle.load(pkl_file)

    W = model.get_weights(borrow=True)
    #Wt = np.transpose(W)
    X_new = np.dot(X,W)

    #X_new = pattern_generator.dea_output_resizeing(X_new)

    X_train, X_test, y_train, y_test = train_test_split(X_new, y, random_state=0)

    C = 1  # SVM regularization parameter

    rbf_svc = svm.SVC(kernel='rbf', gamma=0.7, C=C).fit(X_train, y_train)
    #scores = cross_validation.cross_val_score(rbf_svc, X_new, np.array(y), cv=5)
    #print scores

    print rbf_svc.score(X_test, y_test)


    #0.787 con 2 epoche
    #con 50 epoche rimane uguale

    #0.775 con pattern temporanie di 5

