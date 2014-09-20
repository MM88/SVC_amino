__author__ = 'miky'
import pattern_generator
from sklearn import svm
from sklearn.cross_validation import train_test_split
from pylearn2.config import yaml_parse
import pickle
import numpy as np


hyper_params_l1 = {'train_stop': 50,
                    'batch_size': 100,
                    'monitoring_batches': 1,
                    'nvis' : 231,
                    'nhid': 500,
                    'max_epochs': 500,
                   }
hyper_params_l2 = {'train_stop': 50,
                'batch_size': 100,
                'monitoring_batches': 1,
                'nvis': hyper_params_l1['nhid'],
                'nhid': 100,
                'max_epochs': 500,
                }
hyper_params_l3 = {'train_stop': 50,
                'batch_size': 100,
                'monitoring_batches': 1,
                'nvis': hyper_params_l2['nhid'],
                'nhid': 50,
                'max_epochs': 500,
                }

def train_yaml(yaml_file):

    train = yaml_parse.load(yaml_file)
    train.main_loop()


def train_layer1(yaml_file_path):

    yaml = open("./dae_l1.yaml".format(yaml_file_path), 'r').read()
    yaml = yaml % (hyper_params_l1)
    train_yaml(yaml)


def train_layer2(yaml_file_path):

    yaml = open("./dae_l2.yaml".format(yaml_file_path), 'r').read()
    yaml = yaml % (hyper_params_l2)
    train_yaml(yaml)

def train_layer3(yaml_file_path):

    yaml = open("./dae_l3.yaml".format(yaml_file_path), 'r').read()
    yaml = yaml % (hyper_params_l3)
    train_yaml(yaml)

def train_stack():

    #train_layer1("./dae_l1.yaml",)
    pkl_file = open('./dae_l1.pkl', 'rb')
    model = pickle.load(pkl_file)
    W_1 = model.get_weights(borrow=True)

    train_layer2("./dae_l2.yaml",)
    pkl_file = open('./dae_l2.pkl', 'rb')
    model = pickle.load(pkl_file)
    W_2 = model.get_weights(borrow=True)

    train_layer3("./dae_l3.yaml",)
    pkl_file = open('./dae_l3.pkl', 'rb')
    model = pickle.load(pkl_file)
    W_3 = model.get_weights(borrow=True)

    return W_1,W_2,W_3










