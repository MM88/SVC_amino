import numpy as np
import pylab as pl
from sklearn import svm, datasets
from sklearn.utils import shuffle
from sklearn.metrics import roc_curve, auc
from sklearn.cross_validation import train_test_split
from sklearn.metrics import matthews_corrcoef
from sklearn import cross_validation
import pickle
import model_train
import pattern_generator
import stacked_dae
from sklearn.metrics import roc_auc_score,accuracy_score,recall_score,matthews_corrcoef

def svc_amino(X, y, score_type):
    """

    :param X:
    :param y:
    :param score_type:
    """
    

    if (score_type=="split"):
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

        C = 70  # SVM regularization parameter
        rbf_svc = svm.SVC(kernel='rbf', gamma=0.07, C=C)
        rbf_svc.fit(X_train, y_train)
        y_score = np.array(rbf_svc.predict(X_test))
        y_test = np.array(y_test)
        tn = 0.0
        fp = 0.0

        for i in range(y_score.shape[0]):
            if y_test[i]==-1:
                if y_score[i]==-1:
                    tn = tn+1
                else: fp = fp+1
        spec = tn/(tn+fp)
        print "sensitivity:"
        print recall_score(y_test,y_score)
        print "specificity:"
        print spec
        print "accuracy:"
        print accuracy_score(y_test,y_score)
        print "MCC:"
        print matthews_corrcoef(y_test,y_score)

        


        return "ciao"
        #con binary fa 0.78 con pssm fa 0.80
    else:
        if(score_type=="cross"):
            scores = cross_validation.cross_val_score(rbf_svc, X, np.array(y), cv=5)
            return scores


def train_model(model_type):
    """

    :param model_type:
    """
    if (model_type=="dae"):
        #model_train.train_dae()
        pkl_file = open('./dae.pkl', 'rb')
        model = pickle.load(pkl_file)
        return model.get_weights(borrow=True)
    elif (model_type=="rbm"):
        #model_train.train_rbm()
        pkl_file = open('./rbm_11.pkl', 'rb')
        model = pickle.load(pkl_file)
        return model.get_weights(borrow=True)
    elif (model_type=="dbm"):
        model_train.train_dbm()
        pkl_file = open('./dbm.pkl', 'rb')
        model = pickle.load(pkl_file)
        return model.get_weights(borrow=True)

if __name__=="__main__":

    W = train_model("rbm")
    
    X,y = pattern_generator.load_dataset("binary",True,"86")
    X_new = np.dot(X,W)
   
    print svc_amino(X_new,y,"split")