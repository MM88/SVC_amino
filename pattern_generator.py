import numpy as np
import pickle, cPickle

mydict = {"A":"100000000000000000000",
          "C":"010000000000000000000",
          "D":"001000000000000000000",
          "E":"000100000000000000000",
          "F":"000010000000000000000",
          "G":"000001000000000000000",
          "H":"000000100000000000000",
          "I":"000000010000000000000",
          "K":"000000001000000000000",
          "L":"000000000100000000000",
          "M":"000000000010000000000",
          "N":"000000000001000000000",
          "P":"000000000000100000000",
          "Q":"000000000000010000000",
          "R":"000000000000001000000",
          "S":"000000000000000100000",
          "T":"000000000000000010000",
          "V":"000000000000000001000",
          "W":"000000000000000000100",
          "Y":"000000000000000000010",
          "X":"000000000000000000001",
          "\n":""}

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass

    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
    return False

def binary_pattern():

    """
        :return:
    """
    datasettemp = open("./datasettemp3.txt","r")
    y = []
    patterns = []
    for line in datasettemp:
        if line[1].isalpha():
            for k, v in mydict.iteritems():
                line = line.replace(k, v)
            t = [int(i) for i in line]
            patterns.append(t)
        else:
            y.append(int(line[:-1])) #the last character is a "\n"

    y = np.array(y)
    X = np.array(patterns)

    dataset = [X,y]
    f = open('./binary_dataset_353.pkl','wb')
    cPickle.dump(dataset, f, protocol=2)
    f.close()




def pssm_patterns(N):

    """

    :param N: pattern length
    :return:
    """
    datasettemp = open("./datasettemp3.txt","r")
    y = []

    for line in datasettemp:
          if not line[1].isalpha():
            y.append(int(line[:-1]))

    X_tmp = []
    for i in range(1,87):
        if (i!=5 and i!=35 and i!=83):
            folder = "/home/miky/PycharmProjects/SVC_amino/fasta_files/"+str(i)+"/pssm.txt"
            pssm = []
            with open(folder, 'r') as f_in:
                lines = (line.rstrip() for line in f_in) # All lines including the blank ones
                lines = (line for line in lines if line) # Non-blank lines
                for line in lines:
                    if (is_number(line[4])):
                        #prende la prima matrice pssm
                        #row = [int(j) for j in line[10:71].split()]
                        #prende la matrice delle probabilita
                        row = [float(j)/100 for j in line[73:152].split()]
                        pssm.append(row)

                pssm = np.array(pssm)
                dummy =np.array([[-1 for l in range(0,20)] for t in range(0,(N-1)/2)]) #pssm values for dummy amino acid X
                pssm = np.vstack((dummy,pssm,dummy))
            f_in.close()
            for i in range(0,pssm.shape[0]-10):
                pattern = []
                for j in range(i,i+N):
                    for k in range(0,20):
                        pattern.append(pssm[j][k])
                X_tmp.append(pattern)

    y = np.array(y)
    X = np.array(X_tmp)
    dataset = [X,y]
    f = open('./pssm_dataset.pkl','wb')
    cPickle.dump(dataset, f, protocol=2)
    f.close()


def dataset_splitting(N):
    """

    :param N: pattern length
    """
    f = open("./rb198_111_44.txt","r")
    datasettemp = open("./datasettemp3.txt","w")
    x_str = "X"*((N-1)/2)
    zero_str = "0"*((N-1)/2)
    patterns = []
    classes = []
    for line in f:
     if line[1].isalpha():
        next_line = f.next()
        line = x_str+line[:-1]+x_str
        next_line = zero_str+next_line[:-1]+zero_str
        for i in range(0,len(line)-N+1):
            patterns.append(line[i:i+N])
            classes.append(next_line[i+(N-1)/2])
            datasettemp.write(patterns[-1]+"\n")
            datasettemp.write(classes[-1]+"\n")
    datasettemp.close()


def load_dataset(type,all_dataset,dataset):

    """

    :param type:
    :param all_dataset:
    """
    dataset_path = open("./"+type+"_dataset_"+dataset+".pkl",'r')
    dataset = pickle.load(dataset_path)

    dataset[1] = [1 if element==1 else -1 for element in dataset[1] ]

    if all_dataset:
        return dataset
    else:
        return dataset[0]

def dae_input_generator(all_dataset):

    """
    :return:
    """
    if all_dataset:
        X,y = load_dataset("binary", True,"86")
        N = 11
        amino_size = 21
        pattern_size = 5
        X_new1 = []
        for row in X:
                for j in range(0, (N*amino_size - pattern_size*amino_size) +1, amino_size):
                    X_new1.append(row[j: j+(amino_size*pattern_size)])
        X_new1 = np.array(X_new1)
        return X_new1,y
    else:
        X = load_dataset("binary", False,"439")
        N = 11
        amino_size = 21
        pattern_size = 5
        X_new1 = []
        for row in X:
                for j in range(0, (N*amino_size - pattern_size*amino_size) +1, amino_size):
                    X_new1.append(row[j: j+(amino_size*pattern_size)])
        X_new1 = np.array(X_new1)
        return X_new1

def dae_output_resizeing(X):

	pattern_size = 7
	X_new = []
	for i in range(0, X.shape[0], pattern_size):
		X_new.append(X[i:i + (pattern_size-1)][:].flatten())
	X_new = np.array(X_new)

	return X_new

if __name__ == '__main__':

    N = 11
    dataset_splitting(N)
    binary_pattern()
    # f = open("./rna_86.txt","r")
    # pssm_patterns(N)





			                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               