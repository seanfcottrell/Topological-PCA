import numpy as np
import pandas as pd
import warnings
import os
import sys
from scipy.stats import skew, kurtosis

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
warnings.filterwarnings("ignore")

def load_X(data):
    inpath = rootPath + '/Tests/%s/'%(data)
    X = pd.read_csv(inpath + '%s_full_X.csv'%(data))
    X = X.values[:, 1:].astype(float)
    return X

def load_y(data):
    inpath = rootPath + '/Tests/%s/'%(data)
    y = pd.read_csv(inpath + '%s_full_labels.csv'%(data))
    y = np.array(list(y['Label'])).astype(int)
    return y

if __name__ == '__main__':
    X = load_X(sys.argv[1])
    y = load_y(sys.argv[1])

    print('Shape of Data (Cells x Genes):', np.asarray(X).shape)
    print('Sparsity:', (np.size(X) - np.count_nonzero(X))/np.size(X) * 100)
    non_zero_elements = X[X != 0]
    print('Max Value:', np.max(non_zero_elements))
    print('Mean Value:', np.mean(non_zero_elements))
    print('Median Value:', np.median(non_zero_elements))
    print('Skew:', skew(non_zero_elements))
    print('Kurtosis:', kurtosis(non_zero_elements))
    print('Num Clusters and Cluster Sizes:', np.unique(y,return_counts=True))
