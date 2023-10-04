import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

def SPCA_Algorithm(xMat,beta,k,n):
    obj1 = 0
    obj2 = 0
    thresh = 1e-50
    V = np.eye(n)  # (500, 500)
    vMat = np.mat(V)  # (500, 500)
    for m in range(0, 10):
        Z = -(xMat.T * xMat) + beta * vMat  # (643, 643)
        Z_eigVals, Z_eigVects = np.linalg.eig(np.mat(Z))
        eigValIndice = np.argsort(Z_eigVals)
        n_eigValIndice = eigValIndice[0:k]
        n_Z_eigVect = Z_eigVects[:, n_eigValIndice]
        Q = np.array(n_Z_eigVect)  # (643, 3)
        q = np.linalg.norm(Q, ord=2, axis=1)
        qq = 1.0 / (q * 2)
        VV = np.diag(qq)  # (643, 643)
        vMat = np.mat(VV)  # (643, 643)
        qMat = np.mat(Q)  # (643, 3)
        Y = xMat * qMat  # (20502, 3)
        # obj1 = (np.linalg.norm(xMat - Y * qMat.T, ord='fro')) ** 2 + alpha * (
        #     np.linalg.norm(bMat - A * qMat.T, ord='fro')) ** 2 + beta * np.trace(qMat.T * vMat * qMat) + garma * np.trace(qMat.T * laplace * qMat)
        obj1 = np.linalg.norm(qMat)
        if m > 0:
            diff = obj2 - obj1
            if diff < thresh:
                break
        obj2 = obj1
    return Y

def SPCA_cal_projections(X_data,beta1, k_d):
    # nclass = 2
    # nclass = B_data.shape[1]
    n = len(X_data)  # 500
    Y= SPCA_Algorithm(X_data.transpose(), beta1, k_d, n)
    return Y
