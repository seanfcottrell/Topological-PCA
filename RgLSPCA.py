import numpy as np
import pandas as pd

def gaussian_kernel(dist, t):
    '''
    gaussian kernel function for weighted edges
    '''
    return np.exp(-(dist**2 / t))

def Eu_dis(x):
    """
    Calculate the distance among each raw of x
    :param x: N X D
                N: number of samples
                D: Dimension of the feature
    :return: N X N distance matrix
    """
    x = np.asarray(x)
    aa = np.sum(np.multiply(x, x), 1)
    ab = x @ x.T
    dist_mat = aa + aa.T - 2 * ab
    dist_mat[dist_mat < 0] = 0
    dist_mat = np.sqrt(dist_mat)
    dist_mat = np.maximum(dist_mat, dist_mat.T)
    dist_mat = np.asarray(dist_mat)
    return dist_mat

def cal_weighted_adj(data, n_neighbors, t):
    '''
    Calculate weighted adjacency matrix based on KNN
    For each row of X, put an edge between nodes i and j
    If nodes are among the n_neighbors nearest neighbors of each other
    according to Euclidean distance
    '''
    dist = Eu_dis(data)
    n = dist.shape[0]
    gk_dist = gaussian_kernel(dist, t)
    W_L = np.zeros((n, n))
    for i in range(n):
        index_L = np.argsort(dist[i, :])[1:1 + n_neighbors] 
        len_index_L = len(index_L)
        for j in range(len_index_L):
            W_L[i, index_L[j]] = gk_dist[i, index_L[j]] #weighted edges
    W_L = np.maximum(W_L, W_L.T)
    return W_L

def cal_unweighted_adj(data, n_neighbors):
    '''
    Calculate unweighted adjacency matrix based on KNN
    '''
    dist = Eu_dis(data)
    n = dist.shape[0]
    W_L = np.zeros((n, n))
    for i in range(n):
        index_L = np.argsort(dist[i, :])[1:1 + n_neighbors]
        len_index_L = len(index_L)
        for j in range(len_index_L):
            W_L[i, index_L[j]] = 1 #edges not weighted
    W_L = np.maximum(W_L, W_L.T)
    return W_L

def cal_laplace(adj):
    N = adj.shape[0]
    D = np.zeros_like(adj)
    for i in range(N):
        D[i, i] = np.sum(adj[i]) # Degree Matrix
    L = D - adj  # Laplacian
    return L


def RpLSPCA_Algorithm(xMat,laplace,beta,gamma,k,n):
    '''
    Optimization Algorithm of RgLSPCA / RpLSPCA 
    Solve approximately via ADMM
    Need to compute optimal principal directions matrix U
    Projected Data matrix Q
    Error term matrix E = X - UQ^T
    Z matrix used to solve Q (see supplementary information)

    Inputs are data matrix X, laplacian term, scale parameters, 
    number of reduced dimensions, number of original dimensions
    '''
    # Initialize thresholds, matrices
    obj1 = 0
    obj2 = 0
    thresh = 1e-50
    V = np.eye(n) 
    vMat = np.asarray(V) # Auxillary matrix to optimize L2,1 norm
    E = np.ones((xMat.shape[0],xMat.shape[1]))
    E = np.asarray(E) # Error term X - UQ^T
    C = np.ones((xMat.shape[0],xMat.shape[1]))
    C = np.asarray(C) # Lagrangian Multiplier
    laplace = np.asarray(laplace) #Lplacian
    miu = 1 #Penalty Term
    for m in range(0, 30):
        Z = (-(miu/2) * ((E - xMat + C/miu).T @ (E - xMat + C/miu))) + beta * vMat + gamma * laplace
        # cal Q (Projected Data Matrix)
        Z_eigVals, Z_eigVects = np.linalg.eig(np.asarray(Z))
        eigValIndice = np.argsort(Z_eigVals)
        n_eigValIndice = eigValIndice[0:k]
        n_Z_eigVect = Z_eigVects[:, n_eigValIndice]
        # Optimal Q given by eigenvectors corresponding
        # to smallest k eigenvectors
        Q = np.array(n_Z_eigVect)  
        # cal V 
        q = np.linalg.norm(Q, ord=2, axis=1)
        qq = 1.0 / (q * 2)
        VV = np.diag(qq)
        vMat = np.asarray(VV)
        qMat = np.asarray(Q)
        # cal U (Principal Directions)
        U = (xMat - E - C/miu) @ qMat
        # cal P (intermediate step)
        P = xMat - U @ qMat.T - C/miu
        # cal E (Error Term)
        for i in range(E.shape[1]):
            E[:,i] = (np.max((1 - 1.0 / (miu * np.linalg.norm(P[:,i]))),0)) * P[:,i]
        # update C 
        C = C + miu * (E - xMat + U @ qMat.T)
        # update miu
        miu = 1.2 * miu

        obj1 = np.linalg.norm(qMat)
        if m > 0:
            diff = obj2 - obj1
            if diff < thresh:
                break # end iterations if error within accepted threshold
        obj2 = obj1
    return U #return principal directions matrix


def cal_persistent_laplace(W_L, zetas):
    n = W_L.shape[0]
    np.fill_diagonal(W_L,0)

    L = cal_laplace(W_L)
    #print("Laplace: ", L)

    np.fill_diagonal(L, 1e8) #Make sure diagonal is excluded from maximal and minimal value consideration
    min_l = np.min(L[np.nonzero(L)]) #Establish Min Value
    #print("min: ", min_l)
    np.fill_diagonal(L, -1e8)
    max_l = np.max(L[np.nonzero(L)]) #Establish Max Value
    #print("max: ", max_l)

    d = max_l - min_l
    #print("d: ", d)

    L = cal_laplace(W_L)
    PL = np.zeros((8,n,n))
    for k in range(1,8):
        PL[k,:,:] = np.where(L < (k/7*d + min_l), 1, 0) 
        #print("Threshold for k = ", k, ": ", k/7*d + min_l)
        np.fill_diagonal(PL[k,:,:],0)
        PL[k,:,:] = cal_laplace(PL[k,:,:])
        #print(PL[k,:,:])

    P_L = np.sum(zetas[:, np.newaxis, np.newaxis] * PL, axis=0)
     
    return P_L

def RgLSPCA_cal_projections(X_data, beta1, gamma1, k_d):
    n = len(X_data)  
    #dist = Eu_dis(X_data)
    #max_dist = np.max(dist)
    W_L = cal_unweighted_adj(X_data, n_neighbors=15) 
    A = W_L
    #PL = cal_persistent_laplace(A, zetas)
    L = cal_laplace(A)
    Y = RpLSPCA_Algorithm(X_data.transpose(), L, beta1, gamma1, k_d, n)
    return Y