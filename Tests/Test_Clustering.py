import numpy as np
import pandas as pd
import warnings
import os
import sys
from RpLSPCA import RpLSPCA_cal_projections_KNN, RpLSPCA_cal_projections
from RgLSPCA import RgLSPCA_cal_projections
from sPCA import SPCA_cal_projections
from PCA import PCA_cal_projections
from umap.umap_ import UMAP
from sklearn.decomposition import NMF
from sklearn.manifold import TSNE
from clusim.clustering import Clustering
import clusim.sim as sim
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from wrapper import ParameterWrapper
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
warnings.filterwarnings("ignore")

'''
Code associated with clustering and its scores
'''

def computeClusterScore(y, label):
    #compute the clustering scores
    #ARI, NMI scores
    ari = adjusted_rand_score(y, label)
    nmi = normalized_mutual_info_score(y, label)
    return ari, nmi

def computeElementCenstricScore(y, label):
    clustering1 = Clustering().from_membership_list(y)
    clustering2 = Clustering().from_membership_list(label)
    score = sim.element_sim(clustering1, clustering2, alpha=0.9)
    return score

def computeKMeans(X, y, max_state = 30):
    '''
        compute k-means clustering for the reduction with 30 random instance
        input:
            X: M x N data
            y: M * 1 true labels
            max_state: number of k-means state
        return:
            LABELS: max_state * M label from k-means
            ARI: max_state * 1 ari for each instance of k-means
            NMI: max_state * 1 nmi for each instance of k-means
            Sil: max_state * 1 silhouette score for each instance of k-means
    '''
    M = X.shape[0]
    n_clusters = np.unique(y).shape[0]
    X_scaled = StandardScaler().fit_transform(X)
    LABELS = np.zeros([max_state, M])
    ARI = np.zeros(max_state); NMI = np.zeros(max_state); ECS = np.zeros(max_state)
    for state in range(max_state):
        myKM = KMeans(n_clusters = n_clusters,  n_init = 150, random_state = state)
        myKM.fit(X_scaled)
        label = myKM.labels_
        ARI[state], NMI[state] = computeClusterScore(y, label)
        ECS[state] = computeElementCenstricScore(y, label)
        LABELS[state, :] = label
    return ARI, NMI, ECS

def tSNE_KMeans(X,y,k,state = 20):
    ARI = []
    NMI = []
    ECS = []
    for state in range(state):
        t_sne = TSNE(
        n_components=k,
        init="random",
        method='exact',
        n_iter=250,
        random_state=state,
        ) 

        Q = t_sne.fit_transform(X)

        tsne_ari, tsne_nmi, tsne_ecs = computeKMeans(Q, y, max_state=30)
        ARI.append((tsne_ari.sum()) / 30)
        NMI.append((tsne_nmi.sum()) / 30)
        ECS.append((tsne_ecs.sum()) / 30)
    ARI_score = np.mean(ARI)
    NMI_score = np.mean(NMI)
    ECS_score = np.mean(ECS)

    return ARI_score, NMI_score, ECS_score

def computeNMF(X, k, state):
    myNMF = NMF(n_components=k,  init = 'random', max_iter=300, random_state=state)
    X_nmf = myNMF.fit_transform(X)
    return X_nmf

def NMF_KMeans(X, y, k, state = 20):
    ARI = []
    NMI = []
    ECS = []
    for state in range(state):
        Q = computeNMF(X, k, state)
        NMF_ari, NMF_nmi, NMF_ecs = computeKMeans(Q, y, max_state=30)
        ARI.append((NMF_ari.sum()) / 30)
        NMI.append((NMF_nmi.sum()) / 30)
        ECS.append((NMF_ecs.sum()) / 30)
    ARI_score = np.mean(ARI)
    NMI_score = np.mean(NMI)
    ECS_score = np.mean(ECS)

    return ARI_score, NMI_score, ECS_score

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
    
    # 1e4 Normalizaton for count matrices
    if sys.argv[1] in ['GSE67835', 'GSE84133human1', 'GSE84133human2', 'GSE84133human3', 'GSE84133mouse1', 'GSE84133mouse2']:
        # total counts for each cell 
        cell_totals = np.sum(X, axis=0)
        # scaling factors
        scaling_factors = np.array(1e4 / cell_totals)
        # Normalize matrix
        X = scaling_factors * X

    # Log transform data
    log_transform = np.vectorize(np.log)
    log_X = log_transform(X+1)

    # Set values below 1e-6 to 0
    log_X[log_X < 1e-6] = 0

    #print('X shape:', X.shape)
    #print("y shape:", y.shape)

    # Filter out features with low variance
    row_variances = np.var(log_X, axis=1)
    variance_threshold = np.percentile(row_variances, 25)  # Adjust the percentile as needed
    filtered_X = log_X[row_variances >= variance_threshold]
    #print('Post gene filtering X shape:', filtered_X.shape)

    # Filter out classes with fewer than 15 samples
    filtered_X_transposed = filtered_X.T
    # Get the unique classes and their counts
    unique_classes, class_counts = np.unique(y, return_counts=True)
    # Find the classes with less than 15 samples
    classes_to_remove = unique_classes[class_counts < 15]
    # Create a mask to filter the samples and labels
    mask = np.isin(y, classes_to_remove, invert=True)
    # Filter the dataset and labels
    X_filtered = filtered_X_transposed[mask].T
    y_filtered = y[mask]
    # Print the filtered dataset and labels
    #print("Filtered X shape:", X_filtered.shape)
    #print("Filtered y shape:", y_filtered.shape)

    # Standardize the
    scaler = StandardScaler()
    scaler.fit(X_filtered)
    X_normalized = scaler.transform(X_filtered)
    
    # Get parameters for a specific dataset for kNN tPCA
    param_wrapper = ParameterWrapper(rootPath + '/Tests/paramvals_kNNtPCA.csv')
    params_dataset1 = param_wrapper.get_parameters(sys.argv[1])
    print('kNN-tPCA Parameters:', params_dataset1)  

    k = np.unique(y_filtered).shape[0] # Num Clusters = Num Dimensions
    zeta = [float(params_dataset1['zeta1']), float(params_dataset1['zeta2']), 
            float(params_dataset1['zeta3']), float(params_dataset1['zeta4']),
            float(params_dataset1['zeta5']), float(params_dataset1['zeta6']),
            float(params_dataset1['zeta7']), float(params_dataset1['zeta8'])]
    zeta = np.array(zeta)
    gamma = float(params_dataset1['gamma'])
    beta = float(params_dataset1['beta'])

    # Get parameters for a specific dataset for tPCA
    param_wrapper2 = ParameterWrapper(rootPath + '/Tests/paramvals_tPCA.csv')
    params_dataset2 = param_wrapper2.get_parameters(sys.argv[1])
    print('tPCA Parameters:', params_dataset2)  

    zeta2 = [float(params_dataset2['zeta1']), float(params_dataset2['zeta2']), 
            float(params_dataset2['zeta3']), float(params_dataset2['zeta4']),
            float(params_dataset2['zeta5']), float(params_dataset2['zeta6']),
            float(params_dataset2['zeta7'])]
    zeta2 = np.array(zeta2)
    gamma2 = float(params_dataset2['gamma'])
    beta2 = float(params_dataset2['beta'])

    X_filtered= np.asarray(X_filtered)
    X_normalized = np.asarray(X_normalized)

    print('---------------RpLSPCA KNN-------------')
    #Principal Components
    PDM = RpLSPCA_cal_projections_KNN(X_normalized.T, beta, gamma, k, 15, zeta)
    PDM = np.asarray(PDM)
    TM = ((np.linalg.inv(PDM.T @ PDM)) @ (PDM.T)).T

    #Projected Data Matrix
    Q = (X_normalized.T @ TM)

    #Clustering
    RpLSPCA_KNN_ari, RpLSPCA_KNN_nmi, RpLSPCA_KNN_ecs = computeKMeans(Q, y_filtered, max_state=30)
    RpLSPCA_KNN_ari_mean = (RpLSPCA_KNN_ari.sum()) / 30
    RpLSPCA_KNN_nmi_mean = (RpLSPCA_KNN_nmi.sum()) / 30
    RpLSPCA_KNN_ecs_mean = (RpLSPCA_KNN_ecs.sum()) / 30
    print('RpLSPCA KNN ARI:', RpLSPCA_KNN_ari_mean)
    print('RpLSPCA KNN NMI:', RpLSPCA_KNN_nmi_mean)
    print('RpLSPCA KNN Element Centric Score:', RpLSPCA_KNN_ecs_mean)


    print('---------------RpLSPCA-------------')
    #Principal Components
    PDM = RpLSPCA_cal_projections(X_normalized.T, beta2, gamma2, k, zeta2)
    PDM = np.asarray(PDM)
    TM = ((np.linalg.inv(PDM.T @ PDM)) @ (PDM.T)).T

    #Projected Data Matrix
    Q = (X_normalized.T @ TM)

    #Clustering
    RpLSPCA_ari, RpLSPCA_nmi, RpLSPCA_ecs = computeKMeans(Q, y_filtered, max_state=30)
    RpLSPCA_ari_mean = (RpLSPCA_ari.sum()) / 30
    RpLSPCA_nmi_mean = (RpLSPCA_nmi.sum()) / 30
    RpLSPCA_ecs_mean = (RpLSPCA_ecs.sum()) / 30
    print('RpLSPCA ARI:', RpLSPCA_ari_mean)
    print('RpLSPCA NMI:', RpLSPCA_nmi_mean)
    print('RpLSPCA Element Centric Score:', RpLSPCA_ecs_mean)

    
    print('-----------------NMF-----------------')
    NMF_ari, NMF_nmi, NMF_ecs = NMF_KMeans(X_filtered.T, y_filtered, k, state = 20)
    
    print('NMF ARI:', NMF_ari)
    print('NMF NMI:', NMF_nmi)
    print('NMF Element Centric Score:', NMF_ecs)

    
    print('-----------------PCA-----------------')
    #Dimensionality Reduction (data unscaled)
    PDM = PCA_cal_projections(X_normalized.T, k)
    PDM = np.asarray(PDM)
    TM = ((np.linalg.inv(PDM.T @ PDM)) @ (PDM.T)).T

    #Projected Data Matrix
    Q = (X_normalized.T @ TM)

    #Clustering 
    PCA_ari, PCA_nmi, PCA_ecs = computeKMeans(Q, y_filtered, max_state=30)
    PCA_ari_mean = (PCA_ari.sum()) / 30
    PCA_nmi_mean = (PCA_nmi.sum()) / 30
    PCA_ecs_mean = (PCA_ecs.sum()) / 30
    print('PCA ARI:', PCA_ari_mean)
    print('PCA NMI:', PCA_nmi_mean)
    print('PCA Element Centric Score:', PCA_ecs_mean)


    print('-----------------sPCA-----------------')
    #Dimensionality Reduction (data unscaled)
    PDM = SPCA_cal_projections(X_normalized.T, beta2, k)
    PDM = np.asarray(PDM)
    TM = ((np.linalg.inv(PDM.T @ PDM)) @ (PDM.T)).T

    #Projected Data Matrix
    Q = (X_normalized.T @ TM)

    #Clustering 
    sPCA_ari, sPCA_nmi, sPCA_ecs = computeKMeans(Q, y_filtered, max_state=30)
    sPCA_ari_mean = (sPCA_ari.sum()) / 30
    sPCA_nmi_mean = (sPCA_nmi.sum()) / 30
    sPCA_ecs_mean = (sPCA_ecs.sum()) / 30
    print('sPCA ARI:', sPCA_ari_mean)
    print('sPCA NMI:', sPCA_nmi_mean)
    print('sPCA Element Centric Score:', sPCA_ecs_mean)


    print('---------------RgLSPCA----------------')
    RgLSPCA_ari_mean = 0
    RgLSPCA_nmi_mean = 0
    #Principal Components
    PDM = RgLSPCA_cal_projections(X_normalized.T, beta2, gamma2, k)
    PDM = np.asarray(PDM)
    TM = ((np.linalg.inv(PDM.T @ PDM)) @ (PDM.T)).T

    #Projected Data Matrix
    Q = (X_normalized.T @ TM)

    #Clustering
    RgLSPCA_ari, RgLSPCA_nmi, RgLSPCA_ecs = computeKMeans(Q, y_filtered, max_state=30)
    RgLSPCA_ari_mean = (RgLSPCA_ari.sum()) / 30
    RgLSPCA_nmi_mean = (RgLSPCA_nmi.sum()) / 30
    RgLSPCA_ecs_mean = (RgLSPCA_ecs.sum()) / 30
    print('RgLSPCA ARI:', RgLSPCA_ari_mean)
    print('RgLSPCA NMI:', RgLSPCA_nmi_mean )
    print('RgLSPCA Element Centric Score:', RgLSPCA_ecs_mean)

    
    print('----------------tSNE---------------')
    tSNE_ari, tSNE_nmi, tsne_ecs = tSNE_KMeans(X_normalized.T, y_filtered, k, state = 20)
    
    print('tSNE ARI:', tSNE_ari)
    print('tSNE NMI:', tSNE_nmi)
    print('tSNE Element Centric Score:', tsne_ecs)

    
    print('---------------UMAP----------------')
    reducer = UMAP(n_components = k, min_dist = 0.1)
    Q = reducer.fit_transform(X_filtered.T)

    UMAP_ari, UMAP_nmi, umap_ecs = computeKMeans(Q, y_filtered, max_state=30)
    UMAP_ari_mean = (UMAP_ari.sum()) / 30
    UMAP_nmi_mean = (UMAP_nmi.sum()) / 30
    umap_ecs_mean = (umap_ecs.sum()) / 30
    print('UMAP ARI:', UMAP_ari_mean)
    print('UMAP NMI:', UMAP_nmi_mean )
    print('UMAP Element Centric Score:', umap_ecs_mean)

