import numpy as np
import pandas as pd
import warnings
import os
import sys
from RpLSPCA import RpLSPCA_cal_projections_KNN
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
warnings.filterwarnings("ignore")

from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score


'''
Code associated with clustering and its scores
'''

def computeClusterScore(X, y, label):
    #compute the clustering scores
    #ARI, NMI and silhouette scores
    ari = adjusted_rand_score(y, label)
    nmi = normalized_mutual_info_score(y, label)
    sil = silhouette_score(X, label)
    return ari, nmi, sil
    

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
    ARI = np.zeros(max_state); NMI = np.zeros(max_state); SIL = np.zeros(max_state)
    for state in range(max_state):
        myKM = KMeans(n_clusters = n_clusters,  n_init = 150, random_state = state)
        myKM.fit(X_scaled)
        label = myKM.labels_
        ARI[state], NMI[state], SIL[state] = computeClusterScore(X, y, label)
        LABELS[state, :] = label
    return ARI, NMI


def load_X(data):
    inpath = rootPath + '/tPCA_Workshop/data/%s/'%(data)
    X = pd.read_csv(inpath + '%s_full_X.csv'%(data))
    X = X.values[:, 1:].astype(float)
    return X

def load_y(data):
    inpath = rootPath + '/tPCA_Workshop/data/%s/'%(data)
    y = pd.read_csv(inpath + '%s_full_labels.csv'%(data))
    y = np.array(list(y['Label'])).astype(int)
    return y

if __name__ == '__main__':
    X = load_X('GSE75748cell')
    y = load_y('GSE75748cell')

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
    #print('Gene filtering X shape:', filtered_X.shape)

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

    # Normalize data
    scaler = StandardScaler()
    scaler.fit(X_filtered)
    X_normalized = scaler.transform(X_filtered)
    
    k = np.unique(y_filtered).shape[0]
    zeta = [0,0,0,0,0,0,1,1] #float(sys.argv[1])
    gamma = 100000
    beta = 60

    X_filtered= np.asarray(X_filtered)
    X_normalized = np.asarray(X_normalized)

    
    print('---------------Un-Optimized RpLSPCA KNN-------------')
    RpLSPCA_KNN_ari_mean = 0
    RpLSPCA_KNN_nmi_mean = 0
    #Principal Components
    PDM = RpLSPCA_cal_projections_KNN(X_normalized.T, beta, gamma, k, 15, [1,1/2,1/3,1/4,1/5,1/6,2/7,1/8])
    PDM = np.asarray(PDM)
    TM = ((np.linalg.inv(PDM.T @ PDM)) @ (PDM.T)).T #Construct a transformation matrixwe

    #Projected Data Matrix
    Q = (X_normalized.T @ TM)

    #Clustering
    RpLSPCA_KNN_ari, RpLSPCA_KNN_nmi = computeKMeans(Q, y_filtered, max_state=30)
    RpLSPCA_KNN_ari_mean = (RpLSPCA_KNN_ari.sum()) / 30
    RpLSPCA_KNN_nmi_mean = (RpLSPCA_KNN_nmi.sum()) / 30
    print('RpLSPCA KNN ARI:', RpLSPCA_KNN_ari_mean)
    print( 'RpLSPCA KNN NMI:', RpLSPCA_KNN_nmi_mean )

    print('---------------Optimized RpLSPCA KNN-------------')
    RpLSPCA_KNN_ari_mean = 0
    RpLSPCA_KNN_nmi_mean = 0
    #Principal Components
    PDM = RpLSPCA_cal_projections_KNN(X_normalized.T, beta, gamma, k, 15, zeta)
    PDM = np.asarray(PDM)
    TM = ((np.linalg.inv(PDM.T @ PDM)) @ (PDM.T)).T

    #Projected Data Matrix
    Q = (X_normalized.T @ TM)

    #Clustering
    RpLSPCA_KNN_ari, RpLSPCA_KNN_nmi = computeKMeans(Q, y_filtered, max_state=30)
    RpLSPCA_KNN_ari_mean = (RpLSPCA_KNN_ari.sum()) / 30
    RpLSPCA_KNN_nmi_mean = (RpLSPCA_KNN_nmi.sum()) / 30
    print('RpLSPCA KNN ARI:', RpLSPCA_KNN_ari_mean)
    print( 'RpLSPCA KNN NMI:', RpLSPCA_KNN_nmi_mean )

    #write_file = open('/mnt/home/cottre61/data/parameter_tuning_results/GSE75748time_Classification_scores.csv','a+')
    #write_file.write('Alpha1: %.3f, Alpha2: %.3f, Alpha3: %.3f, Alpha4: %.3f, Alpha5: %.3f, Alpha6: %.3f, Alpha7: %.3f, Alpha8: %.3f, Beta: %.3f, Gamma: %.3f, ACC: %.6f, PRE: %.6f, REC: %.6f, F1: %.6f\n'%(zeta[0], zeta[1],zeta[2], zeta[3],zeta[4],zeta[5],zeta[6],zeta[7],beta, gamma, np.mean(accuracylist),np.mean(precisionlist),np.mean(recalllist),np.mean(f1list)))
    #write_file.close()

