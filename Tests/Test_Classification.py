import numpy as np
import pandas as pd
import warnings
import os
import sys
from RpLSPCA import RpLSPCA_cal_projections_KNN
from RgLSPCA import RgLSPCA_cal_projections
from sPCA import SPCA_cal_projections
from PCA import PCA_cal_projections
from umap.umap_ import UMAP
from sklearn.decomposition import NMF
from sklearn.manifold import TSNE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from wrapper import ParameterWrapper
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
warnings.filterwarnings("ignore")

def tSNE(X,k,state = 1):
    t_sne = TSNE(
        n_components=k,
        init="random",
        method='exact',
        n_iter=300,
        random_state=state,
    ) 
    Q = t_sne.fit_transform(X)
    return Q

def umap(X, k):
    reducer = UMAP(n_components = k, min_dist = 0.1)
    Q = reducer.fit_transform(X)
    return Q

def computeNMF(X, k, state = 1):
    myNMF = NMF(n_components=k,  init = 'random', max_iter=300, random_state=state)
    X_nmf = myNMF.fit_transform(X)
    return X_nmf

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
    variance_threshold = np.percentile(row_variances, 20)  # Adjust the percentile as needed
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

    # Standardize the data
    scaler = StandardScaler()
    scaler.fit(X_filtered)
    X_normalized = scaler.transform(X_filtered)
    
    # Get parameters for a specific dataset for kNN tPCA
    param_wrapper = ParameterWrapper(rootPath + '/Tests/paramvals_kNNtPCA_class.csv')
    params_dataset1 = param_wrapper.get_parameters(sys.argv[1])
    print('kNN-tPCA Parameters:', params_dataset1)  

    zeta = [float(params_dataset1['zeta1']), float(params_dataset1['zeta2']), 
            float(params_dataset1['zeta3']), float(params_dataset1['zeta4']),
            float(params_dataset1['zeta5']), float(params_dataset1['zeta6']),
            float(params_dataset1['zeta7']), float(params_dataset1['zeta8'])]
    zeta = zeta[::-1]
    zeta = np.array(zeta)
    gamma = float(params_dataset1['gamma'])
    beta = float(params_dataset1['beta'])

    # Previously pubslihed parameter values
    beta2 = 60
    gamma2 = 3

    X_filtered= np.asarray(X_filtered)
    X_normalized = np.asarray(X_normalized)
    y_filtered = np.asarray(y_filtered)

    knc = KNeighborsClassifier(n_neighbors=1)

    accuracylist = []
    precisionlist = []
    recalllist = []
    f1list = []

    print('---------------RpLSPCA KNN----------------')
    for k in [1,10,20,30,40,50,60,70,80,90,100]:
        print('k = ', k)
        accuracy = 0
        precision = 0
        recall = 0
        f1 = 0
        for per in range(1, 6):
            #Principal Components
            x_train, x_test, y_train, y_test = train_test_split(X_normalized.T, y_filtered.T, test_size=int(round(X_normalized.shape[1]*0.4)), random_state=per)
            PDM = RpLSPCA_cal_projections_KNN(x_train, beta, gamma, k, 15, zeta)
            PDM = np.asarray(PDM)
            TM = ((np.linalg.inv(PDM.T @ PDM)) @ (PDM.T)).T

            #Projected Data Matrix
            Q_train = (np.asarray(x_train) @ TM)
            Q_test = (np.asarray(x_test) @ TM)

            #Classification
            knc.fit(np.real(Q_train), y_train)
            y_predict = knc.predict(np.real(Q_test))
        
            #K-Fold Metrics
            accuracy += accuracy_score(y_test, y_predict)
            precision += precision_score(y_test, y_predict, average='macro')
            recall += recall_score(y_test, y_predict, average='macro')
            f1 += f1_score(y_test, y_predict, average='macro')
        accuracylist.append(accuracy / 5)
        precisionlist.append(precision / 5)
        recalllist.append(recall / 5)
        f1list.append(f1 / 5)
        print('---------------------------------------')
    print('RpLSPCA KNN ACC:', np.mean(accuracylist))
    print('RpLSPCA KNN REC:', np.mean(recalllist))
    print('RpLSPCA KNN PRE:', np.mean(precisionlist))
    print('RpLSPCA KNN F1:', np.mean(f1list))


    print('---------------RgLSPCA----------------')
    accuracylist.clear()
    precisionlist.clear()
    recalllist.clear()
    f1list.clear()
    for k in [1,10,20,30,40,50,60,70,80,90,100]:
        print('k = ', k)
        accuracy = 0
        precision = 0
        recall = 0
        f1 = 0
        for per in range(1, 6):
            #Principal Components
            x_train, x_test, y_train, y_test = train_test_split(X_normalized.T, y_filtered.T, test_size=int(round(X_normalized.shape[1]*0.4)), random_state=per)
            PDM = RgLSPCA_cal_projections(x_train, beta2, gamma2, k)
            PDM = np.asarray(PDM)
            TM = ((np.linalg.inv(PDM.T @ PDM)) @ (PDM.T)).T

            #Projected Data Matrix
            Q_train = (np.asarray(x_train) @ TM)
            Q_test = (np.asarray(x_test) @ TM)

            #Classification
            knc.fit(np.real(Q_train), y_train)
            y_predict = knc.predict(np.real(Q_test))
        
            #K-Fold Metrics
            accuracy += accuracy_score(y_test, y_predict)
            precision += precision_score(y_test, y_predict, average='macro')
            recall += recall_score(y_test, y_predict, average='macro')
            f1 += f1_score(y_test, y_predict, average='macro')
        accuracylist.append(accuracy / 5)
        precisionlist.append(precision / 5)
        recalllist.append(recall / 5)
        f1list.append(f1 / 5)
        print('---------------------------------------')
    print('RgLSPCA ACC:', np.mean(accuracylist))
    print('RgLSPCA REC:', np.mean(recalllist))
    print('RgLSPCA PRE:', np.mean(precisionlist))
    print('RgLSPCA F1:', np.mean(f1list))


    print('---------------SPCA----------------')
    accuracylist.clear()
    precisionlist.clear()
    recalllist.clear()
    f1list.clear()
    for k in [1,10,20,30,40,50,60,70,80,90,100]:
        print('k = ', k)
        accuracy = 0
        precision = 0
        recall = 0
        f1 = 0
        for per in range(1, 6):
            #Principal Components
            x_train, x_test, y_train, y_test = train_test_split(X_normalized.T, y_filtered.T, test_size=int(round(X_normalized.shape[1]*0.4)), random_state=per)
            PDM = SPCA_cal_projections(x_train, beta2, k)
            PDM = np.asarray(PDM)
            TM = ((np.linalg.inv(PDM.T @ PDM)) @ (PDM.T)).T

            #Projected Data Matrix
            Q_train = (np.asarray(x_train) @ TM)
            Q_test = (np.asarray(x_test) @ TM)

            #Classification
            knc.fit(np.real(Q_train), y_train)
            y_predict = knc.predict(np.real(Q_test))
        
            #K-Fold Metrics
            accuracy += accuracy_score(y_test, y_predict)
            precision += precision_score(y_test, y_predict, average='macro')
            recall += recall_score(y_test, y_predict, average='macro')
            f1 += f1_score(y_test, y_predict, average='macro')
        accuracylist.append(accuracy / 5)
        precisionlist.append(precision / 5)
        recalllist.append(recall / 5)
        f1list.append(f1 / 5)
        print('---------------------------------------')
    print('SPCA ACC:', np.mean(accuracylist))
    print('SPCA REC:', np.mean(recalllist))
    print('SPCA PRE:', np.mean(precisionlist))
    print('SPCA F1:', np.mean(f1list))


    print('---------------PCA----------------')
    accuracylist.clear()
    precisionlist.clear()
    recalllist.clear()
    f1list.clear()
    for k in [1,10,20,30,40,50,60,70,80,90,100]:
        print('k = ', k)
        accuracy = 0
        precision = 0
        recall = 0
        f1 = 0
        for per in range(1, 6):
            #Principal Components
            x_train, x_test, y_train, y_test = train_test_split(X_normalized.T, y_filtered.T, test_size=int(round(X_normalized.shape[1]*0.4)), random_state=per)
            PDM = PCA_cal_projections(x_train, k)
            PDM = np.asarray(PDM)
            TM = ((np.linalg.inv(PDM.T @ PDM)) @ (PDM.T)).T

            #Projected Data Matrix
            Q_train = (np.asarray(x_train) @ TM)
            Q_test = (np.asarray(x_test) @ TM)

            #Classification
            knc.fit(np.real(Q_train), y_train)
            y_predict = knc.predict(np.real(Q_test))
        
            #K-Fold Metrics
            accuracy += accuracy_score(y_test, y_predict)
            precision += precision_score(y_test, y_predict, average='macro')
            recall += recall_score(y_test, y_predict, average='macro')
            f1 += f1_score(y_test, y_predict, average='macro')
        accuracylist.append(accuracy / 5)
        precisionlist.append(precision / 5)
        recalllist.append(recall / 5)
        f1list.append(f1 / 5)
        print('---------------------------------------')
    print('PCA ACC:', np.mean(accuracylist))
    print('PCA REC:', np.mean(recalllist))
    print('PCA PRE:', np.mean(precisionlist))
    print('PCA F1:', np.mean(f1list))


    print('---------------NMF----------------')
    accuracylist.clear()
    precisionlist.clear()
    recalllist.clear()
    f1list.clear()
    for k in [1,10,20,30,40,50,60,70,80,90,100]:
        print('k = ', k)
        accuracy = 0
        precision = 0
        recall = 0
        f1 = 0
        for per in range(1, 6):
            #Train-Test Split
            x_train, x_test, y_train, y_test = train_test_split(X_filtered.T, y_filtered.T, test_size=int(round(X_normalized.shape[1]*0.4)), random_state=per)

            #Reduced Data Matrix
            Q_train = computeNMF(np.asarray(x_train), k)
            Q_test = computeNMF(np.asarray(x_test), k)

            #Classification
            knc.fit(np.real(Q_train), y_train)
            y_predict = knc.predict(np.real(Q_test))
        
            #K-Fold Metrics
            accuracy += accuracy_score(y_test, y_predict)
            precision += precision_score(y_test, y_predict, average='macro')
            recall += recall_score(y_test, y_predict, average='macro')
            f1 += f1_score(y_test, y_predict, average='macro')
        accuracylist.append(accuracy / 5)
        precisionlist.append(precision / 5)
        recalllist.append(recall / 5)
        f1list.append(f1 / 5)
        print('---------------------------------------')
    print('NMF ACC:', np.mean(accuracylist))
    print('NMF REC:', np.mean(recalllist))
    print('NMF PRE:', np.mean(precisionlist))
    print('NMF F1:', np.mean(f1list))


    print('---------------tSNE----------------')
    accuracylist.clear()
    precisionlist.clear()
    recalllist.clear()
    f1list.clear()
    for k in [1,10,20,30,40,50,60,70,80,90,100]:
        print('k = ', k)
        accuracy = 0
        precision = 0
        recall = 0
        f1 = 0
        for per in range(1, 6):
            #Train-Test Split
            x_train, x_test, y_train, y_test = train_test_split(X_filtered.T, y_filtered.T, test_size=int(round(X_normalized.shape[1]*0.4)), random_state=per)

            #Reduced Data Matrix
            Q_train = tSNE(np.asarray(x_train), k)
            Q_test = tSNE(np.asarray(x_test), k)

            #Classification
            knc.fit(np.real(Q_train), y_train)
            y_predict = knc.predict(np.real(Q_test))
        
            #K-Fold Metrics
            accuracy += accuracy_score(y_test, y_predict)
            precision += precision_score(y_test, y_predict, average='macro')
            recall += recall_score(y_test, y_predict, average='macro')
            f1 += f1_score(y_test, y_predict, average='macro')
        accuracylist.append(accuracy / 5)
        precisionlist.append(precision / 5)
        recalllist.append(recall / 5)
        f1list.append(f1 / 5)
        print('---------------------------------------')
    print('tSNE ACC:', np.mean(accuracylist))
    print('tSNE REC:', np.mean(recalllist))
    print('tSNE PRE:', np.mean(precisionlist))
    print('tSNE F1:', np.mean(f1list))


    print('---------------UMAP----------------')
    accuracylist.clear()
    precisionlist.clear()
    recalllist.clear()
    f1list.clear()
    for k in [1,10,20,30,40,50,60,70,80,90,100]:
        print('k = ', k)
        accuracy = 0
        precision = 0
        recall = 0
        f1 = 0
        for per in range(1, 6):
            #Train-Test Split
            x_train, x_test, y_train, y_test = train_test_split(X_filtered.T, y_filtered.T, test_size=int(round(X_normalized.shape[1]*0.4)), random_state=per)

            #Reduced Data Matrix
            Q_train = umap(np.asarray(x_train).T, k)
            Q_test = umap(np.asarray(x_test).T, k)

            #Classification
            knc.fit(np.real(Q_train), y_train)
            y_predict = knc.predict(np.real(Q_test))
        
            #K-Fold Metrics
            accuracy += accuracy_score(y_test, y_predict)
            precision += precision_score(y_test, y_predict, average='macro')
            recall += recall_score(y_test, y_predict, average='macro')
            f1 += f1_score(y_test, y_predict, average='macro')
        accuracylist.append(accuracy / 5)
        precisionlist.append(precision / 5)
        recalllist.append(recall / 5)
        f1list.append(f1 / 5)
        print('---------------------------------------')
    print('UMAP ACC:', np.mean(accuracylist))
    print('UMAP REC:', np.mean(recalllist))
    print('UMAP PRE:', np.mean(precisionlist))
    print('UMAP F1:', np.mean(f1list))

 

