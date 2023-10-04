import numpy as np
import pandas as pd
import warnings
import os
import sys
from sklearn.neighbors import KNeighborsClassifier
from RgLSPCA import RgLSPCA_cal_projections
from RpLSPCA import RpLSPCA_cal_projections_KNN
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
from sklearn.preprocessing import StandardScaler

curPath = os.path.abspath(os.path.dirname(__file__)) #retrieve current directory name
rootPath = os.path.split(curPath)[0] #split and get just directory part
sys.path.append(rootPath) #add the parent directory of the current script to the list of paths where Python searches for modules
warnings.filterwarnings("ignore") # suppress warning messages


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

if __name__ == '__main__': #executed only when the script is run directly
    X = load_X('GSE67835')
    y = load_y('GSE67835')

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

    # Normalize data
    scaler = StandardScaler()
    scaler.fit(X_filtered)
    X_normalized = scaler.transform(X_filtered)
    
    '''GSE82187'''
    #k = np.unique(y_filtered).shape[0]
    #zeta = [0,0,1,1,0,0,1,1] #float(sys.argv[1])
    #zeta2 = [1/8,1/7,1/6,1/5,1/4,1/3,1/2,1]
    #gamma = 1000
    #beta = 100
    '''GSE67835'''
    #k = np.unique(y_filtered).shape[0]
    zeta = [1,1,1,0,1,1,1,1] #float(sys.argv[1])
    #zeta2 = [1/8,1/7,1/6,1/5,1/4,1/3,1/2,1]
    gamma = 10000
    beta = 10000

    X_filtered= np.asarray(X_filtered)
    X_normalized = np.asarray(X_normalized)
    y_filtered = np.asarray(y_filtered)

    knc = KNeighborsClassifier(n_neighbors=1)

    accuracylist = []
    precisionlist = []
    recalllist = []
    f1list = []

    print('---------------Optimized RpLSPCA KNN----------------')
    
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
        print('F1:', f1 / 5)
        print('ACC:', accuracy / 5)
        print('---------------------------------------')
    print('Optimized RpLSPCA KNN ACC:', np.mean(accuracylist))
    print('Optimized RpLSPCA KNN REC:', np.mean(recalllist))
    print('Optimized RpLSPCA KNN PRE:', np.mean(precisionlist))
    print('Optimized RpLSPCA KNN F1:', np.mean(f1list))

    #write_file = open('/mnt/home/cottre61/data/parameter_tuning_results/GSE75748time_Classification_scores.csv','a+')
    #write_file.write('Alpha1: %.3f, Alpha2: %.3f, Alpha3: %.3f, Alpha4: %.3f, Alpha5: %.3f, Alpha6: %.3f, Alpha7: %.3f, Alpha8: %.3f, Beta: %.3f, Gamma: %.3f, ACC: %.6f, PRE: %.6f, REC: %.6f, F1: %.6f\n'%(zeta[0], zeta[1],zeta[2], zeta[3],zeta[4],zeta[5],zeta[6],zeta[7],beta, gamma, np.mean(accuracylist),np.mean(precisionlist),np.mean(recalllist),np.mean(f1list)))
    #write_file.close()

    print('---------------RgLSPCA----------------')
    accuracylist = []
    precisionlist = []
    recalllist = []
    f1list = []
    for k in [1,10,20,30,40,50,60,70,80,90,100]:
        print('k = ', k)
        accuracy = 0
        precision = 0
        recall = 0
        f1 = 0
        for per in range(1, 6):
            #Principal Components
            x_train, x_test, y_train, y_test = train_test_split(X_normalized.T, y_filtered.T, test_size=int(round(X_normalized.shape[1]*0.4)), random_state=per)
            PDM = RgLSPCA_cal_projections(x_train, beta, gamma, k)
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
        print('F1:', f1 / 5)
        print('ACC:', accuracy / 5)
        print('---------------------------------------')
    print('RgLSPCA KNN ACC:', np.mean(accuracylist))
    print('RgLSPCA KNN REC:', np.mean(recalllist))
    print('RgLSPCA KNN PRE:', np.mean(precisionlist))
    print('RgLSPCA KNN F1:', np.mean(f1list))
