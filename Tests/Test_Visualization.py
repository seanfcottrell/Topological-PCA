import numpy as np
import pandas as pd
import warnings
import os
import sys
from tPCA import RpLSPCA_cal_projections_KNN
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn import manifold
from matplotlib import ticker
import umap

warnings.filterwarnings("ignore")


def plot_colored_scatter(points, labels, title, filename):
    fig, ax = plt.subplots(figsize=(6, 6), facecolor="white", constrained_layout=True)
    fig.suptitle(title, size=16)

    plt.xticks([])
    plt.yticks([])

    unique_labels = np.unique(labels)
    
    label_colors = {
        0: 'purple',
        1: 'blue',
        2: 'cyan',
        3: 'green',
        4: 'orange',
        5: 'red',
        6: 'magenta',
        7: 'yellow'
    }
    
    for label in unique_labels:
        mask = labels == label
        color = label_colors.get(label, 'gray')  # Default to gray if label not in dictionary
        ax.scatter(points[mask, 0], points[mask, 1], c=color, s=50, alpha=0.8, label=f'Label {label}')

    #ax.set_title(title)
    #ax.legend(loc="upper left", bbox_to_anchor=(1, 1))

    plt.savefig(filename)
    plt.show()

def load_X(data):
    inpath = '/Users/seancottrell/tPCA_Workshop/data/%s/'%(data)
    X = pd.read_csv(inpath + '%s_full_X.csv'%(data))
    X = X.values[:, 1:].astype(float)
    return X

def load_y(data):
    inpath = '/Users/seancottrell/tPCA_Workshop/data/%s/'%(data)
    y = pd.read_csv(inpath + '%s_full_labels.csv'%(data))
    y = np.array(list(y['Label'])).astype(int)
    return y

if __name__ == '__main__':
    X = load_X('GSE45719')
    y = load_y('GSE45719')
    #print(X)

    # Log transform data
    log_transform = np.vectorize(np.log)
    log_X = log_transform(X+1)
    #print(log_X)
    
    # Set values below 1e-6 to 0
    log_X[log_X < 1e-6] = 0

    #print('X shape:', X.shape)
    #print("y shape:", y.shape)
    
    # Filter out features with low variance
    row_variances = np.var(log_X, axis=1)
    variance_threshold = np.percentile(row_variances, 20)  # Adjust the percentile as needed
    filtered_X = log_X[row_variances >= variance_threshold]
    #print('Gene filtering X shape:', filtered_X.shape)
    filtered_X = np.asarray(filtered_X)
    
    # Filter out classes with fewer than 15 samples
    #filtered_X_transposed = filtered_X.T
    # Get the unique classes and their counts
    #unique_classes, class_counts = np.unique(y, return_counts=True)
    # Find the classes with less than 15 samples
    #classes_to_remove = unique_classes[class_counts < 15]
    # Create a mask to filter the samples and labels
    #mask = np.isin(y, classes_to_remove, invert=True)
    # Filter the dataset and labels
    #X_filtered = filtered_X_transposed[mask].T
    #y_filtered = y[mask]
    # Print the filtered dataset and labels
    #print("Filtered X shape:", X_filtered.shape)
    #print("Filtered y shape:", y_filtered.shape)

    # Normalize data
    #scaler = StandardScaler()
    #scaler.fit(X_filtered)
    #X_normalized = scaler.transform(X_filtered)
    
    k = np.unique(y).shape[0]
    zetas = [1/8,1/7,1/6,1/5,1/4,1/3,1/2,1]
    gamma = 1000
    beta = 60

    #X_filtered= np.array(X_filtered)
    #X_normalized = np.array(X_normalized)

    #T_SNE Visualization
    '''
    n_components = 2

    t_sne = manifold.TSNE(
    n_components=n_components,
    perplexity=14,
    init="pca",
    n_iter=350,
    random_state=0,
    ) 
    
    #t-SNE PLOTS
    print('---------------RpLSPCA----------------')
    RpLSPCA_ari_mean = 0
    RpLSPCA_nmi_mean = 0
    #Principal Components
    PDM = RpLSPCA_cal_projections_KNN(X_normalized.T, beta, gamma, 50, 15, zetas)
    PDM = np.array(PDM)
    PDM = ((np.linalg.inv(PDM.T @ PDM)) @ (PDM.T)).T

    #Projected Data Matrix
    Q_PLPCA = (X_normalized.T @ PDM)  

    PLPCA_t_sne = t_sne.fit_transform(Q_PLPCA)
    plot_colored_scatter(PLPCA_t_sne, y_filtered, "RpLSPCA:T-SNE", 'GSE75748cell_tsne_PLPCA.pdf')

    print('----------No Reduction--------')
    NR_t_sne = t_sne.fit_transform(X_normalized.T)
    plot_colored_scatter(NR_t_sne, y_filtered, "T-SNE", 'GSE75748cell_tsne_NR.pdf')
    '''

    reducer = umap.UMAP()

    #UMAP PLOTS
    print('---------------UMAP----------------')
    '''
    RpLSPCA_ari_mean = 0
    RpLSPCA_nmi_mean = 0
    #Principal Components
    PDM = RpLSPCA_cal_projections_KNN(X_normalized.T, beta, gamma, 50, 15, zetas)
    PDM = np.array(PDM)
    PDM = ((np.linalg.inv(PDM.T @ PDM)) @ (PDM.T)).T

    #Projected Data Matrix
    Q_PLPCA = (X_normalized.T @ PDM)  

    PLPCA_t_sne = t_sne.fit_transform(Q_PLPCA)
    plot_colored_scatter(PLPCA_t_sne, y_filtered, "RpLSPCA:T-SNE", 'GSE75748cell_tsne_PLPCA.pdf')
    '''
    print('----------No Reduction--------')
    NR_umap = reducer.fit_transform(filtered_X.T)
    plot_colored_scatter(NR_umap, y, "UMAP GSE45719", 'GSE45719_umap.png')
    
