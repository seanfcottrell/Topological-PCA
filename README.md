# Topological-PCA
Topological PCA for scRNA-seq dimensionality reduction

This work builds upon previous improvements to PCA, most notably the PLPCA procedure of Cottrell et. al for Microarray data analysis, by introducing the Persistent Laplacian for scRNA-seq dimensionality reduction. The Persistent Laplacian is induced via a filtration procedure on the Weighted Graph Laplacian, which is analogous to the Vietoris Rips complex. This term then captures local topological geometrical structure information through Persistent Spectral Graphs. We also introduce a new algebraic topology tool known as the kNN Persistent Laplacian, where we induce filtration by varying the number of nearest neighbors at each node in our graph structure. 


---

## MODEL:

Included here are several files for different PCA methods that can be used to compare relative clustering, classification, and visualization performances on different datasets (after dimensionality reduction). Included are: PCA, sPCA, RgLSPCA, and tPCA, as well as NMF, UMAP, and tSNE. 

Paramvals provides parameter values used to recreate published results. 

## DATASETS: 

Datasets can be obtained from the Single Cell Data Processing repository referenced in the paper.  

## REFERENCES: 

I.) S. Cottrell, R. Wang, G. Wei. PLPCA: Persistent Laplacian-Enhanced PCA for Microarray Data Analysis. 2023.

II.) R. Wang, D. D. Nguyen, and G.-W. Wei. Persistent spectral graph. International Journal for Numerical Methods in Biomedical Engineering, page e3376, 2020.

## CITING:

You may use the following bibtex entry to cite tPCA:

...

## INSTALLATION: 

You can install PLPCA directly from the repo in your terminal via: 

```bash
git clone https://github.com/seanfcottrell/Topological-PCA.git
```
