import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import os
import anndata as ann

from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
warnings.filterwarnings('ignore')
sc.settings.verbosity = 3

def kmeans_clustering(adata,n_clusters =10,embeddings='X_pca'):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    adata.obs['cluster'] = kmeans.fit_predict(adata.obsm[embeddings])
    return adata.obs['cluster'].values


print("reading the data")
combined_unhar = sc.read_h5ad("combined_data.h5ad")
combined_unhar_scetm = sc.read_h5ad("combined_unharm_scetm.h5ad")
combined_har = sc.read_h5ad("combined_har_data.h5ad")
print("Data imported")


combined_unhar = combined_unhar[:780000]
combined_unhar_scetm = combined_unhar_scetm[:780000]


emdeddings = [ 'X_pca', 'X_pca_harmony', 'X_scanorama', 'X_scvi']

results = {}
for embedding in emdeddings:
    print(f"running the kmeans clustering on {embedding}")
    kmeans_clustering(combined_unhar,n_clusters=10,embeddings=embedding)
    results[embedding] = combined_unhar.obs['cluster'].values
    print("done")

#save the results
print("saving the results")
pd.DataFrame(results).to_csv("results_unhar.csv")


results = {}
for embedding in emdeddings:
    print(f"running the kmeans clustering on {embedding}")
    kmeans_clustering(combined_unhar_scetm,n_clusters=10,embeddings=embedding)
    results[embedding] = combined_unhar_scetm.obs['cluster'].values
    print("done")

#save the results
print("saving the results")
pd.DataFrame(results).to_csv("results_unhar_scetm.csv")


results = {}
kmeans_clustering(combined_har,n_clusters=10,embeddings='X_pca')
results['X_pca'] = combined_har.obs['cluster'].values
#save the results
print("saving the results")
pd.DataFrame(results).to_csv("results_har.csv")