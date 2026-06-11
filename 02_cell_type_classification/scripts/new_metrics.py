import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import normalized_mutual_info_score, silhouette_samples
from sklearn.neighbors import NearestNeighbors
print("Importing libraries...")
print("Reading data...")
data = sc.read_h5ad("combined_data.h5ad")
data_harm = sc.read_h5ad("combined_har_data.h5ad")
data_scetm = sc.read_h5ad("combined_unharm_scetm.h5ad")

data = data[data.obs['batch'] == 'batch1']
print(data.shape)
data_harm = data_harm[data_harm.obs['batch'] == 'batch1']
print(data_harm.shape)
data_scetm = data_scetm[data_scetm.obs['batch'] == 'batch1']
print(data_scetm.shape)

og = sc.read_h5ad("../downloaded_adata_diseases_unharmonized.h5ad")


data.obs['dataset_id'] = og.obs['dataset_id']
data_harm.obs['dataset_id'] = og.obs['dataset_id']
data_scetm.obs['dataset_id'] = og.obs['dataset_id']

print("Data read successfully")
print('sample')
data = data[:10000]
data_harm = data_harm[:10000]
data_scetm = data_scetm[:10000]

def compute_nmi(adata, cluster_key, label_key='cell_types'):
    clusters = adata.obs[cluster_key].astype(str)
    labels = adata.obs[label_key].astype(str)
    return normalized_mutual_info_score(labels, clusters)


def compute_ilisi(adata, batch_key='dataset_id', use_rep='X_pca', n_neighbors=30):
    X = adata.obsm[use_rep]
    batches = adata.obs[batch_key].astype('category').cat.codes.values

    nn = NearestNeighbors(n_neighbors=n_neighbors+1).fit(X)
    idx = nn.kneighbors(return_distance=False)[:, 1:]

    ilis = []
    for i in range(idx.shape[0]):
        neigh_batches = batches[idx[i]]
        _, counts = np.unique(neigh_batches, return_counts=True)
        probs = counts / counts.sum()
        isi = 1.0 / np.sum(probs ** 2)
        ilis.append(isi)
    return np.mean(ilis)



def compute_asw(adata, label_key='cell_types', use_rep='X_pca'):
    X = adata.obsm[use_rep]
    labels = adata.obs[label_key].astype('category').cat.codes.values
    sil = silhouette_samples(X, labels, metric='euclidean')
    return np.mean(sil)


integration_keys = ['X_pca', 'X_pca_harmony', 'X_scanorama', 'X_scetm', 'X_scvi']



results = []
print("Evaluating Unharmonized data")
# Unharmonized data
for rep in integration_keys:
    print(f"\n🔹 Evaluating {rep} ...")

    # 1️⃣ Build neighbors & Leiden clusters for this embedding
    sc.pp.neighbors(data, use_rep=rep)
    leiden_key = f'leiden_{rep}'
    sc.tl.leiden(data, resolution=1.0, key_added=leiden_key)

    # 2️⃣ Compute metrics
    nmi = compute_nmi(data, cluster_key=leiden_key, label_key='cell_types')
    ilisi = compute_ilisi(data, batch_key='dataset_id', use_rep=rep)
    asw_cell = compute_asw(data, label_key='cell_types', use_rep=rep)
    asw_batch = compute_asw(data, label_key='dataset_id', use_rep=rep)

    results.append({
        'Embedding Unharmonized': rep,
        'NMI Unharmonized': nmi,
        'Graph_iLISI Unharmonized': ilisi,
        'ASW_celltype Unharmonized': asw_cell,
        'ASW_batch Unharmonized': asw_batch
    })


df_results = pd.DataFrame(results)
print("\n📊 Integration Evaluation Summary:\n")
print(df_results.round(3))

df_results.to_csv("integration_evaluation_summary_unharm.csv", index=False)
print("Integration evaluation summary saved to integration_evaluation_summary.csv")
# scetm data
print("Evaluating scetm data")
results = []
integration_keys = ['X_pca', 'X_pca_harmony', 'X_scanorama', 'X_scvi']
for rep in integration_keys:
    print(f"\n🔹 Evaluating {rep} ...")
    sc.pp.neighbors(data_scetm, use_rep=rep)
    leiden_key = f'leiden_{rep}'
    sc.tl.leiden(data_scetm, resolution=1.0, key_added=leiden_key)
    nmi = compute_nmi(data_scetm, cluster_key=leiden_key, label_key='cell_types')
    ilisi = compute_ilisi(data_scetm, batch_key='dataset_id', use_rep=rep)
    asw_cell = compute_asw(data_scetm, label_key='cell_types', use_rep=rep)
    asw_batch = compute_asw(data_scetm, label_key='dataset_id', use_rep=rep)
    results.append({
        'Embedding scetm': rep,
        'NMI scetm': nmi,
        'Graph_iLISI scetm': ilisi,
        'ASW_celltype scetm': asw_cell,
        'ASW_batch scetm': asw_batch
    })
    
df_results = pd.DataFrame(results)
print("\n📊 Integration Evaluation Summary:\n")
print(df_results.round(3))

df_results.to_csv("integration_evaluation_summary_scetm.csv", index=False)
print("Integration evaluation summary saved to integration_evaluation_summary.csv")

# Harmonized data
results = []
print("Evaluating Harmonized data")
integration_keys = ['X_pca']
for rep in integration_keys:
    print(f"\n🔹 Evaluating {rep} ...")
    sc.pp.neighbors(data_harm, use_rep=rep)
    leiden_key = f'leiden_{rep}'
    sc.tl.leiden(data_harm, resolution=1.0, key_added=leiden_key)
    nmi = compute_nmi(data_harm, cluster_key=leiden_key, label_key='cell_type_level1')
    ilisi = compute_ilisi(data_harm, batch_key='dataset_id', use_rep=rep)
    asw_cell = compute_asw(data_harm, label_key='cell_type_level1', use_rep=rep)
    asw_batch = compute_asw(data_harm, label_key='dataset_id', use_rep=rep)
    results.append({
        'Embedding Harmonized': rep,
        'NMI Harmonized': nmi,
        'Graph_iLISI Harmonized': ilisi,
        'ASW_celltype Harmonized': asw_cell,
        'ASW_batch Harmonized': asw_batch
    }) 


df_results = pd.DataFrame(results)
print("\n📊 Integration Evaluation Summary:\n")
print(df_results.round(3))

df_results.to_csv("integration_evaluation_summary_harm.csv", index=False)
print("Integration evaluation summary saved to integration_evaluation_summary.csv")