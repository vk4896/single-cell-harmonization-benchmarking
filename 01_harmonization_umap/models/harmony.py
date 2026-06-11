import scanpy as sc
import pandas as pd
import harmonypy

def run_harmony_integration(adata, batch_key='dataset_id', target_sum=1e4, n_pcs=50, theta=2, max_iter_harmony=20):
    
    # Batch labels
    batch_labels = adata.obs[batch_key].values
    meta_data = pd.DataFrame({batch_key: batch_labels})
    
    # Run Harmony
    harmony_out = harmonypy.run_harmony(
        adata.obsm['X_pca'],
        meta_data,
        vars_use=[batch_key],
        theta=theta,
        max_iter_harmony=max_iter_harmony
    )
    
    # Store corrected PCs
    adata.obsm['X_pca_harmony'] = harmony_out.Z_corr.T
    
    return adata.obsm['X_pca_harmony']
