import scanpy as sc
import scanorama

def run_scanorama_integration(adata, batch_key='dataset_id'):
    # Split into batches
    batches = adata.obs[batch_key].unique().tolist()
    adatas_batch = [
        adata[adata.obs[batch_key] == b].copy()
        for b in batches
    ]
    
    # Run Scanorama correction
    adatas_integrated = scanorama.correct_scanpy(adatas_batch, return_dimred=True)
    
    # Re-assign batch labels
    for ad, b in zip(adatas_integrated, batches):
        ad.obs[batch_key] = b
    
    # Concatenate back into single AnnData
    adata_scanorama = sc.concat(adatas_integrated, index_unique="-")
    
    return adata_scanorama.obsm['X_scanorama']
