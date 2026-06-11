from scETM import scETM, UnsupervisedTrainer
import anndata
import scanpy as sc
import os

def run_scetm_integration(adata, dataset_key="dataset_id", celltype_key="cell_type_original",
                          n_epochs=1200, results_dir="./scetm_results"):
    """
    Run scETM integration and return updated AnnData with embeddings.
    """
    # Copy to avoid modifying original

    # Prepare metadata
    adata.obs['batch_indices'] = adata.obs[dataset_key].astype('category').cat.codes
    adata.obs['cell_types'] = adata.obs[celltype_key]

    # Init model
    model = scETM(
        adata.n_vars,
        adata.obs['batch_indices'].nunique(),
        enable_batch_bias=True
    )

    # Results dir
    os.makedirs(results_dir, exist_ok=True)

    # Trainer
    trainer = UnsupervisedTrainer(
        model, adata,
        train_instance_name="scETM_training",
        ckpt_dir=results_dir
    )

    # Train model
    trainer.train(n_epochs=n_epochs, eval_every=200, n_samplers=4, eval=False)

    # Extract cell embeddings (theta)
    model.get_cell_embeddings_and_nll(adata, emb_names=['theta'])
    adata.obsm['X_scetm'] = adata.obsm['theta']

    return adata.obsm['X_scetm']
