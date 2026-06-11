import scvi
import torch

def run_scvi_integration(adata, batch_key="dataset_id", max_epochs=200, early_stopping=True, use_gpu=True):
    """
    Perform scVI batch correction on an AnnData object.
    """
    
    # Setup scVI
    scvi.model.SCVI.setup_anndata(adata, batch_key=batch_key)
    
    # Initialize model
    model = scvi.model.SCVI(adata)
    
    # Configure GPU / CPU
    if use_gpu and torch.cuda.is_available():
        accelerator = "gpu"
        devices = 1
        print("Training scVI using **GPU**")
    else:
        accelerator = "cpu"
        devices = 1
        print("Training scVI using **CPU** (GPU not available or disabled)")
    
    # Train model
    model.train(
        max_epochs=max_epochs,
        early_stopping=early_stopping,
        accelerator=accelerator,
        devices=devices
    )
    
    # Save latent representation
    adata.obsm["X_scvi"] = model.get_latent_representation()
    
    return adata.obsm["X_scvi"]
