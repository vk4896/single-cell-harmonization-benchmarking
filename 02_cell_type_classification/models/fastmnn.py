import scanpy as sc
import numpy as np

def run_fastmnn_integration(adata, batch_key='batch'):
    """
    Run fastMNN integration on AnnData object.
    
    Parameters:
    -----------
    adata : AnnData
        Annotated data object
    batch_key : str
        Key in adata.obs that contains batch information
        
    Returns:
    --------
    np.ndarray
        Integrated data matrix stored in obsm['X_fastmnn']
    """
    import scib
    
    # Create a copy for integration
    adata_integrated = adata.copy()
    
    # Run fastMNN using scib
    scib.ig.mnn(adata_integrated, batch=batch_key)
    
    # Get the integrated data (scib stores it in X_pca)
    integrated_data = adata_integrated.obsm['X_pca']
    
    return integrated_data

