import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import os
import anndata as ann

warnings.filterwarnings('ignore')
np.random.seed(42)
sc.settings.verbosity = 3

from models.scVI_model import run_scvi_integration

print("Models imported")

def run_and_plot_umap(
    adata,
    use_rep,
    color=["cell_type_level1","batch"],
    save_dir="umap_plots",
    title_prefix=None,
    figsize=(10, 8),
    dpi=200,
    dot_size=3   # <-- new parameter for dot size
):
    os.makedirs(save_dir, exist_ok=True)

    # Compute neighbors + UMAP
    sc.pp.neighbors(adata, use_rep=use_rep)
    sc.tl.umap(adata)

    # Defaults
    title_prefix = title_prefix if title_prefix else "UMAP"

    # Create figure
    plt.figure(figsize=figsize)
    sc.pl.umap(
        adata,
        color=color,
        title=f"{title_prefix} from {use_rep}",
        show=False,
        frameon=True,
        s=dot_size  # set point size
    )

    # Save
    save_path = os.path.join(save_dir, f"{title_prefix}.png")
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"Saved {save_path}")

print("Reading data")
combined_unharm = sc.read_h5ad("combined_unharm_sampled.h5ad")
combined_unharm_scetm = sc.read_h5ad("combined_unharm_scetm_sampled.h5ad")
print("Data imported")

print("Running scVI on unharmonized data")
run_scvi_integration(combined_unharm,batch_key='batch',max_epochs=200)
print("done")
combined_unharm.write("combined_unharm_sampled.h5ad")

print("Running scVI on scETM data")
run_scvi_integration(combined_unharm_scetm,batch_key='batch',max_epochs=200)
print("done")
combined_unharm_scetm.write("combined_unharm_scetm_sampled.h5ad")

print("plotting the umap for unharmonized data with scVI")
run_and_plot_umap(combined_unharm, "X_scvi", ["cell_type_level1", "batch"],title_prefix="unharmonized_scVI")
print("done")

print("plotting the umap for scETM data with scVI")
run_and_plot_umap(combined_unharm_scetm, "X_scvi", ["cell_type_level1", "batch"],title_prefix="unharmonized_scetm_scVI")
print("done")