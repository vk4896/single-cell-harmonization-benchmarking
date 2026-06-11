import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import os

warnings.filterwarnings('ignore')
np.random.seed(42)
sc.settings.verbosity = 3   

def run_and_plot_umap(
    adata,
    use_rep,
    color=["batch"],
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