import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import os
import anndata as ann


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

warnings.filterwarnings('ignore')
np.random.seed(42)
sc.settings.verbosity = 3
print("Reading data")
combined_unharm = sc.read_h5ad("combined_unharm.h5ad")
combined_harm = sc.read_h5ad("combined_harm.h5ad")

#making the cells types proper order
# Example mapping
print("making the cells types proper order")
mapping = {
    "B_cell": "B_cell",
    "b_cell": "B_cell",
    "T_cell": "T_cell",
    "t_cell": "T_cell"
    # add others if needed
}

combined_unharm.obs["cell_type_level1"] = combined_unharm.obs["cell_type_level1"].replace(mapping)
combined_harm.obs["cell_type_level1"] = combined_harm.obs["cell_type_level1"].replace(mapping)

combined_unharm.write("combined_unharm.h5ad")
combined_harm.write("combined_harm.h5ad")
print("done")

print("making a new h5ad file with scETM data with unharmonized data obs values")
combined_unharm_scetm = ann.AnnData(
    X = combined_unharm.obsm['X_scetm'],
    obs = combined_unharm.obs.copy(),
    var = pd.DataFrame(index=[f"scETM_{i}" for i in range(combined_unharm.obsm['X_scetm'].shape[1])]),
    uns = combined_unharm.uns.copy()
)

combined_unharm_scetm.write("combined_unharm_scetm.h5ad")
print("done")


print("downsampling the unharmonized data")
keep_indices = []
for b in combined_unharm.obs["batch"].unique():
    batch_indices = np.where(combined_unharm.obs["batch"] == b)[0]
    n_keep = len(batch_indices) // 2   # take half from this batch
    sampled = np.random.choice(batch_indices, n_keep, replace=False)
    keep_indices.extend(sampled)

print("downsampling the unharmonized data")
combined_unharm_sampled = combined_unharm[keep_indices].copy()
combined_unharm_sampled.write("combined_unharm_sampled.h5ad")
print("combined_unharm_sampled.h5ad file created")


print("downsampling the harmonized data")
combined_harm_sampled = combined_harm[keep_indices].copy()
combined_harm_sampled.write("combined_harm_sampled.h5ad")
print("combined_harm_sampled.h5ad file created")


print("downsampling the scETM data")
combined_unharm_scetm_sampled = combined_unharm_scetm[keep_indices].copy()
combined_unharm_scetm_sampled.write("combined_unharm_scetm_sampled.h5ad")
print("combined_unharm_scetm_sampled.h5ad file created")


print("running UMAP for unharmonized data")
run_and_plot_umap(combined_unharm_sampled, "X_pca", save_dir="umap_plots", title_prefix="unharmonized")
print("done")
print("running UMAP for harmonized data")
run_and_plot_umap(combined_harm_sampled, "X_pca", save_dir="umap_plots", title_prefix="harmonized")
print("done")
print("running UMAP for scETM data")
run_and_plot_umap(combined_unharm_scetm_sampled, "X_scetm", save_dir="umap_plots", title_prefix="unharmonized_scETM")
print("done")