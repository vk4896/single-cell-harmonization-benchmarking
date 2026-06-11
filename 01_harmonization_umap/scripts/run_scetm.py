import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import os

warnings.filterwarnings('ignore')
np.random.seed(42)
sc.settings.verbosity = 3

# Instead of set_figure_params, configure matplotlib manually
# plt.rcParams['figure.dpi'] = 100
# plt.rcParams['figure.facecolor'] = 'white'
# plt.rcParams['savefig.dpi'] = 100


from models.scETM import run_scetm_integration
print("Models imported")


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




print("Reading data")
unharmonized_data = sc.read_h5ad("../downloaded_adata_diseases_unharmonized.h5ad")
print("Unharmonized data imported")
harmonized_data = sc.read_h5ad("../downloaded_adata_diseases_harmonized.h5ad")
print("Harmonized data imported")

print("Unharmonized data shape: ", unharmonized_data.shape)
print("Harmonized data shape: ", harmonized_data.shape)

print("Unharmonized data: ", unharmonized_data)
print("Harmonized data: ", harmonized_data)
assert all(unharmonized_data.obs_names == harmonized_data.obs_names), "Obs indices do not match!"
print("asserted")
parse_data = sc.read_h5ad("../adata_parse.h5ad")
print("Parse data imported")
print("Parse data shape: ", parse_data.shape)
print("Parse data: ", parse_data)
n_per_group = 25000
sampled_idx = (
    unharmonized_data.obs.groupby("perturbation", group_keys=False)
    .apply(lambda x: x.sample(n_per_group, random_state=42))
    .index
)

unharmonized_data_sampled = unharmonized_data[sampled_idx].copy()
harmonized_data_sampled = harmonized_data[sampled_idx].copy()

# Check shapes
print("Shapes:", unharmonized_data_sampled.shape, harmonized_data_sampled.shape)
print("Perturbation counts:\n", unharmonized_data_sampled.obs["perturbation"].value_counts())
print("Perturbation counts:\n", harmonized_data_sampled.obs["perturbation"].value_counts())


assert all(unharmonized_data_sampled.obs_names == harmonized_data_sampled.obs_names), "Obs indices do not match!"
print("asserted")


print("checking the cell types and their counts")
print("Unharmonized data cell types: ", unharmonized_data_sampled.obs["cell_type_level1"].value_counts())
print("Harmonized data cell types: ", harmonized_data_sampled.obs["cell_type_level1"].value_counts())


n_cells = unharmonized_data_sampled.n_obs
print(f"Sampling {n_cells} cells from parse_data (original {parse_data.n_obs})")

parse_sampled = parse_data[np.random.choice(parse_data.obs_names, n_cells, replace=False)].copy()

print("adding batch information")
unharmonized_data_sampled.obs["batch"] = "batch1"
harmonized_data_sampled.obs["batch"]   = "batch1"
parse_sampled.obs["batch"]= "batch2"

print("renaming the gene names")
unharmonized_data_sampled.var_names = [str(i) for i in range(unharmonized_data_sampled.n_vars)]
harmonized_data_sampled.var_names   = [str(i) for i in range(harmonized_data_sampled.n_vars)]
parse_sampled.var_names        = [str(i) for i in range(parse_sampled.n_vars)]

print("concatenating the data")
combined_unharm = sc.concat([parse_sampled, unharmonized_data_sampled], axis=0, join="inner")
combined_harm   = sc.concat([parse_sampled, harmonized_data_sampled], axis=0, join="inner")

print("combined_unharm shape: ", combined_unharm.shape)
print("combined_harm shape: ", combined_harm.shape)

print("combined_unharm: ", combined_unharm)
print("combined_harm: ", combined_harm)

print("writing the data")
combined_unharm.write("combined_unharm.h5ad")
combined_harm.write("combined_harm.h5ad")

print("checking the batch and cell type information")
print(combined_unharm.obs["batch"].value_counts())
print(combined_harm.obs["batch"].value_counts())

print(combined_unharm.obs["cell_type_level1"].value_counts())
print(combined_harm.obs["cell_type_level1"].value_counts())

print('checking the data')
print(combined_unharm)
print(combined_harm)

# pca calculation
print("calculating the pca for unharmonized data")
sc.pp.normalize_total(combined_unharm, target_sum=1e4)
sc.pp.log1p(combined_unharm)
sc.pp.pca(combined_unharm,n_comps=50)

print("calculating the pca for harmonized data")
sc.pp.normalize_total(combined_harm, target_sum=1e4)
sc.pp.log1p(combined_harm)
sc.pp.pca(combined_harm,n_comps=50)

print("combined_data along with pca obsm")
print(combined_unharm)
print(combined_harm)

print("writing the data")
combined_unharm.write("combined_unharm.h5ad")
combined_harm.write("combined_harm.h5ad")
print("done")


print("running the umap")
run_and_plot_umap(combined_unharm, "X_pca", ["cell_type_level1", "batch"],title_prefix="unharmonized")
run_and_plot_umap(combined_harm, "X_pca", ["cell_type_level1", "batch"],title_prefix="harmonized")
print("done")

print("running sctem model on unharmonized data")
run_scetm_integration(combined_unharm,dataset_key='batch',celltype_key='cell_type_level1',n_epochs=1200)
print("done")


print("writing the data")
combined_unharm.write("combined_unharm.h5ad")

print("plotting the umap for unharmonized data with sctem")
run_and_plot_umap(combined_unharm, "X_scetm", ["cell_type_level1", "batch"],title_prefix="unharmonized_sctem")
print("done")
