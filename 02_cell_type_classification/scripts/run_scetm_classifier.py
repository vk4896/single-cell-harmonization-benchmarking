import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import os
import anndata as ann

warnings.filterwarnings('ignore')
sc.settings.verbosity = 3

from models.scETM import run_scetm_integration


print("reading the data")
combined_data = sc.read_h5ad("combined_data.h5ad")
print(combined_data.shape)


print("running the scETM integration")
run_scetm_integration(combined_data,dataset_key='batch',celltype_key='cell_type_level1',n_epochs=1200)
print("done")

print("saving the data")
combined_data.write_h5ad("combined_data.h5ad")
print("done")

print("saving the scETM as the X obs values in the combined_data.h5ad file")
print("making a new h5ad file with scETM data with unharmonized data obs values")
combined_unharm_scetm = ann.AnnData(
    X = combined_data.obsm['X_scetm'],
    obs = combined_data.obs.copy(),
    var = pd.DataFrame(index=[f"scETM_{i}" for i in range(combined_data.obsm['X_scetm'].shape[1])]),
    uns = combined_data.uns.copy()
)

print("saving the data")
combined_unharm_scetm.write_h5ad("combined_unharm_scetm.h5ad")
print("done")