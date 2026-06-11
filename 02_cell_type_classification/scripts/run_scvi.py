import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import os
import anndata as ann

warnings.filterwarnings('ignore')
sc.settings.verbosity = 3

from models.scVI_model import run_scvi_integration

print("Models imported")

print("reading the data")
combined_data = sc.read_h5ad("combined_data.h5ad")
combined_unharm_data = sc.read_h5ad("combined_unharm_scetm.h5ad")
print("Data imported")

print("running the scVI integration on unharmonized data")
run_scvi_integration(combined_data,batch_key='batch',max_epochs=200)
print("done")

print("saving the data")
combined_data.write_h5ad("combined_data.h5ad")
print("done")

print("running the scVI integration on unharmonized data_scetm")
run_scvi_integration(combined_unharm_data,batch_key='batch',max_epochs=200)
print("done")

print("saving the data")
combined_unharm_data.write_h5ad("combined_unharm_scetm.h5ad")
print("done")