import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import os
import anndata as ann

warnings.filterwarnings('ignore')
sc.settings.verbosity = 3

from models.harmony import run_harmony_integration

print("Models imported")

print("reading the data")
combined_data = sc.read_h5ad("combined_data.h5ad")
combined_data_scetm = sc.read_h5ad("combined_unharm_scetm.h5ad")
print("Data imported")

print("running the harmony integration on unharmonized data") 
run_harmony_integration(combined_data,batch_key='batch')      
print("done")

combined_data.write_h5ad("combined_data.h5ad")

print("running the harmony integration on unharmonized data_scetm") 
run_harmony_integration(combined_data_scetm,batch_key='batch')      
print("done")

combined_data_scetm.write_h5ad("combined_unharm_scetm.h5ad")