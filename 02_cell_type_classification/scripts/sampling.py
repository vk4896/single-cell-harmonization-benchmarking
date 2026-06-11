import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import os
import anndata as ann

warnings.filterwarnings('ignore')
sc.settings.verbosity = 3


print("reading the data")

combined_har_data = sc.read_h5ad("combined_har_data.h5ad")

combined_har_data = combined_har_data[:780000]

combined_har_data.write_h5ad("combined_har_data.h5ad")
print("saved the data")