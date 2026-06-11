import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from sklearn.cluster import KMeans


warnings.filterwarnings('ignore')
np.random.seed(42)
sc.settings.verbosity = 3
print("imported all the libraries")
unharmonized_data = sc.read_h5ad("downloaded_adata_diseases_unharmonized.h5ad")
harmonized_data = sc.read_h5ad("downloaded_adata_diseases_harmonized.h5ad")
parse = sc.read_h5ad("adata_parse.h5ad")

print("read the data")


print("getting the true labels for later use")
true_labels = unharmonized_data.obs['perturbation']
true_labels.to_csv("classifer/true_labels.csv")
print("wrote the true labels to a csv file")

print("handling the var_gene names")
unharmonized_data.var_names = [str(i) for i in range(unharmonized_data.n_vars)]
parse.var_names = [str(i) for i in range(parse.n_vars)]
harmonized_data.var_names = [str(i) for i in range(harmonized_data.n_vars)]

unharmonized_data.obs['batch'] = 'batch1'
harmonized_data.obs['batch'] = 'batch1'
parse.obs['batch'] = 'batch2'

print("concatenated the data")
combined_data = sc.concat([unharmonized_data,parse], axis=0,join='inner')
combined_har_data = sc.concat([harmonized_data,parse], axis=0,join='inner')

print("concatenated the data")

print("normalized the data for unharmonized data")
sc.pp.normalize_total(combined_data, target_sum=1e4)
sc.pp.log1p(combined_data)
sc.tl.pca(combined_data, n_comps=50)

print("normalized the data for harmonized data")
sc.pp.normalize_total(combined_har_data, target_sum=1e4)
sc.pp.log1p(combined_har_data)
sc.tl.pca(combined_har_data, n_comps=50)


print("saving the data")
combined_data.write_h5ad("classifer/combined_data.h5ad")
combined_har_data.write_h5ad("classifer/combined_har_data.h5ad")