import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import os
import anndata as ann
from pandas import Categorical
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score


original_labels = pd.read_csv("true_labels.csv")
original_labels['perturbation'].astype(Categorical)
original_labels['perturbation_code'] = original_labels['perturbation'].cat.codes
true_labels = original_labels['perturbation_code'].values

results = pd.read_csv("results.csv")


# nmi_score = normalized_mutual_info_score(true_labels, pred_labels)
# ari_score = adjusted_rand_score(true_labels, pred_labels)