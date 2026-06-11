#!/usr/bin/env python3
"""
Compute UMAPs for different embeddings and plot them colored by cell cycle type.
Saves plots in a dedicated output folder.
"""

import os
import math
import numpy as np
import pandas as pd
import types
import scanpy as sc
import anndata as ad
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score

# TensorFlow 1.x compatibility for ccAF
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
tf.placeholder = tf.compat.v1.placeholder
tf.Session = tf.compat.v1.Session

import ccAF

# Fix ccAF bug (set used as index in prep)
def _fixed_prep_predict_data(self, data):
    missing = set(self.genes).difference(data.index)
    if len(missing) > 0:
        data = pd.concat([data, pd.DataFrame(0, index=list(missing), columns=data.columns)])
    return data.loc[list(self.genes)]

ccAF.ccAF._Classifier_ACTINN__prep_predict_data = types.MethodType(
    _fixed_prep_predict_data, ccAF.ccAF
)

print("ccAF loaded and fixed")


def build_ccaf_ann_from_embedding(embedding_matrix: np.ndarray, obs_index) -> ad.AnnData:
    """Create a ccAF-shaped AnnData using an embedding matrix as X."""
    ccaf_genes = list(ccAF.ccAF.genes)
    k = len(ccaf_genes)  # usually 1472
    n, d = embedding_matrix.shape
    if d >= k:
        Xk = embedding_matrix[:, :k]
    else:
        pad = np.zeros((n, k - d), dtype=embedding_matrix.dtype)
        Xk = np.hstack([embedding_matrix, pad])
    var_df = pd.DataFrame(index=ccaf_genes)
    ad_emb = ad.AnnData(X=Xk, obs=pd.DataFrame(index=obs_index), var=var_df)
    return ad_emb


def ccaf_predict_batched(adata_like, batch_size: int = 50000) -> np.ndarray:
    """Run ccAF predictions in batches."""
    n = adata_like.n_obs
    out = []
    steps = math.ceil(n / batch_size)
    print(f"Running ccAF predictions: {n} cells in {steps} batches")
    for i in range(steps):
        s = i * batch_size
        e = min((i + 1) * batch_size, n)
        print(f"  Batch {i+1}/{steps}: rows {s}:{e}")
        sub = adata_like[s:e, :].copy()
        out.extend(ccAF.ccAF.predict_labels(sub))
    return np.array(out)


def main():
    # Create output folder
    output_dir = "umap_cell_cycle_plots"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}/")
    
    # Number of cells to sample
    n_cells_to_sample = 30000
    
    # Load data
    print("\nLoading data...")
    adata = sc.read_h5ad('../combined_data.h5ad')
    print(f"Loaded: {adata.n_obs} cells, {adata.n_vars} genes")
    
    # Sample subset of cells
    if adata.n_obs > n_cells_to_sample:
        print(f"\nSampling {n_cells_to_sample} cells from {adata.n_obs} total cells...")
        np.random.seed(42)  # For reproducibility
        indices = np.random.choice(adata.n_obs, size=n_cells_to_sample, replace=False)
        adata = adata[indices].copy()
        print(f"Subset to {adata.n_obs} cells")
    else:
        print(f"\nUsing all {adata.n_obs} cells (less than requested {n_cells_to_sample})")
    
    # Get cell cycle predictions from X_pca
    print("\nGetting cell cycle predictions from X_pca...")
    X_pca = adata.obsm['X_pca']
    ad_pca = build_ccaf_ann_from_embedding(X_pca, obs_index=adata.obs_names)
    cell_cycle_predictions = ccaf_predict_batched(ad_pca, batch_size=50000)
    
    # Add as obs column
    adata.obs['cell_cycle'] = cell_cycle_predictions
    print(f"\nCell cycle predictions added to adata.obs['cell_cycle']")
    print(f"Unique cell cycle types: {adata.obs['cell_cycle'].unique()}")
    print(f"\nCell cycle distribution:")
    print(adata.obs['cell_cycle'].value_counts())
    
    # Compute UMAPs for each embedding
    embeddings_to_plot = ['X_pca', 'X_pca_harmony', 'X_scanorama', 'X_scvi', 'X_scetm']
    
    # Check which embeddings are available
    available_embeddings = [emb for emb in embeddings_to_plot if emb in adata.obsm.keys()]
    print(f"\nAvailable embeddings: {available_embeddings}")
    
    # Compute UMAP for each embedding
    print("\nComputing UMAPs...")
    for emb_name in available_embeddings:
        print(f"\nComputing UMAP for {emb_name}...")
        # Create temporary AnnData with this embedding
        adata_temp = ad.AnnData(obs=adata.obs.copy())
        adata_temp.obsm[emb_name] = adata.obsm[emb_name]
        
        # Compute UMAP (scanpy will use the embedding in obsm)
        sc.pp.neighbors(adata_temp, use_rep=emb_name, n_neighbors=15, n_pcs=None)
        sc.tl.umap(adata_temp)
        
        # Store UMAP coordinates
        adata.obsm[f'X_umap_{emb_name}'] = adata_temp.obsm['X_umap']
        print(f"  UMAP computed and stored in obsm['X_umap_{emb_name}']")
    
    # Calculate ASW scores for each embedding based on cell_cycle
    print("\nCalculating ASW (Average Silhouette Width) scores...")
    asw_scores = {}
    for emb_name in available_embeddings:
        print(f"\nCalculating ASW for {emb_name}...")
        # Get embedding matrix
        embedding = adata.obsm[emb_name]
        
        # Get cell cycle labels
        cell_cycle_labels = adata.obs['cell_cycle'].values
        
        # Calculate ASW score using sklearn's silhouette_score
        asw_score = silhouette_score(embedding, cell_cycle_labels, metric='euclidean')
        asw_scores[emb_name] = asw_score
        print(f"  ASW score: {asw_score:.4f}")
    
    # Print summary of ASW scores
    print("\n" + "="*50)
    print("ASW Scores Summary (based on cell_cycle):")
    print("="*50)
    for emb_name, score in sorted(asw_scores.items(), key=lambda x: x[1], reverse=True):
        print(f"  {emb_name:20s}: {score:.4f}")
    print("="*50)
    
    # Save ASW scores to file
    asw_df = pd.DataFrame({
        'embedding': list(asw_scores.keys()),
        'asw_score': list(asw_scores.values())
    }).sort_values('asw_score', ascending=False)
    asw_file = os.path.join(output_dir, 'asw_scores.csv')
    asw_df.to_csv(asw_file, index=False)
    print(f"\nASW scores saved to: {asw_file}")
    
    # Plot UMAPs colored by cell cycle type
    print("\nGenerating plots...")
    
    # Plot 1: Combined grid plot
    n_embeddings = len(available_embeddings)
    n_cols = 2
    n_rows = (n_embeddings + 1) // 2
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 6 * n_rows))
    if n_embeddings == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for idx, emb_name in enumerate(available_embeddings):
        ax = axes[idx]
        
        # Get UMAP coordinates
        umap_key = f'X_umap_{emb_name}'
        umap_coords = adata.obsm[umap_key]
        
        # Create scatter plot colored by cell cycle
        scatter = ax.scatter(umap_coords[:, 0], umap_coords[:, 1], 
                            c=pd.Categorical(adata.obs['cell_cycle']).codes,
                            s=0.5, alpha=0.6, cmap='tab20')
        
        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')
        asw_score = asw_scores.get(emb_name, None)
        title = f'UMAP: {emb_name}\n(colored by cell cycle)'
        if asw_score is not None:
            title += f'\nASW: {asw_score:.3f}'
        ax.set_title(title)
        ax.set_aspect('equal')
        
        # Add colorbar with cell cycle labels
        unique_cycles = adata.obs['cell_cycle'].unique()
        cbar = plt.colorbar(scatter, ax=ax, ticks=range(len(unique_cycles)))
        cbar.set_ticklabels(unique_cycles)
        cbar.set_label('Cell Cycle Type', rotation=270, labelpad=15)
    
    # Hide extra subplots if any
    for idx in range(n_embeddings, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    combined_plot_path = os.path.join(output_dir, 'umap_cell_cycle_combined.png')
    plt.savefig(combined_plot_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved combined plot: {combined_plot_path}")
    
    # Plot 2: Individual plots with better color scheme
    for emb_name in available_embeddings:
        umap_key = f'X_umap_{emb_name}'
        umap_coords = adata.obsm[umap_key]
        
        plt.figure(figsize=(10, 8))
        
        # Create a dataframe for easier plotting
        plot_df = pd.DataFrame({
            'UMAP_1': umap_coords[:, 0],
            'UMAP_2': umap_coords[:, 1],
            'cell_cycle': adata.obs['cell_cycle']
        })
        
        # Plot with seaborn for better color handling
        unique_cycles = sorted(plot_df['cell_cycle'].unique())
        palette = sns.color_palette("tab20", n_colors=len(unique_cycles))
        
        for i, cycle_type in enumerate(unique_cycles):
            subset = plot_df[plot_df['cell_cycle'] == cycle_type]
            plt.scatter(subset['UMAP_1'], subset['UMAP_2'], 
                       label=cycle_type, s=0.5, alpha=0.6, color=palette[i])
        
        plt.xlabel('UMAP 1', fontsize=12)
        plt.ylabel('UMAP 2', fontsize=12)
        asw_score = asw_scores.get(emb_name, None)
        title = f'UMAP: {emb_name} (colored by cell cycle type)'
        if asw_score is not None:
            title += f'\nASW: {asw_score:.3f}'
        plt.title(title, fontsize=14, fontweight='bold')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        plt.tight_layout()
        
        individual_plot_path = os.path.join(output_dir, f'umap_{emb_name}_cell_cycle.png')
        plt.savefig(individual_plot_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"Saved individual plot: {individual_plot_path}")
    
    print(f"\nAll plots saved to: {output_dir}/")
    print("Done!")


if __name__ == "__main__":
    main()

