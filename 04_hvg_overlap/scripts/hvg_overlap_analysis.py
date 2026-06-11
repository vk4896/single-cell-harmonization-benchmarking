"""
HVG Overlap Analysis: Before and After Harmonization

This script measures Highly Variable Gene (HVG) overlap before and after 
harmonization using different methods (Harmony, Scanorama, SCVI).

Author: Generated script
Date: 2024
"""

import scanpy as sc
import pandas as pd
import numpy as np
from scipy.sparse import issparse
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import argparse
import os


def reconstruct_from_pca(pca_embeddings, pca_components, mean=None):
    """
    Reconstruct gene expression from PCA embeddings using inverse transform.
    
    Parameters:
    -----------
    pca_embeddings : array-like
        PCA embeddings (cells x n_components)
    pca_components : array-like
        PCA components/loadings (n_components x genes)
    mean : array-like, optional
        Mean vector used during PCA fitting (for centering)
    
    Returns:
    --------
    reconstructed : array-like
        Reconstructed gene expression (cells x genes)
    """
    # Inverse transform: X_reconstructed = X_pca @ components
    # pca_embeddings: (n_cells, n_components)
    # pca_components: (n_components, n_genes)
    # Result: (n_cells, n_genes)
    reconstructed = pca_embeddings @ pca_components
    
    # Add back the mean if provided
    if mean is not None:
        reconstructed = reconstructed + mean
    
    return reconstructed


def compute_hvgs_per_batch(expression_data, batch_labels, n_top_genes=2000, flavor='seurat_v3'):
    """
    Compute HVGs separately for each batch.
    
    Parameters:
    -----------
    expression_data : array-like
        Gene expression matrix (cells x genes)
    batch_labels : array-like
        Batch labels for each cell
    n_top_genes : int
        Number of top HVGs to select per batch
    flavor : str
        Method for computing HVGs ('seurat_v3', 'cell_ranger', 'seurat')
    
    Returns:
    --------
    hvg_dict : dict
        Dictionary mapping batch -> set of HVG gene indices
    """
    hvg_dict = {}
    unique_batches = np.unique(batch_labels)
    
    for batch in unique_batches:
        # Subset data for this batch
        batch_mask = batch_labels == batch
        batch_data = expression_data[batch_mask, :]
        
        # Create temporary AnnData for this batch
        temp_adata = sc.AnnData(batch_data)
        
        # Normalize and log transform
        sc.pp.normalize_total(temp_adata, target_sum=1e4)
        sc.pp.log1p(temp_adata)
        
        # Compute HVGs
        sc.pp.highly_variable_genes(
            temp_adata, 
            n_top_genes=n_top_genes,
            flavor=flavor,
            subset=False
        )
        
        # Get HVG gene indices
        hvg_genes = np.where(temp_adata.var['highly_variable'])[0]
        hvg_dict[batch] = set(hvg_genes.tolist())
        
        print(f"  Batch {batch}: {len(hvg_genes)} HVGs")
    
    return hvg_dict


def compute_overlap_metrics(hvg_dict):
    """
    Compute pairwise Jaccard index and overlap coefficient between batches.
    
    Parameters:
    -----------
    hvg_dict : dict
        Dictionary mapping batch -> set of HVG gene indices
    
    Returns:
    --------
    jaccard_matrix : pd.DataFrame
        Jaccard index matrix (batches x batches)
    overlap_matrix : pd.DataFrame
        Overlap coefficient matrix (batches x batches)
    """
    batches = sorted(hvg_dict.keys())
    n_batches = len(batches)
    
    jaccard_matrix = np.zeros((n_batches, n_batches))
    overlap_matrix = np.zeros((n_batches, n_batches))
    
    for i, batch1 in enumerate(batches):
        for j, batch2 in enumerate(batches):
            set1 = hvg_dict[batch1]
            set2 = hvg_dict[batch2]
            
            intersection = len(set1 & set2)
            union = len(set1 | set2)
            
            # Jaccard index: |A ∩ B| / |A ∪ B|
            if union > 0:
                jaccard_matrix[i, j] = intersection / union
            
            # Overlap coefficient: |A ∩ B| / min(|A|, |B|)
            if len(set1) > 0 and len(set2) > 0:
                overlap_matrix[i, j] = intersection / min(len(set1), len(set2))
    
    jaccard_df = pd.DataFrame(jaccard_matrix, index=batches, columns=batches)
    overlap_df = pd.DataFrame(overlap_matrix, index=batches, columns=batches)
    
    return jaccard_df, overlap_df


def compute_mean_pairwise_overlap(jaccard_matrix):
    """
    Compute mean pairwise Jaccard index (excluding diagonal).
    """
    # Get upper triangle (excluding diagonal)
    upper_triangle = np.triu(jaccard_matrix.values, k=1)
    mean_overlap = upper_triangle[upper_triangle > 0].mean()
    return mean_overlap


def sample_data(adata, n_cells=780000 , random_state=42):
    """
    Sample a subset of cells from the AnnData object.
    
    Parameters:
    -----------
    adata : AnnData
        Annotated data object
    n_cells : int
        Number of cells to sample
    random_state : int
        Random seed for reproducibility
    
    Returns:
    --------
    adata_subset : AnnData
        Subsampled AnnData object
    """
    if n_cells is None or n_cells >= adata.n_obs:
        print(f"Using all {adata.n_obs} cells")
        return adata
    
    np.random.seed(random_state)
    indices = np.random.choice(adata.n_obs, size=n_cells, replace=False)
    adata_subset = adata[indices].copy()
    print(f"Sampled {n_cells} cells from {adata.n_obs} total cells")
    return adata_subset


def main():
    parser = argparse.ArgumentParser(description='HVG Overlap Analysis')
    parser.add_argument('--input', '-i', type=str, default='../combined_data.h5ad',
                        help='Path to input h5ad file')
    parser.add_argument('--output-dir', '-o', type=str, default='.',
                        help='Output directory for results')
    parser.add_argument('--n-cells', '-n', type=int, default=100000,
                        help='Number of cells to sample (None for all cells)')
    parser.add_argument('--n-top-genes', type=int, default=2000,
                        help='Number of top HVGs to select per batch')
    parser.add_argument('--random-seed', type=int, default=42,
                        help='Random seed for sampling')
    
    args = parser.parse_args()
    
    # Create output directories if they don't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create separate folders for plots and results
    plots_dir = os.path.join(args.output_dir, 'plots')
    results_dir = os.path.join(args.output_dir, 'results')
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    print(f"Output directory: {args.output_dir}")
    print(f"Plots will be saved to: {plots_dir}")
    print(f"Results will be saved to: {results_dir}")
    
    print("=" * 60)
    print("HVG OVERLAP ANALYSIS")
    print("=" * 60)
    
    # Load data
    print(f"\nLoading data from: {args.input}")
    adata = sc.read_h5ad(args.input)
    print(f"Original data shape: {adata.shape}")
    
    # Sample cells if requested
    if args.n_cells:
        adata = sample_data(adata, n_cells=args.n_cells, random_state=args.random_seed)
        print(f"Working with {adata.n_obs} cells")
    
    # Step 1: Check batch information and prepare data
    print("\n" + "=" * 60)
    print("Step 1: Data Preparation")
    print("=" * 60)
    print(f"Number of batches: {adata.obs['batch'].nunique()}")
    print(f"Batch distribution:\n{adata.obs['batch'].value_counts()}")
    
    # Check if raw data exists, otherwise use X
    if adata.raw is not None:
        print("\nUsing adata.raw for gene expression")
        count_data = adata.raw.X
    else:
        print("\nUsing adata.X for gene expression")
        count_data = adata.X
    
    print(f"Count data shape: {count_data.shape}")
    print(f"Sparse matrix: {issparse(count_data)}")
    
    # Step 2: Compute HVGs BEFORE harmonization
    print("\n" + "=" * 60)
    print("Step 2: Computing HVGs BEFORE Harmonization")
    print("=" * 60)
    
    # Use original count data
    if issparse(count_data):
        count_data_dense = count_data.toarray()
    else:
        count_data_dense = count_data.copy()
    
    print(f"\nComputing HVGs per batch from original data...")
    hvg_dict_before = compute_hvgs_per_batch(
        count_data_dense, 
        adata.obs['batch'].values,
        n_top_genes=args.n_top_genes,
        flavor='seurat_v3'
    )
    
    # Compute overlap metrics
    jaccard_before, overlap_before = compute_overlap_metrics(hvg_dict_before)
    mean_jaccard_before = compute_mean_pairwise_overlap(jaccard_before)
    
    print(f"\nMean pairwise Jaccard index (before harmonization): {mean_jaccard_before:.4f}")
    
    # Step 3: Reconstruct and compute HVGs AFTER harmonization - HARMONY
    print("\n" + "=" * 60)
    print("Step 3: Computing HVGs AFTER Harmony Harmonization")
    print("=" * 60)
    
    # Get PCA components and embeddings
    # Note: In scanpy, varm['PCs'] is stored as (n_genes, n_components)
    # We need to transpose it to (n_components, n_genes) for reconstruction
    pca_components_raw = adata.varm['PCs']  # (n_genes, n_components)
    pca_components = pca_components_raw.T  # (n_components, n_genes)
    pca_harmony_embeddings = adata.obsm['X_pca_harmony']
    
    print(f"PCA components shape (raw): {pca_components_raw.shape}")
    print(f"PCA components shape (transposed): {pca_components.shape}")
    print(f"Harmony embeddings shape: {pca_harmony_embeddings.shape}")
    
    # Reconstruct gene expression from harmony embeddings
    n_components = min(pca_harmony_embeddings.shape[1], pca_components.shape[0])
    harmony_reconstructed = reconstruct_from_pca(
        pca_harmony_embeddings[:, :n_components],
        pca_components[:n_components, :]
    )
    
    print(f"Reconstructed expression shape: {harmony_reconstructed.shape}")
    
    # Ensure non-negative
    harmony_reconstructed = np.clip(harmony_reconstructed, 0, None)
    
    # Compute HVGs per batch
    print("\nComputing HVGs per batch from harmony-reconstructed data...")
    hvg_dict_harmony = compute_hvgs_per_batch(
        harmony_reconstructed,
        adata.obs['batch'].values,
        n_top_genes=args.n_top_genes,
        flavor='seurat_v3'
    )
    
    # Compute overlap metrics
    jaccard_harmony, overlap_harmony = compute_overlap_metrics(hvg_dict_harmony)
    mean_jaccard_harmony = compute_mean_pairwise_overlap(jaccard_harmony)
    
    print(f"\nMean pairwise Jaccard index (after harmony): {mean_jaccard_harmony:.4f}")
    
    # Step 4: Reconstruct and compute HVGs AFTER harmonization - SCANORAMA
    print("\n" + "=" * 60)
    print("Step 4: Computing HVGs AFTER Scanorama Harmonization")
    print("=" * 60)
    
    scanorama_embeddings = adata.obsm['X_scanorama']
    print(f"Scanorama embeddings shape: {scanorama_embeddings.shape}")
    
    # For scanorama, reconstruct using PCA
    # If scanorama has more dimensions than original PCA components, use PCA projection
    if scanorama_embeddings.shape[1] <= pca_components.shape[0]:
        # Can directly use the PCA components
        n_components = scanorama_embeddings.shape[1]
        scanorama_reconstructed = reconstruct_from_pca(
            scanorama_embeddings,
            pca_components[:n_components, :]
        )
    else:
        # If dimensions don't match, use PCA to project scanorama embeddings
        # to the original PCA component space, then reconstruct
        print("Warning: Scanorama dimensions don't match PCA. Using PCA projection.")
        n_proj_components = min(scanorama_embeddings.shape[1], pca_components.shape[0])
        pca_sc = PCA(n_components=n_proj_components)
        pca_sc.fit(scanorama_embeddings)
        intermediate = pca_sc.transform(scanorama_embeddings)
        # Now use the intermediate (projected) embeddings with original PCA components
        scanorama_reconstructed = reconstruct_from_pca(
            intermediate,
            pca_components[:n_proj_components, :]
        )
    
    # Ensure non-negative
    scanorama_reconstructed = np.clip(scanorama_reconstructed, 0, None)
    
    print(f"Reconstructed expression shape: {scanorama_reconstructed.shape}")
    
    # Compute HVGs per batch
    print("\nComputing HVGs per batch from scanorama-reconstructed data...")
    hvg_dict_scanorama = compute_hvgs_per_batch(
        scanorama_reconstructed,
        adata.obs['batch'].values,
        n_top_genes=args.n_top_genes,
        flavor='seurat_v3'
    )
    
    # Compute overlap metrics
    jaccard_scanorama, overlap_scanorama = compute_overlap_metrics(hvg_dict_scanorama)
    mean_jaccard_scanorama = compute_mean_pairwise_overlap(jaccard_scanorama)
    
    print(f"\nMean pairwise Jaccard index (after scanorama): {mean_jaccard_scanorama:.4f}")
    
    # Step 5: Reconstruct and compute HVGs AFTER harmonization - SCVI
    print("\n" + "=" * 60)
    print("Step 5: Computing HVGs AFTER SCVI Harmonization")
    print("=" * 60)
    
    scvi_embeddings = adata.obsm['X_scvi']
    print(f"SCVI embeddings shape: {scvi_embeddings.shape}")
    
    # Reconstruct using PCA approach
    if scvi_embeddings.shape[1] <= pca_components.shape[0]:
        # Can directly use the PCA components
        n_components = scvi_embeddings.shape[1]
        scvi_reconstructed = reconstruct_from_pca(
            scvi_embeddings,
            pca_components[:n_components, :]
        )
    else:
        # Use PCA projection to match dimensions
        print("Using PCA projection for SCVI reconstruction.")
        n_proj_components = min(scvi_embeddings.shape[1], pca_components.shape[0])
        pca_scvi = PCA(n_components=n_proj_components)
        pca_scvi.fit(scvi_embeddings)
        intermediate = pca_scvi.transform(scvi_embeddings)
        # Use the projected embeddings with original PCA components
        scvi_reconstructed = reconstruct_from_pca(
            intermediate,
            pca_components[:n_proj_components, :]
        )
    
    # Ensure non-negative
    scvi_reconstructed = np.clip(scvi_reconstructed, 0, None)
    
    print(f"Reconstructed expression shape: {scvi_reconstructed.shape}")
    
    # Compute HVGs per batch
    print("\nComputing HVGs per batch from scvi-reconstructed data...")
    hvg_dict_scvi = compute_hvgs_per_batch(
        scvi_reconstructed,
        adata.obs['batch'].values,
        n_top_genes=args.n_top_genes,
        flavor='seurat_v3'
    )
    
    # Compute overlap metrics
    jaccard_scvi, overlap_scvi = compute_overlap_metrics(hvg_dict_scvi)
    mean_jaccard_scvi = compute_mean_pairwise_overlap(jaccard_scvi)
    
    print(f"\nMean pairwise Jaccard index (after scvi): {mean_jaccard_scvi:.4f}")
    
    # Step 6: Summary and comparison
    print("\n" + "=" * 60)
    print("Step 6: Summary - HVG Overlap Comparison")
    print("=" * 60)
    
    results_summary = pd.DataFrame({
        'Method': ['Before Harmonization', 'Harmony', 'Scanorama', 'SCVI'],
        'Mean Jaccard Index': [
            mean_jaccard_before,
            mean_jaccard_harmony,
            mean_jaccard_scanorama,
            mean_jaccard_scvi
        ]
    })
    
    print("\nMean Pairwise Jaccard Index (higher = better overlap):")
    print(results_summary.to_string(index=False))
    
    # Calculate improvement
    results_summary['Improvement'] = results_summary['Mean Jaccard Index'] - results_summary.loc[0, 'Mean Jaccard Index']
    results_summary.loc[0, 'Improvement'] = 0  # No improvement for baseline
    
    print("\nImprovement over baseline:")
    print(results_summary.to_string(index=False))
    
    # Step 7: Visualization - Bar plots
    print("\n" + "=" * 60)
    print("Step 7: Creating Visualizations")
    print("=" * 60)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Mean Jaccard Index comparison
    ax1 = axes[0]
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
    bars = ax1.bar(results_summary['Method'], results_summary['Mean Jaccard Index'], 
                    color=colors, alpha=0.7)
    ax1.set_ylabel('Mean Pairwise Jaccard Index', fontsize=12)
    ax1.set_title('HVG Overlap: Before vs After Harmonization', fontsize=14, fontweight='bold')
    ax1.set_ylim([0, max(results_summary['Mean Jaccard Index']) * 1.1])
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, val in zip(bars, results_summary['Mean Jaccard Index']):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                 f'{val:.4f}',
                 ha='center', va='bottom', fontsize=10)
    
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Plot 2: Improvement over baseline
    ax2 = axes[1]
    bars2 = ax2.bar(results_summary['Method'], results_summary['Improvement'], 
                    color=colors, alpha=0.7)
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax2.set_ylabel('Improvement over Baseline', fontsize=12)
    ax2.set_title('HVG Overlap Improvement', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, val in zip(bars2, results_summary['Improvement']):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                 f'{val:+.4f}',
                 ha='center', va='bottom' if val >= 0 else 'top', fontsize=10)
    
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    output_path = os.path.join(plots_dir, 'hvg_overlap_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved comparison plot: {output_path}")
    plt.close()
    
    # Step 8: Heatmap visualization
    sample_batches = sorted(list(hvg_dict_before.keys()))
    if len(sample_batches) > 10:
        sample_batches = sample_batches[:10]
        print(f"\nNote: Showing heatmaps for first 10 batches (out of {len(hvg_dict_before.keys())})")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    # Before harmonization
    jaccard_before_sample = jaccard_before.loc[sample_batches, sample_batches]
    sns.heatmap(jaccard_before_sample, annot=True, fmt='.3f', cmap='YlOrRd', 
                vmin=0, vmax=1, ax=axes[0, 0], cbar_kws={'label': 'Jaccard Index'})
    axes[0, 0].set_title('Before Harmonization', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Batch')
    axes[0, 0].set_ylabel('Batch')
    
    # After Harmony
    jaccard_harmony_sample = jaccard_harmony.loc[sample_batches, sample_batches]
    sns.heatmap(jaccard_harmony_sample, annot=True, fmt='.3f', cmap='YlOrRd',
                vmin=0, vmax=1, ax=axes[0, 1], cbar_kws={'label': 'Jaccard Index'})
    axes[0, 1].set_title('After Harmony', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Batch')
    axes[0, 1].set_ylabel('Batch')
    
    # After Scanorama
    jaccard_scanorama_sample = jaccard_scanorama.loc[sample_batches, sample_batches]
    sns.heatmap(jaccard_scanorama_sample, annot=True, fmt='.3f', cmap='YlOrRd',
                vmin=0, vmax=1, ax=axes[1, 0], cbar_kws={'label': 'Jaccard Index'})
    axes[1, 0].set_title('After Scanorama', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Batch')
    axes[1, 0].set_ylabel('Batch')
    
    # After SCVI
    jaccard_scvi_sample = jaccard_scvi.loc[sample_batches, sample_batches]
    sns.heatmap(jaccard_scvi_sample, annot=True, fmt='.3f', cmap='YlOrRd',
                vmin=0, vmax=1, ax=axes[1, 1], cbar_kws={'label': 'Jaccard Index'})
    axes[1, 1].set_title('After SCVI', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Batch')
    axes[1, 1].set_ylabel('Batch')
    
    plt.suptitle('Pairwise HVG Overlap (Jaccard Index)', 
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    output_path = os.path.join(plots_dir, 'hvg_overlap_heatmaps.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved heatmap plot: {output_path}")
    plt.close()
    
    # Step 9: Save results to CSV
    print("\n" + "=" * 60)
    print("Step 9: Saving Results")
    print("=" * 60)
    
    output_path = os.path.join(results_dir, 'hvg_overlap_results.csv')
    results_summary.to_csv(output_path, index=False)
    print(f"Saved summary: {output_path}")
    
    # Save detailed Jaccard matrices
    jaccard_before.to_csv(os.path.join(results_dir, 'jaccard_matrix_before.csv'))
    jaccard_harmony.to_csv(os.path.join(results_dir, 'jaccard_matrix_harmony.csv'))
    jaccard_scanorama.to_csv(os.path.join(results_dir, 'jaccard_matrix_scanorama.csv'))
    jaccard_scvi.to_csv(os.path.join(results_dir, 'jaccard_matrix_scvi.csv'))
    
    print("Saved all Jaccard matrices to CSV files.")
    print("\nAnalysis complete!")


if __name__ == '__main__':
    main()

