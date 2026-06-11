#!/usr/bin/env python3
"""
Comprehensive ccAF evaluation:
1. Baseline: X_pca from combined_data.h5ad
2. Compare other embeddings from combined_data.h5ad vs baseline
3. Compare harmonized X_pca (from classifer/combined_har_data.h5ad) vs baseline

Generates confusion matrices, metrics, and plots for all comparisons.
"""

import os
import math
import sys
from datetime import datetime
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import types
import logging

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns

# Setup logging
def setup_logging(log_file="ccaf_comprehensive_eval.log"):
    """Setup logging to both file and console."""
    # Create logger
    logger = logging.getLogger('ccaf_eval')
    logger.setLevel(logging.DEBUG)
    
    # Remove existing handlers
    logger.handlers = []
    
    # File handler
    fh = logging.FileHandler(log_file, mode='w')
    fh.setLevel(logging.DEBUG)
    
    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', 
                                  datefmt='%Y-%m-%d %H:%M:%S')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    # Add handlers
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger

# Initialize logger
logger = setup_logging()

# TensorFlow 1.x compatibility for ccAF
import tensorflow as tf

tf.compat.v1.disable_eager_execution()
tf.placeholder = tf.compat.v1.placeholder
tf.Session = tf.compat.v1.Session

import ccAF
logger.info('loaded all imports')

# Fix ccAF bug (set used as index in prep)
logger.info('fixed ccAF bug')
def _fixed_prep_predict_data(self, data):
    missing = set(self.genes).difference(data.index)
    if len(missing) > 0:
        data = pd.concat([data, pd.DataFrame(0, index=list(missing), columns=data.columns)])
    return data.loc[list(self.genes)]

ccAF.ccAF._Classifier_ACTINN__prep_predict_data = types.MethodType(
    _fixed_prep_predict_data, ccAF.ccAF
)


def build_ccaf_ann_from_embedding(embedding_matrix: np.ndarray, obs_index) -> ad.AnnData:
    """Create a ccAF-shaped AnnData using an embedding matrix as X."""
    ccaf_genes = list(ccAF.ccAF.genes)
    k = len(ccaf_genes)  # usually 1472
    n, d = embedding_matrix.shape
    logger.debug(f"Preparing embedding matrix with shape {embedding_matrix.shape} for ccAF (expects {k} features)")
    if d >= k:
        Xk = embedding_matrix[:, :k]
        if d > k:
            logger.debug(f" - Truncated features: kept first {k} of {d}")
        else:
            logger.debug(" - Exact feature count; no truncation needed")
    else:
        pad = np.zeros((n, k - d), dtype=embedding_matrix.dtype)
        Xk = np.hstack([embedding_matrix, pad])
        logger.debug(f" - Padded features: added {k - d} zero columns to reach {k}")
    var_df = pd.DataFrame(index=ccaf_genes)
    ad_emb = ad.AnnData(X=Xk, obs=pd.DataFrame(index=obs_index), var=var_df)
    logger.debug(f" - Built AnnData: X shape {ad_emb.shape}, var names set to ccAF genes")
    return ad_emb


def ccaf_predict_batched(adata_like: ad.AnnData, batch_size: int = 50000) -> np.ndarray:
    """Run ccAF predictions in batches."""
    n = adata_like.n_obs
    out: list[str] = []
    steps = math.ceil(n / batch_size)
    logger.info(f"Starting batched ccAF predictions: {n} cells in {steps} batches of up to {batch_size}")
    for i in range(steps):
        s = i * batch_size
        e = min((i + 1) * batch_size, n)
        logger.info(f"Batch {i+1}/{steps}: rows {s}:{e}")
        sub = adata_like[s:e, :].copy()
        out.extend(ccAF.ccAF.predict_labels(sub))
    logger.info("Completed batched predictions")
    return np.array(out)


def plot_confusion_heatmap(cm_df: pd.DataFrame, title: str, out_path: str, baseline_label: str = "Baseline") -> None:
    """Plot confusion matrix as heatmap."""
    logger.info(f"Plotting heatmap: {title}")
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_df, annot=True, cmap="Blues", fmt="d", cbar_kws={'label': 'Count'})
    plt.title(title)
    plt.ylabel(baseline_label)
    plt.xlabel("Predictions")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    logger.info(f" - Saved heatmap to {out_path}")


def main():
    logger.info('='*70)
    logger.info('COMPREHENSIVE ccAF EVALUATION')
    logger.info('='*70)
    logger.info(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    original_path = "combined_data.h5ad"
    harmonized_path = "../classifer/combined_har_data.h5ad"
    max_cells = 780_000
    baseline_name = "X_pca"
    original_embedding_names = ["X_pca", "X_pca_harmony", "X_scanorama", "X_scetm", "X_scvi"]
    
    # ========================================================================
    # STEP 1: Create baseline from original data X_pca
    # ========================================================================
    logger.info("\n" + "="*70)
    logger.info("STEP 1: Creating baseline from original data X_pca")
    logger.info("="*70)
    logger.info(f"Loading original data from: {original_path}")
    adata_original = sc.read_h5ad(original_path)
    logger.info(f"Loaded original: {adata_original.n_obs} cells, {adata_original.n_vars} genes")
    
    adata_orig_subset = adata_original[:max_cells, :].copy()
    logger.info(f"Subset to first {adata_orig_subset.n_obs} cells")
    
    if baseline_name not in adata_orig_subset.obsm:
        logger.error(f"Baseline embedding '{baseline_name}' not found in original data")
        raise ValueError(f"Baseline embedding '{baseline_name}' not found in original data")
    
    logger.info(f"Getting baseline predictions from original {baseline_name}...")
    orig_X_pca = adata_orig_subset.obsm[baseline_name]
    ad_orig_baseline = build_ccaf_ann_from_embedding(orig_X_pca, obs_index=adata_orig_subset.obs_names)
    baseline_predictions = ccaf_predict_batched(ad_orig_baseline, batch_size=50_000)
    logger.info(f"Baseline predictions shape: {len(baseline_predictions)}")
    logger.info("Baseline prediction distribution:")
    logger.info("\n" + str(pd.Series(baseline_predictions).value_counts().head(10)))
    
    # ========================================================================
    # STEP 2: Compare other embeddings from original data vs baseline
    # ========================================================================
    logger.info("\n" + "="*70)
    logger.info("STEP 2: Comparing other embeddings from original data vs baseline")
    logger.info("="*70)
    
    # Get other embeddings from original data
    original_embs = {}
    for name in original_embedding_names:
        if name == baseline_name:
            continue  # Skip baseline, already done
        if name in adata_orig_subset.obsm:
            original_embs[name] = adata_orig_subset.obsm[name]
            logger.info(f"Found embedding '{name}' with shape {original_embs[name].shape}")
        else:
            logger.warning(f"Embedding {name} not found; skipping")
    
    # Predict for other embeddings
    original_labels = {}
    for name, mat in original_embs.items():
        logger.info("\n" + "-"*60)
        logger.info(f"Predicting for original embedding: {name}")
        logger.info("-"*60)
        ad_emb = build_ccaf_ann_from_embedding(mat, obs_index=adata_orig_subset.obs_names)
        preds = ccaf_predict_batched(ad_emb, batch_size=50_000)
        original_labels[name] = preds
        logger.info("Prediction distribution (top 10):")
        logger.info("\n" + str(pd.Series(preds).value_counts().head(10)))
    
    # ========================================================================
    # STEP 3: Compare harmonized X_pca vs baseline
    # ========================================================================
    logger.info("\n" + "="*70)
    logger.info("STEP 3: Comparing harmonized X_pca vs baseline")
    logger.info("="*70)
    
    logger.info(f"Loading harmonized data from: {harmonized_path}")
    adata_harmonized = sc.read_h5ad(harmonized_path)
    logger.info(f"Loaded harmonized: {adata_harmonized.n_obs} cells, {adata_harmonized.n_vars} genes")
    
    adata_harm_subset = adata_harmonized[:max_cells, :].copy()
    logger.info(f"Subset to first {adata_harm_subset.n_obs} cells")
    
    if baseline_name not in adata_harm_subset.obsm:
        logger.error(f"X_pca not found in harmonized data")
        raise ValueError(f"X_pca not found in harmonized data")
    
    logger.info(f"Getting predictions from harmonized {baseline_name}...")
    harm_X_pca = adata_harm_subset.obsm[baseline_name]
    ad_harm_baseline = build_ccaf_ann_from_embedding(harm_X_pca, obs_index=adata_harm_subset.obs_names)
    harmonized_predictions = ccaf_predict_batched(ad_harm_baseline, batch_size=50_000)
    logger.info(f"Harmonized predictions shape: {len(harmonized_predictions)}")
    logger.info("Harmonized prediction distribution:")
    logger.info("\n" + str(pd.Series(harmonized_predictions).value_counts().head(10)))
    
    # ========================================================================
    # STEP 4: Generate confusion matrices and metrics
    # ========================================================================
    logger.info("\n" + "="*70)
    logger.info("STEP 4: Generating confusion matrices and metrics")
    logger.info("="*70)
    
    from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
    
    # Collect all predictions and classes
    all_predictions = [baseline_predictions]
    all_predictions.extend(original_labels.values())
    all_predictions.append(harmonized_predictions)
    classes = np.unique(np.concatenate(all_predictions))
    logger.info(f"Unique classes across all predictions: {len(classes)}")
    logger.info(f"Classes: {classes}")
    
    # Create output directories
    out_dir_original = "ccaf_original_comparisons"
    out_dir_harmonized = "ccaf_harmonized_comparisons"
    os.makedirs(out_dir_original, exist_ok=True)
    os.makedirs(out_dir_harmonized, exist_ok=True)
    logger.info(f"Output directories created: {out_dir_original}/, {out_dir_harmonized}/")
    
    # Build predictions dataframe
    obs_index_orig = np.array(adata_orig_subset.obs_names)
    obs_index_harm = np.array(adata_harm_subset.obs_names)
    
    all_preds_df = pd.DataFrame({
        "cell_id_original": np.concatenate([obs_index_orig, np.full(len(obs_index_harm), "N/A")]),
        "cell_id_harmonized": np.concatenate([np.full(len(obs_index_orig), "N/A"), obs_index_harm]),
        f"{baseline_name}_baseline": np.concatenate([baseline_predictions, np.full(len(harmonized_predictions), "N/A")])
    })
    
    # Add original embedding predictions
    for name, preds in original_labels.items():
        all_preds_df[f"original_{name}"] = np.concatenate([preds, np.full(len(harmonized_predictions), "N/A")])
    
    # Add harmonized predictions
    all_preds_df[f"harmonized_{baseline_name}"] = np.concatenate([
        np.full(len(baseline_predictions), "N/A"), harmonized_predictions
    ])
    
    # Metrics storage
    metrics_rows = []
    
    # Compare original embeddings vs baseline
    logger.info("\n" + "-"*70)
    logger.info("Original data embeddings vs baseline:")
    logger.info("-"*70)
    for name, preds in original_labels.items():
        cm = confusion_matrix(baseline_predictions, preds, labels=classes)
        cm_df = pd.DataFrame(cm, index=classes, columns=classes)
        
        title = f"ccAF: Original {name} vs Baseline ({baseline_name})"
        out_png = os.path.join(out_dir_original, f"confusion_{name}_vs_{baseline_name}.png")
        plot_confusion_heatmap(cm_df, title, out_png, baseline_label=f"Baseline ({baseline_name})")
        
        out_csv = os.path.join(out_dir_original, f"confusion_{name}_vs_{baseline_name}.csv")
        cm_df.to_csv(out_csv)
        logger.info(f"Saved confusion matrix CSV: {out_csv}")
        
        # Metrics
        acc = accuracy_score(baseline_predictions, preds)
        f1_macro = f1_score(baseline_predictions, preds, average="macro")
        f1_weighted = f1_score(baseline_predictions, preds, average="weighted")
        diff_count = int(np.sum(baseline_predictions != preds))
        
        logger.info(f"  Accuracy: {acc:.4f}")
        logger.info(f"  F1 (macro): {f1_macro:.4f}")
        logger.info(f"  F1 (weighted): {f1_weighted:.4f}")
        logger.info(f"  Cells differing: {diff_count}/{len(baseline_predictions)}")
        
        metrics_rows.append({
            "comparison_type": "original_vs_baseline",
            "embedding": name,
            "baseline": baseline_name,
            "accuracy": acc,
            "f1_macro": f1_macro,
            "f1_weighted": f1_weighted,
            "diff_count": diff_count,
            "total_cells": len(baseline_predictions)
        })
        
        # Add disagreement flag
        all_preds_df[f"diff_original_{name}_vs_baseline"] = np.concatenate([
            baseline_predictions != preds, np.full(len(harmonized_predictions), False)
        ])
    
    # Compare harmonized X_pca vs baseline
    logger.info("\n" + "-"*70)
    logger.info("Harmonized X_pca vs baseline:")
    logger.info("-"*70)
    
    # Truncate baseline to match harmonized length if needed
    base_for_harm = baseline_predictions[:len(harmonized_predictions)]
    
    cm = confusion_matrix(base_for_harm, harmonized_predictions, labels=classes)
    cm_df = pd.DataFrame(cm, index=classes, columns=classes)
    
    title = f"ccAF: Harmonized {baseline_name} vs Baseline ({baseline_name})"
    out_png = os.path.join(out_dir_harmonized, f"confusion_harmonized_{baseline_name}_vs_baseline.png")
    plot_confusion_heatmap(cm_df, title, out_png, baseline_label=f"Baseline ({baseline_name})")
    
    out_csv = os.path.join(out_dir_harmonized, f"confusion_harmonized_{baseline_name}_vs_baseline.csv")
    cm_df.to_csv(out_csv)
    logger.info(f"Saved confusion matrix CSV: {out_csv}")
    
    # Metrics
    acc = accuracy_score(base_for_harm, harmonized_predictions)
    f1_macro = f1_score(base_for_harm, harmonized_predictions, average="macro")
    f1_weighted = f1_score(base_for_harm, harmonized_predictions, average="weighted")
    diff_count = int(np.sum(base_for_harm != harmonized_predictions))
    
    logger.info(f"  Accuracy: {acc:.4f}")
    logger.info(f"  F1 (macro): {f1_macro:.4f}")
    logger.info(f"  F1 (weighted): {f1_weighted:.4f}")
    logger.info(f"  Cells differing: {diff_count}/{len(harmonized_predictions)}")
    
    metrics_rows.append({
        "comparison_type": "harmonized_vs_baseline",
        "embedding": f"harmonized_{baseline_name}",
        "baseline": baseline_name,
        "accuracy": acc,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "diff_count": diff_count,
        "total_cells": len(harmonized_predictions)
    })
    
    # Add disagreement flag
    all_preds_df[f"diff_harmonized_{baseline_name}_vs_baseline"] = np.concatenate([
        np.full(len(baseline_predictions), False), base_for_harm != harmonized_predictions
    ])
    
    # ========================================================================
    # STEP 5: Save results
    # ========================================================================
    logger.info("\n" + "="*70)
    logger.info("STEP 5: Saving results")
    logger.info("="*70)
    
    # Save predictions
    preds_csv = "ccaf_all_predictions_comparison.csv"
    all_preds_df.to_csv(preds_csv, index=False)
    logger.info(f"Saved all predictions: {preds_csv}")
    
    # Save metrics
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_csv = "ccaf_all_metrics_comparison.csv"
    metrics_df.to_csv(metrics_csv, index=False)
    logger.info(f"Saved metrics: {metrics_csv}")
    
    # Plot metrics summary
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Original comparisons
    orig_metrics = metrics_df[metrics_df["comparison_type"] == "original_vs_baseline"]
    if len(orig_metrics) > 0:
        ax1 = axes[0]
        width = 0.25
        x = np.arange(len(orig_metrics))
        ax1.bar(x - width, orig_metrics["accuracy"], width=width, label="Accuracy")
        ax1.bar(x, orig_metrics["f1_macro"], width=width, label="F1 Macro")
        ax1.bar(x + width, orig_metrics["f1_weighted"], width=width, label="F1 Weighted")
        ax1.set_xticks(x)
        ax1.set_xticklabels(orig_metrics["embedding"], rotation=15)
        ax1.set_ylim(0, 1.0)
        ax1.set_ylabel("Score")
        ax1.set_title("Original Embeddings vs Baseline")
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
    
    # Harmonized comparison
    harm_metrics = metrics_df[metrics_df["comparison_type"] == "harmonized_vs_baseline"]
    if len(harm_metrics) > 0:
        ax2 = axes[1]
        width = 0.25
        x = np.arange(len(harm_metrics))
        ax2.bar(x - width, harm_metrics["accuracy"], width=width, label="Accuracy")
        ax2.bar(x, harm_metrics["f1_macro"], width=width, label="F1 Macro")
        ax2.bar(x + width, harm_metrics["f1_weighted"], width=width, label="F1 Weighted")
        ax2.set_xticks(x)
        ax2.set_xticklabels(harm_metrics["embedding"], rotation=15)
        ax2.set_ylim(0, 1.0)
        ax2.set_ylabel("Score")
        ax2.set_title("Harmonized X_pca vs Baseline")
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    metrics_png = "ccaf_metrics_summary_all.png"
    plt.savefig(metrics_png, dpi=200)
    plt.close()
    logger.info(f"Saved metrics summary plot: {metrics_png}")
    
    logger.info("\n" + "="*70)
    logger.info("ANALYSIS COMPLETE!")
    logger.info("="*70)
    logger.info(f"\nSummary:")
    logger.info(f"  - Baseline predictions: {len(baseline_predictions)} cells")
    logger.info(f"  - Original embedding comparisons: {len(original_labels)}")
    logger.info(f"  - Harmonized predictions: {len(harmonized_predictions)} cells")
    logger.info(f"\nOutput files:")
    logger.info(f"  - Predictions: {preds_csv}")
    logger.info(f"  - Metrics: {metrics_csv}")
    logger.info(f"  - Metrics plot: {metrics_png}")
    logger.info(f"  - Original comparisons: {out_dir_original}/")
    logger.info(f"  - Harmonized comparisons: {out_dir_harmonized}/")
    logger.info(f"  - Log file: ccaf_comprehensive_eval.log")
    logger.info(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()

