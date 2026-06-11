# Deep-Learning–Driven Harmonization of Single-Cell Transcriptomic Data

### Cross-Study Integration and Downstream Inference — Anubio AI Internship

> Modern scRNA-seq experiments profile gene expression across thousands to millions
> of individual cells. Datasets from different labs, sequencing technologies, or
> handling protocols carry pronounced **batch effects** that hamper joint analysis.
> This project develops and benchmarks a **harmonization network** — a neural
> framework that removes batch effects while preserving true biology — so that cells
> from multiple experiments can be analysed jointly.

**Project:** [Anubio AI](https://anubio.ai/) · **Role:** Intern — extend, validate,
and benchmark the harmonization model.

Once harmonized, the latent representations support **perturbation modelling**
(drug/cytokine-driven state shifts), **patient stratification** (healthy vs. diseased,
responder vs. non-responder), and **cell-type discovery / trajectory inference**.

---

## Repository architecture

The folders are organized to mirror the three **high-level objectives** of the
project brief — *Data Curation & QC*, *Evaluation & Benchmarking*, and *Downstream
Integration* — plus supporting environment and report material.

```
.
├── README.md
├── .gitignore
│ 
│
├── environment/                         # Deliverable: reproducible training/eval setup
│   ├── requirements.txt                 #   core deps (scanpy, anndata, sklearn)
│   ├── requirements_gpu.txt             #   GPU deps (PyTorch, scVI, scETM)
│   ├── requirements_full.txt            #   full frozen environment
│   ├── requirements_fastmnn.txt         #   isolated env for the fastMNN experiment
│   └── config_example.yaml              #   pipeline configuration (epochs, theta, splits)
│
├── 01_harmonization_umap/               # ── OBJECTIVE: Evaluation & Benchmarking (qualitative) ──
│   │                                    #   UMAP overlays + integration metrics vs. baselines
│   ├── models/                          #   method wrappers: harmony, scanorama, scVI, scETM
│   ├── scripts/                         #   per-method run scripts + UMAP / metric pipelines
│   ├── notebooks/                       #   integration_metrics, umaps, umaps_final, umaps_pdf
│   └── results/
│       ├── umap_plots/                  #   UMAP figures (Figures 1–9 of the review)
│       └── scetm_training_logs/         #   scETM training run logs
│
├── 02_cell_type_classification/         # ── OBJECTIVE: Evaluation & Benchmarking (downstream) ──
│   │                                    #   biological preservation + perturbation recovery
│   ├── models/                          #   method wrappers (incl. fastmnn)
│   ├── scripts/                         #   classifier, per-method runs, perturbation prep
│   ├── notebooks/                       #   classification_analysis, metrics
│   └── results/                         #   eval summaries, prediction CSVs, training logs
│
├── 03_cell_cycle_analysis/              # ── Benchmarking: marker / cell-cycle preservation ──
│   ├── UMAP_Cell_Cycle_Analysis_Report.md  #   section write-up
│   ├── scripts/                         #   ccaf_comprehensive_eval, umap_cell_cycle_plots
│   ├── notebooks/                       #   cell_cycle_analysis, ccaf_classifier (ccAF)
│   ├── reference/                       #   gene-symbol→Ensembl mapping, picked genes
│   └── results/                         #   ASW UMAPs + confusion matrices vs. baseline
│
├── 04_hvg_overlap/                      # ── OBJECTIVE: Data Curation & QC (shared gene sets) ──
│   ├── scripts/                         #   hvg_overlap_analysis
│   ├── notebooks/                       #   hvg_overlap_exploration
│   └── results/                         #   per-method Jaccard matrices + plots/
│
└── 05_fastmnn/                          # ── Benchmarking: fastMNN ablation ──
    ├── scripts/                         #   fastmnn_full_obsm
    └── results/                         #   batch-corrected embedding outputs
```

Every section follows the same layout — `scripts/` (code), `notebooks/`
(exploratory), `results/` (outputs), plus `models/` where multiple methods are
compared and `reference/` for curated inputs.

> **Naming note:** original folders used the spellings `umpas`, `classifer`, and
> `scaranoma`; these were normalized to `umap`, `classifier`, and `scanorama`. Empty
> placeholders (`sample/`, `memory_overlap/overlap.ipynb`) were removed, and a ccAF
> cell-cycle notebook was moved from the classification folder into
> `03_cell_cycle_analysis/`.

---

## Mapping: project objectives → where the work lives

| Project-brief objective | Concrete tasks | Location |
|---|---|---|
| **Data Curation & QC** | Gene-ID reconciliation; shared gene sets; QC filtering | `04_hvg_overlap/` (gene-symbol→Ensembl mapping in `03_cell_cycle_analysis/reference/`) |
| **Evaluation & Benchmarking — Quantitative** | kBET, LISI, silhouette (ASW), reconstruction error | `01_harmonization_umap/` (metrics + logs) |
| **Evaluation & Benchmarking — Qualitative** | UMAP/t-SNE overlays, marker-gene preservation, downstream classification | `01_harmonization_umap/`, `02_cell_type_classification/`, `03_cell_cycle_analysis/` |
| **Downstream Integration** | Clean latent representations for perturbation/disease inference on immunotherapy data (anti-PD1/PDL1, Parse) | `02_cell_type_classification/` (perturbation analysis) |

### Research fields exercised
Computational Biology / Bioinformatics · Machine Learning / Deep Learning (VAEs) ·
Statistics (ZINB, batch diagnostics, DR metrics) · Data Engineering (10⁶⁺ cells, GPU
pipelines, cloud storage).

---

## Results summary

Benchmarks the custom harmonization model (VAE with dataset embeddings) against
**Harmony, Scanorama, scVI, scETM** on an Anti-PD1 + Parse Biosciences compendium.

### Integration metrics — raw features

| Integration            | kBET ↑ | NMI ↑ | Graph iLISI ↑ | ASW_cell | ASW_batch |
|------------------------|:------:|:-----:|:-------------:|:--------:|:---------:|
| **Harmonized (ours)**  | **0.234** | **0.551** | **5.544** | 0.285 | -0.101 |
| Harmony                | 0.033  | 0.490 | 4.028 | 0.328 | -0.092 |
| Scanorama              | 0.001  | 0.497 | 4.020 | 0.261 | -0.055 |
| PCA (unharmonized)     | 0.001  | 0.501 | 3.989 | 0.316 | -0.073 |
| scVI                   | 0.090  | 0.474 | 4.177 | 0.238 | -0.028 |

### Integration metrics — scETM embeddings

| Integration (scETM)    | kBET ↑ | NMI ↑  | Graph iLISI ↑ | ASW_cell | ASW_batch |
|------------------------|:------:|:------:|:-------------:|:--------:|:---------:|
| **Harmonized (ours)**  | **0.234** | **0.551** | **5.544** | 0.285 | -0.101 |
| Scanorama              | 0.030  | 0.486  | 4.653 | 0.388 | -0.168 |
| scVI                   | 0.026  | 0.4345 | 5.526 | 0.221 | -0.271 |
| PCA (unharmonized)     | 0.012  | 0.480  | 4.675 | 0.400 | -0.175 |
| Harmony                | 0.010  | 0.4809 | 4.698 | 0.413 | -0.228 |

### Cell-type classification (biological preservation)

| Metric       | Unharmonized | Harmonized |
|--------------|:------------:|:----------:|
| Accuracy     | 0.9001 | 0.9069 |
| F1-Macro     | 0.4335 | 0.4266 |
| F1-Weighted  | 0.8933 | 0.8986 |

Performance is essentially unchanged — biological identity is preserved, but the
metric is not sensitive enough to separate methods, which motivated the perturbation
study below.

### Perturbation recovery (KMeans vs. true perturbation labels)

| Method                | NMI ↑  | ARI ↑  | Accuracy ↑ | F1 ↑   |
|-----------------------|:------:|:------:|:----------:|:------:|
| Harmonized PCA (ours) | 0.3162 | 0.1562 | 0.3523 | **0.3042** |
| Unharmonized PCA      | 0.2605 | 0.1361 | 0.3045 | 0.2530 |
| Unharmonized scETM    | 0.2663 | 0.1305 | 0.2787 | 0.2043 |
| Scanorama (counts)    | 0.2939 | 0.1814 | 0.3292 | 0.2547 |
| Scanorama on scETM    | 0.2662 | 0.1325 | 0.2891 | 0.2274 |
| Harmony (counts PCA)  | 0.2676 | 0.1680 | 0.3084 | 0.2453 |
| Harmony on scETM      | 0.2616 | 0.1219 | 0.2808 | 0.2079 |
| scVI                  | 0.3275 | 0.1776 | 0.3542 | 0.2663 |
| scVI on scETM         | 0.2670 | 0.1542 | 0.2960 | 0.2141 |

The harmonized model and scVI are strongest at recovering perturbation states; the
harmonized model gives the best F1-score overall.

### Supporting analyses
- **Cell-cycle separation** (`03_cell_cycle_analysis/`) — ASW from −0.033
  (`X_pca_harmony`) to −0.125 (`X_scetm`); all near zero ⇒ little residual cell-cycle
  structure in any embedding.
- **HVG overlap** (`04_hvg_overlap/`) — Jaccard overlap of highly-variable genes
  before vs. after each method (baseline mean Jaccard 0.324).
- **fastMNN** (`05_fastmnn/`) — fastMNN integration ablation and batch outputs.

---

## Environment & reproduction

```bash
pip install -r environment/requirements.txt        # core
pip install -r environment/requirements_gpu.txt     # GPU: PyTorch / scVI / scETM
```

The input `.h5ad` datasets are stored in private cloud storage and are not included
in this repository; obtain them through the project's internal data access.

Run order: data inputs → `01_harmonization_umap/` → `02_cell_type_classification/` →
supporting analyses (`03`–`05`). Per-method scripts live in each section's `scripts/`;
training knobs are in `environment/config_example.yaml`.

---

## Deliverables (per project brief) — status

| Deliverable | Status in this repo |
|---|---|
| Harmonised multi-study PBMC reference atlas (AnnData) | Produced as `.h5ad` (private storage, not committed) |
| Source code (Python + PyTorch) with reproducible scripts | `01`–`05` `scripts/` + `models/`, `environment/` |
| Slide deck / manuscript draft | _pending_ |

