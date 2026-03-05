# Multiple Myeloma Survival Prediction & Biomarker Discovery

## Overview
End-to-end machine learning pipeline for predicting survival outcomes in Multiple
Myeloma patients using gene expression data (transcriptomics). The project combines
survival analysis, dimensionality reduction, and explainable AI to identify
prognostic biomarkers from high-dimensional genomic data.

This project is the second in a series focused on AI applications in Multiple Myeloma,
complementing the clinical staging project
[myeloma-explainability](https://github.com/miguelm71/myeloma-explainability).

## Dataset
**GSE24080 — MAQC-II Multiple Myeloma Dataset**
Mulligan G. et al., Blood 2007 (PMID: 20676074)
NCBI GEO: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE24080

- 559 Multiple Myeloma patients from the APEX clinical trial
- Gene expression profiles generated with Affymetrix Human Genome U133 Plus 2.0 Array
- Pre-defined Training (340) / Validation (214) split
- Clinical endpoints: Overall Survival and Event-Free Survival at 24 months

## Clinical Context
Survival outcomes in newly diagnosed Multiple Myeloma are highly heterogeneous —
overall survival ranges from months to over 10 years. Gene expression profiling
can capture molecular subtypes that drive this heterogeneity, enabling more
accurate prognosis and personalized treatment decisions.

**Survival endpoints:**
| Endpoint | Positive (event) | Negative (no event) |
|----------|-----------------|---------------------|
| OS 24m (Overall Survival) | 78 deceased (14%) | 481 alive (86%) |
| EFS 24m (Event-Free Survival) | 118 events (21%) | 441 no event (79%) |

## Project Structure
```
myeloma-survival-genomics/
├── data/                          # Processed data (not versioned)
│   ├── GSE24080_family.soft.gz    # Raw GEO SOFT file
│   ├── clinical_data.csv          # 559 patients x 7 clinical features
│   ├── expression_matrix.csv      # 559 patients x 21,655 genes
│   ├── probe_annotations.csv      # 42,986 probe → gene symbol mappings
│   ├── gene_clusters.csv          # 5,000 genes → 20 cluster assignments
│   └── model_results.csv          # All 18 baseline + 3 tuned model results
├── notebooks/
│   ├── 01_data_download.ipynb     # Data extraction and preprocessing
│   ├── 02_eda.ipynb               # Exploratory Data Analysis
│   ├── 03_modeling.ipynb          # Survival modeling
│   └── 04_explainability.ipynb    # SHAP biomarker analysis (planned)
├── models/
│   ├── xgb_B_efs_tuned.joblib     # Best model: XGBoost + Strategy B + EFS
│   ├── xgb_A_os_tuned.joblib      # XGBoost + Strategy A + OS
│   ├── features_A.joblib          # Feature names for Strategy A
│   └── features_B.joblib          # Feature names for Strategy B
├── docs/
│   └── soft_structure.html        # Interactive GEO SOFT file structure diagram
├── src/                           # Source code
├── mlruns/                        # MLflow experiment tracking (not versioned)
└── README.md
```

## Methodology

### 1. Data Extraction (✅ Complete)
- Download GSE24080 SOFT file from NCBI GEO
- Parse clinical metadata: age, sex, OS/EFS outcomes, Training/Validation split
- Extract gene expression matrix: 559 patients × 54,675 probes
- Annotate probes to gene symbols using GPL570 platform table
- Deduplicate multi-probe genes (keep highest mean expression probe)
- Final matrix: 559 patients × 21,655 unique genes

### 2. Exploratory Data Analysis (✅ Complete)

**Clinical data:**
- 559 patients, age 57.2 ± 9.5 years, 60.3% male
- OS events: 78 (14%), EFS events: 118 (21%)
- No missing values, consistent normalization across all 559 arrays

**Gene co-expression clustering:**
20 gene clusters identified from top 5,000 most variable genes.
Two clusters showed meaningful association with survival outcomes:

| Cluster | Correlation OS | Correlation EFS | Biological interpretation |
|---------|---------------|-----------------|--------------------------|
| Cluster 12 | +0.224 | +0.272 | Tumor proliferation signature |
| Cluster 15 | -0.123 | -0.169 | Immune activity + CCND1 subtype |

**Cluster 12 — Tumor Proliferation Signature (127 genes):**
Contains key oncogenes and cell cycle drivers associated with poor prognosis:
MYC, CDK1, CCNB2, BIRC5 (Survivin), FOXM1, NEK2, RRM2, CDC20, KIF18A, UHRF1.
Higher expression → faster tumor proliferation → worse survival.

**Cluster 15 — Protective Signature (199 genes):**
Contains immune activity markers and MM molecular subtype genes associated
with better prognosis: CD4, CD79A, CXCR5, SYK, TLR3, TLR8, CCND1, EDNRB.

**Dimensionality reduction:**
- PCA: variance highly distributed across components — no single dominant
  survival-associated direction. PC1 captures mixed biological/technical variation.
- UMAP: no discrete molecular subtypes, continuous patient cloud. Cluster 12
  score shows a clear spatial gradient confirming its biological relevance.

### 3. Modeling (✅ Complete)

**Feature Engineering — Three Parallel Strategies:**

| Strategy | Approach | Features |
|----------|----------|----------|
| A — Cluster scores | Mean expression of 20 gene co-expression clusters | 20 |
| B — LASSO strict | LassoCV automatic alpha selection (α=0.076) | 31 |
| C — LASSO flexible | Manual alpha (α=0.030) for richer gene selection | 246 |

**Models:**
Three classifiers evaluated for each strategy and target:
- **Logistic Regression** — baseline, most interpretable
- **Random Forest** — robust with small datasets, handles class imbalance
- **XGBoost** — generally most powerful on tabular data

Class imbalance addressed with `class_weight='balanced'` (LR, RF)
and `scale_pos_weight` (XGBoost): 5.7x for OS, 3.1x for EFS.

**AUC on Validation set (214 patients):**

| Model | Strategy | OS 24m | EFS 24m |
|-------|----------|--------|---------|
| LogReg | A — Clusters | 0.631 | 0.641 |
| LogReg | B — LASSO strict | 0.628 | 0.676 |
| LogReg | C — LASSO flexible | 0.656 | 0.667 |
| RandomForest | A — Clusters | 0.628 | 0.611 |
| RandomForest | B — LASSO strict | 0.607 | 0.677 |
| RandomForest | C — LASSO flexible | 0.679 | 0.658 |
| XGBoost | A — Clusters | 0.584 | 0.619 |
| XGBoost | B — LASSO strict | 0.593 | **0.681** |
| XGBoost | C — LASSO flexible | 0.637 | 0.654 |

**Best model after hyperparameter tuning:**
XGBoost + Strategy B (LASSO strict, 31 genes) + EFS 24m
- Validation AUC: **0.702**
- Validation F1_macro: **0.604**

**Key findings:**
- EFS is more predictable than OS — more events (24% vs 15%) and stronger
  molecular signal
- Strategy B (31 LASSO genes) outperforms larger feature sets — aggressive
  regularization selects the most relevant genes and avoids noise
- Performance ceiling at AUC ~0.70 — determined by dataset size (340 training
  patients), not model complexity. Hyperparameter tuning showed CV AUC of 0.810
  but validation AUC of 0.702, indicating overfitting risk with small datasets
- Accessing larger datasets (MMRF CoMMpass, ~1000 patients) would be expected
  to significantly improve generalization

**Experiment Tracking:**
All 18 baseline combinations + 3 tuned models tracked with MLflow.
Dataset metadata, model parameters, and artifacts logged for full reproducibility.

### 4. Explainability (SHAP)
- Identify top prognostic genes from SHAP values
- Validate findings against known MM biomarkers
- Kaplan-Meier survival curves stratified by gene expression

## Documentation
- [GEO SOFT File Structure](docs/soft_structure.html) — interactive diagram explaining
  the format of the raw data file and how GEOparse maps it to Python objects

## Tech Stack
- Python 3.13
- GEOparse, pandas, numpy
- scikit-learn, XGBoost, lifelines
- SHAP, MLflow
- FastAPI (planned)

## Status
| Phase | Status |
|-------|--------|
| Data extraction & preprocessing | ✅ Complete |
| EDA | ✅ Complete |
| Modeling | ✅ Complete |
| Explainability (SHAP) | 🔜 Planned |
| Deployment | 🔜 Planned |

## References
Mulligan G. et al. "Gene expression profiling and correlation with outcome in clinical
trials of the proteasome inhibitor bortezomib." Blood. 2007;109(8):3177-88.
https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE24080
