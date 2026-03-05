# Multiple Myeloma Survival Prediction & Biomarker Discovery

## ⚠️ Disclaimer

**This project is not a medical product and is not intended for clinical use.**

This repository is a portfolio project demonstrating the application of
explainable AI (XAI) techniques to high-dimensional genomic data. Its purpose
is to showcase a complete ML pipeline including data extraction, exploratory
analysis, modeling, and SHAP-based explainability — not to provide clinical
predictions or medical guidance.

The models developed here:
- Have not been clinically validated
- Are based on a limited dataset (340 training patients)
- Use a surrogate time variable for Kaplan-Meier analysis instead of
  real follow-up times
- Should not be used to inform medical decisions under any circumstances

For research use only.

## AI Assistance

This project was developed with the assistance of [Claude](https://claude.ai)
(Anthropic)

All results, analyses and conclusions were reviewed and validated by the author.
Biological interpretations of specific genes (particularly less characterized
lncRNAs such as LOC646762, ASH1L-AS1) are based on general molecular biology
knowledge and should be validated against primary literature before drawing
clinical conclusions.

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
│   └── 04_explainability.ipynb    # SHAP biomarker analysis
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

### 4. Explainability — SHAP (✅ Complete)

SHAP (SHapley Additive exPlanations) was applied to the best model
(XGBoost + Strategy B + EFS 24m) on the 214-patient validation set.

**Top genes by mean absolute SHAP value:**

| Rank | Gene | Mean |SHAP| | Direction | Biological role |
|------|------|-------------|-----------|-----------------|
| 1 | LOC646762 | 0.438 | Protective | Uncharacterized lncRNA — novel candidate |
| 2 | AC005523.3 | 0.346 | Risk | Uncharacterized lncRNA |
| 3 | NMU | 0.296 | Mixed | Neuropeptide, tumor progression |
| 4 | BCAR3 | 0.290 | Risk | Treatment resistance adaptor |
| 5 | ASH1L-AS1 | 0.289 | Protective | Epigenetic regulation (lncRNA) |
| 6 | ARPC5 | 0.276 | Risk | Cell migration (Arp2/3 complex) |
| 7 | WDR12 | 0.254 | Risk | RNA processing |
| 8 | ZWILCH | 0.219 | Risk | Kinetochore — cell division ✅ EDA |
| 9 | TCRBV15S1 | 0.193 | Protective | T-cell receptor — immune signal ✅ EDA |

**Validation against EDA findings:**
- ZWILCH (cell division) confirms the proliferation signal from Cluster 12 ✅
- TCRBV15S1 (T-cell receptor) confirms the immune protective signal from Cluster 15 ✅

**Local explanations (waterfall plots):**
- High risk patient (predicted: 0.991, actual event: 1) — model correctly identifies
  risk driven by BCAR3, LOC646762, NMU, YWHAH all pushing in the same direction
- Low risk patient (predicted: 0.005, actual event: 0) — LOC646762 low expression
  is the dominant protective factor (-1.17 SHAP)
- Median risk patient (predicted: 0.145 ≈ base rate 0.14) — mixed signals,
  model appropriately defaults to average when evidence is ambiguous

**Kaplan-Meier survival curves:**
Patients stratified by median expression of top SHAP genes:

| Gene | p-value | Significant | Finding |
|------|---------|-------------|---------|
| ASH1L-AS1 | 0.021 | ✅ | High expression → better EFS |
| LOC646762 | 0.080 | borderline | Trend towards better EFS |
| BCAR3 | 0.390 | ❌ | No significant individual effect |
| AC005523.3 | 0.442 | ❌ | No significant individual effect |
| NMU | 0.396 | ❌ | No significant individual effect |
| ARPC5 | 0.243 | ❌ | No significant individual effect |

**Key finding — ASH1L-AS1:**
The only gene with statistically significant survival stratification (p=0.021).
High expression of this antisense lncRNA to ASH1L histone methyltransferase
is associated with better EFS — suggesting a protective epigenetic regulatory
role in MM progression. Novel candidate biomarker for future investigation.

**Methodological note:**
Risk rank was used as a surrogate for follow-up time since exact survival
times are not available in GSE24080. Results should be validated with
actual survival time data (available in MMRF CoMMpass).

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
| Explainability (SHAP) | ✅ Complete |
| Deployment | 🔜 Planned |

## References
Mulligan G. et al. "Gene expression profiling and correlation with outcome in clinical
trials of the proteasome inhibitor bortezomib." Blood. 2007;109(8):3177-88.
https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE24080
