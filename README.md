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
│   └── probe_annotations.csv      # 42,986 probe → gene symbol mappings
├── notebooks/
│   ├── 01_data_download.ipynb     # Data extraction and preprocessing
│   ├── 02_eda.ipynb               # Exploratory Data Analysis (planned)
│   ├── 03_modeling.ipynb          # Survival modeling (planned)
│   └── 04_explainability.ipynb    # SHAP biomarker analysis (planned)
├── docs/
│   └── soft_structure.html        # Interactive GEO SOFT file structure diagram
├── src/                           # Source code
├── models/                        # Trained models
└── README.md
```

## Methodology (planned)

### 1. Data Extraction (✅ Complete)
- Download GSE24080 SOFT file from NCBI GEO
- Parse clinical metadata: age, sex, OS/EFS outcomes, Training/Validation split
- Extract gene expression matrix: 559 patients × 54,675 probes
- Annotate probes to gene symbols using GPL570 platform table
- Deduplicate multi-probe genes (keep highest mean expression probe)
- Final matrix: 559 patients × 21,655 unique genes

### 2. Exploratory Data Analysis
- Clinical data distribution: age, sex, survival outcomes
- Expression data quality: missing values, normalization check
- Dimensionality reduction: PCA and UMAP visualization
- Identify natural patient clusters in expression space

### 3. Modeling
- Target: OS 24m and EFS 24m (binary classification)
- Feature selection: LASSO regression to reduce 21,655 genes to key predictors
- Models: Random Forest, XGBoost with survival-aware evaluation
- Experiment tracking: MLflow

### 4. Explainability (SHAP)
- Identify top prognostic genes from SHAP values
- Validate findings against known MM biomarkers (B2M, CCND1, TP53)
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
| EDA | 🔜 Planned |
| Modeling | 🔜 Planned |
| Explainability (SHAP) | 🔜 Planned |
| Deployment | 🔜 Planned |

## References
Mulligan G. et al. "Gene expression profiling and correlation with outcome in clinical
trials of the proteasome inhibitor bortezomib." Blood. 2007;109(8):3177-88.
https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE24080