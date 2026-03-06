# End-to-End Transaction Risk Scoring System

A production-grade fraud detection system that minimizes expected financial loss through cost-sensitive machine learning, featuring temporal validation, calibrated probability scoring, and distribution shift monitoring.

## Project Overview

This project implements a real-world fraud detection pipeline that:

- **Optimizes for business outcomes**, not accuracy — minimizes expected financial loss under asymmetric cost structures ($500 false negative vs $10 false positive)
- **Uses temporal validation** to respect label delay and avoid data leakage
- **Produces calibrated probability scores** with optimal threshold selection based on economic cost modeling
- **Monitors distribution drift** as a deployment readiness criterion
- **Implements production-ready artifacts** with model serialization and metadata tracking

### Key Results

- **ROC-AUC**: 0.8844 on temporal test set
- **PR-AUC**: 0.5166 (under 3.5% class imbalance)
- **Optimal Threshold**: ~0.0644 (far below 0.5 default due to 50:1 cost ratio)
- **Expected Loss Reduction**: Significant improvement over accuracy-based approaches

## Problem Statement

Credit card fraud detection is an **economic optimization problem**, not a classification accuracy problem. Key insights:

- **Class Imbalance**: Only ~3.5% of transactions are fraudulent
- **Asymmetric Costs**: Missing fraud costs 50× more than false declines
- **Temporal Nature**: Fraud patterns evolve; labels arrive with ~30-day delay
- **Calibration Requirement**: Need probability scores, not just binary predictions

### Cost Structure

| Error Type | Business Impact | Unit Cost |
|------------|----------------|-----------|
| False Negative (missed fraud) | Chargeback + investigation + penalties | **$500** |
| False Positive (false decline) | Customer friction + review cost | **$10** |

The optimal decision threshold is: τ* = C_fp / (C_fn + C_fp) ≈ **0.0196** (not 0.5!)

## ️ Project Architecture

```
fraud_pipeline.ipynb # Main analysis notebook (11 sections)
├── 1. Problem Definition # Business objective, cost model
├── 2. Cost Model # Research-grounded cost constants
├── 3. Exploratory Analysis # Fraud patterns, temporal analysis
├── 4. Data Splitting # Chronological train/val/test
├── 5. Feature Engineering # Domain-driven features
├── 6. Baseline Models # Initial implementations
├── 7. Model Selection # Compare LogReg vs XGBoost
├── 8. Threshold Tuning # Cost-optimal threshold
├── 9. Calibration # Probability reliability
├── 10. Final Evaluation # Test set performance
└── 11. Deployment # Artifact serialization

data/
├── raw/ # IEEE-CIS Fraud Detection dataset (Kaggle)
├── processed/ # Feature-engineered data splits
└── artifacts/ # Model + metadata for deployment
```

## Getting Started

### Prerequisites

- Python 3.8+
- Kaggle account (for dataset access)

### Installation

1. **Clone the repository**
 ```bash
 git clone <repository-url>
 cd "Credit Card Fraud"
 ```

2. **Install dependencies**
 ```bash
 pip install -r requirements.txt
 ```

3. **Set up Kaggle API** (for dataset download)
 
 Follow [Kaggle API documentation](https://github.com/Kaggle/kaggle-api) to configure your credentials:
 ```bash
 # Place kaggle.json in:
 # Windows: C:\Users\<username>\.kaggle\kaggle.json
 # Linux/Mac: ~/.kaggle/kaggle.json
 ```

4. **Set up pre-commit hooks** (optional, recommended for development)
 
 Install pre-commit to automatically strip notebook outputs before committing:
 ```bash
 pip install pre-commit
 pre-commit install
 ```
 
 This prevents large notebook outputs from bloating the repository. Configuration is in `.pre-commit-config.yaml`.

### Usage

1. **Run the main pipeline notebook**
 
 Open `fraud_pipeline.ipynb` in Jupyter Lab or VS Code and run all cells:
 ```bash
 jupyter lab fraud_pipeline.ipynb
 ```

2. **Dataset download**
 
 The notebook automatically downloads the [IEEE-CIS Fraud Detection dataset](https://www.kaggle.com/datasets/lnasiri007/ieeecis-fraud-detection) from Kaggle on first run. Files are cached locally for subsequent runs.

3. **Output artifacts**
 
 After execution, find trained models and metadata in:
 - `data/artifacts/fraud_model.joblib` — Serialized model + preprocessing pipeline
 - `data/artifacts/fraud_model_metadata.json` — Training metrics, config, timestamps

## Methodology Highlights

### 1. Cost-Sensitive Learning

Instead of maximizing accuracy, we minimize:

```
L(τ) = |FN(τ)| × $500 + |FP(τ)| × $10
```

This fundamentally changes how we:
- Select thresholds (optimize for expected loss, not F1)
- Evaluate models (PR-AUC > ROC-AUC for imbalance)
- Train algorithms (class weights, custom loss functions)

### 2. Temporal Validation

Fraud patterns shift over time. Standard random splits leak future information. We use:

- **Chronological splits** (train → validation → test)
- **Gap windows** to respect label delay (30-day assumption)
- **Forward-only validation** (never train on future data)

### 3. Feature Engineering

Domain-driven features capturing fraud signals:

- Transaction velocity (daily/hourly counts per card)
- Amount deviations from user history
- Time-of-day patterns (fraud spikes at night)
- Device/email domain risk scores
- Cross-product interactions (card × merchant patterns)

### 4. Model Comparison

Systematic evaluation of LogisticRegression vs XGBoost:

| Model | PR-AUC | ROC-AUC | Interpretability | Speed |
|-------|--------|---------|------------------|-------|
| XGBoost | 0.5166 | 0.8844 | Medium | Fast |

Champion model selected based on PR-AUC (most relevant for imbalanced problems).

## Dataset

**Source**: [IEEE-CIS Fraud Detection (Kaggle)](https://www.kaggle.com/datasets/lnasiri007/ieeecis-fraud-detection)

- **Size**: ~590K transactions (train), ~506K transactions (test)
- **Features**: 394 anonymized features (transaction details, device info, identity)
- **Target**: `isFraud` (binary label, ~3.5% positive class)
- **Time Range**: 6 months of e-commerce transaction data

**Note**: Raw data files (~3.7GB) are excluded from this repository. Download via Kaggle API (automated in notebook).

## Key Technical Decisions

### Why These Costs?

- **$500 FN cost**: Based on research from Mastercard (2023), LexisNexis True Cost of Fraud (2023), showing 3-4× multiplier on fraud amount losses
- **$10 FP cost**: Based on Javelin Strategy & Research, MIT/BBVA studies on false decline impacts

See Section 2 of the notebook for full research citations.

### Why Temporal Splitting?

Fraud detection is fundamentally a **time-series problem**:
- Fraud tactics evolve (concept drift)
- Labels arrive delayed (label lag)
- Random splits create "omniscient" models that fail in production

### Why Logistic Regression?

Surprisingly competitive with XGBoost:
- Faster inference (critical for real-time scoring)
- Better interpretability (regulatory compliance)
- More stable calibration
- Sufficient for well-engineered features

## ️ Dependencies

Core libraries:
- **pandas**: Data manipulation
- **numpy**: Numerical computing
- **scikit-learn**: ML algorithms, metrics, preprocessing
- **xgboost**: Gradient boosting
- **matplotlib** + **seaborn**: Visualization
- **kagglehub**: Dataset download

See `requirements.txt` for complete list with versions.

## Results Summary

### Validation Performance

- **PR-AUC**: 0.5166 (key metric for imbalanced problems)
- **ROC-AUC**: 0.8844
- **Brier Score**: 0.023 (well-calibrated probabilities)

### Optimal Threshold Analysis

- **Data-optimal threshold**: 0.0644 (from grid search on validation set)
- **Theory-based threshold**: 0.0196 (from cost ratio)
- **Empirical optimal**: ~0.019 (validated on data)
- **Recall at 1% FPR**: 0.808 (catch 80% of fraud at acceptable false decline rate)

### Test Set Results

- Temporal test set (held-out, forward-looking data)
- Maintains strong performance across distribution shift
- See Section 10 of notebook for full breakdown

## Future Enhancements

**Model Improvements**:
- Ensemble methods (stacking, blending)
- Deep learning architectures (TabNet, FT-Transformer)
- Online learning for real-time adaptation

**System Features**:
- Real-time API deployment (FastAPI + Docker)
- A/B testing framework for threshold tuning
- Automated drift detection and retraining triggers
- Explainability dashboard (SHAP values, counterfactuals)

**Research Directions**:
- Multi-stage risk scoring (fast screening → detailed analysis)
- Graph neural networks for transaction networks
- Causal inference for feature selection

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **Dataset**: IEEE Computational Intelligence Society + Vesta Corporation
- **Platform**: Kaggle for hosting and API access
- **Research**: Cost estimates grounded in industry reports (Mastercard, LexisNexis, Javelin)

## Contact

For questions or collaboration opportunities, please open an issue or reach out via [your contact method].

---

**Note**: This is an educational/portfolio project demonstrating production-grade ML engineering practices. Not for use in actual fraud detection systems without proper validation, compliance review, and regulatory approval.
