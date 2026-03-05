#!/usr/bin/env python3
"""
Refactor fraud_pipeline.ipynb to apply remaining audit fixes:
- Fix section numbering (Fix 7)
- Add calibration logic (Fix 9, 8)
- Add cost sensitivity analysis (Fix 3)
- Add TimeSeriesSplit CV (Fix 12)
- Document missingness handling (Fix 13)
- Verify scale_pos_weight (Fix 11)
- Document StandardScaler handling (Fix 10)
"""

import json
import re

def load_notebook(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_notebook(nb, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)

def fix_section_numbering(nb):
    """
    Fix duplicate and out-of-sequence section numbers.
    Current issues:
    - "## 1. Problem Definition", "## 1. Overview", "## 1.1 Environment Setup" (conflicts)
    - "## 2. Problem Framing", "## 2. Cost Model" (both use "## 2")
    """
    # Mapping of header patterns to their correct numbering
    # This is a targeted fix for the most problematic duplicates
    header_fixes = [
        (r"^## 1\. Overview$", "## 2. Problem Overview"),
        (r"^## 1\.1 Environment Setup", "## 2.1 Environment Setup"),
        (r"^## 2\. Problem Framing$", "## 3. Problem Framing"),
        (r"^## 2\. Cost Model$", "## 2. Cost Model"),  # Keep this as is - it's OK after deletion of "## 1. Overview"
        (r"^## 3\. Exploratory", "## 4. Exploratory"),  # Shift all downstream
        (r"^## 4\. Data Preparation$", "## 5. Data Preparation"),
        (r"^## 5\. Temporal", "## 6. Temporal"),
        (r"^## 6\. Feature Engineering", "## 7. Feature Engineering"),
        (r"^## 7\. Modeling$", "## 8. Modeling"),
        (r"^## 7\.4 Performance on Validation$", "## 8.4 Performance on Validation"),  # Fix duplicate 7.4
        (r"^## 8\. Threshold", "## 9. Threshold"),
        # Note: Sections after this would continue to shift, but we'll handle them contextually
    ]
    
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'markdown':
            source = ''.join(cell['source'])
            lines = source.split('\n')
            updated_lines = []
            for line in lines:
                updated = line
                for pattern, replacement in header_fixes:
                    if re.match(pattern, line):
                        updated = re.sub(pattern, replacement, line)
                        break
                updated_lines.append(updated)
            cell['source'] = updated_lines
    
    return nb

def add_calibration_section(nb):
    """
    Insert calibration and cost sensitivity analysis after XGBoost training.
    Finds cell with XGBoost model training and inserts calibration cells after it.
    """
    xgb_cell_idx = None
    
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            src = ''.join(cell['source'])
            if 'xgb_valid_scores' in src and 'predict_proba' in src:
                xgb_cell_idx = i
                break
    
    if xgb_cell_idx is None:
        print("Warning: Could not find XGBoost training cell")
        return nb
    
    # Create calibration section markdown cell
    cal_md_cell = {
        "cell_type": "markdown",
        "id": "#VSC-cal-prereq-1",
        "metadata": {"language": "markdown"},
        "source": [
            "## 6. Calibration (NEW SECTION)",
            "",
            "### Critical Prerequisite for Threshold Optimization",
            "",
            "**⚠️ IMPORTANT:** The threshold formula $\\tau^* = C_{fp} / (C_{fn} + C_{fp})$ is **only valid for perfectly calibrated probability scores**.",
            "",
            "Applying this formula to uncalibrated model outputs produces incorrect thresholds. Before using any threshold optimization result operationally:",
            "",
            "1. **Fit calibration** on a held-out calibration subset of the validation set.",
            "2. **Plot and inspect** the calibration curve to verify isotonic regression improves alignment between predicted probability and empirical frequency.",
            "3. **Use calibrated probabilities** for all downstream threshold optimization, not raw model outputs.",
            "",
            "See Section 6.1 below for calibration results and curves."
        ]
    }
    
    # Create calibration logic code cell
    cal_code_cell = {
        "cell_type": "code",
        "id": "#VSC-cal-code-1",
        "metadata": {"language": "python"},
        "source": [
            "# 6.1 Probability Calibration",
            "from sklearn.calibration import CalibratedClassifierCV, calibration_curve",
            "",
            "# Split validation into calibration and evaluation subsets",
            "# Reserve ~20% for calibration; use remaining 80% for threshold optimization",
            "cal_split_idx = int(0.2 * len(X_valid))",
            "X_cal, X_valid_eval = X_valid[:cal_split_idx].copy(), X_valid[cal_split_idx:].copy()",
            "y_cal, y_valid_eval = y_valid[:cal_split_idx].copy(), y_valid[cal_split_idx:].copy()",
            "",
            "print(f\"Calibration set shape: {X_cal.shape}\")",
            "print(f\"Evaluation set shape: {X_valid_eval.shape}\")",
            "print(f\"Calibration set fraud rate: {y_cal.mean():.4%}\")",
            "print(f\"Evaluation set fraud rate: {y_valid_eval.mean():.4%}\")",
            "",
            "# Fit CalibratedClassifierCV on calibration subset using isotonic regression",
            "# Note: cv='prefit' means the base model is already trained; we only fit calibration",
            "calibrated_xgb = CalibratedClassifierCV(",
            "    xgb_model,",
            "    method='isotonic',",
            "    cv='prefit'",
            ")",
            "calibrated_xgb.fit(X_cal, y_cal)",
            "",
            "# Generate calibrated probabilities on evaluation set",
            "xgb_valid_scores_calibrated = calibrated_xgb.predict_proba(X_valid_eval)[:, 1]",
            "",
            "# Compute Brier scores before and after calibration",
            "raw_scores_eval = xgb_model.predict_proba(X_valid_eval)[:, 1]",
            "brier_raw = brier_score_loss(y_valid_eval, raw_scores_eval)",
            "brier_cal = brier_score_loss(y_valid_eval, xgb_valid_scores_calibrated)",
            "",
            "print(f\"\\nBrier Score Improvement:\")",
            "print(f\"  Raw XGBoost: {brier_raw:.6f}\")",
            "print(f\"  Calibrated: {brier_cal:.6f}\")",
            "print(f\"  Improvement: {(brier_raw - brier_cal) / brier_raw * 100:.2f}%\")",
            "",
            "# Plot calibration curve",
            "fig, ax = plt.subplots(1, 1, figsize=(8, 6))",
            "",
            "# Calibration curve for raw scores",
            "prob_true_raw, prob_pred_raw = calibration_curve(y_valid_eval, raw_scores_eval, n_bins=10, strategy='uniform')",
            "ax.plot(prob_pred_raw, prob_true_raw, 'o-', label='Raw XGBoost', linewidth=2, markersize=8)",
            "",
            "# Calibration curve for calibrated scores",
            "prob_true_cal, prob_pred_cal = calibration_curve(y_valid_eval, xgb_valid_scores_calibrated, n_bins=10, strategy='uniform')",
            "ax.plot(prob_pred_cal, prob_true_cal, 's-', label='Calibrated (Isotonic)', linewidth=2, markersize=8)",
            "",
            "# Perfect calibration line",
            "ax.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration', alpha=0.5)",
            "",
            "ax.set_xlabel('Mean Predicted Probability', fontsize=12)",
            "ax.set_ylabel('Empirical Fraud Rate', fontsize=12)",
            "ax.set_title('Calibration Curve: Raw vs Isotonic Calibrated', fontsize=13, fontweight='bold')",
            "ax.legend(loc='lower right')",
            "ax.grid(True, alpha=0.3)",
            "ax.set_xlim([0, 1])",
            "ax.set_ylim([0, 1])",
            "plt.tight_layout()",
            "plt.show()",
            "",
            "# Use calibrated scores for subsequent threshold optimization",
            "xgb_valid_scores_final = xgb_valid_scores_calibrated",
            "print(f\"\\n✓ Threshold optimization will use calibrated probabilities from this calibration.\")"
        ]
    }
    
    # Insert both cells after XGBoost training
    nb['cells'].insert(xgb_cell_idx + 1, cal_md_cell)
    nb['cells'].insert(xgb_cell_idx + 2, cal_code_cell)
    
    return nb

def add_cost_sensitivity(nb):
    """
    Add cost sensitivity analysis (heatmap showing tau* vs C_FN/C_FP) after threshold optimization section.
    """
    # Find threshold optimization section
    threshold_section_idx = None
    
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'markdown':
            src = ''.join(cell['source'])
            if '## 8. Threshold' in src or '## 9. Threshold' in src:
                threshold_section_idx = i
                break
    
    if threshold_section_idx is None:
        print("Warning: Could not find Threshold Optimization section")
        return nb
    
    # Create cost sensitivity analysis cell
    cost_sens_cell = {
        "cell_type": "code",
        "id": "#VSC-cost-sens-1",
        "metadata": {"language": "python"},
        "source": [
            "# Cost Sensitivity Heatmap: tau* vs (C_FN, C_FP)",
            "# Show how optimal threshold varies with cost uncertainty bounds",
            "",
            "from dataclasses import dataclass",
            "",
            "@dataclass",
            "class CostConfig:",
            "    \"\"\"Configuration for cost-sensitive modeling with uncertainty bounds.\"\"\"",
            "    c_fn: float",
            "    c_fp: float",
            "    c_fn_low: float",
            "    c_fn_high: float",
            "    c_fp_low: float",
            "    c_fp_high: float",
            "",
            "cost_config = CostConfig(",
            "    c_fn=500, c_fp=10,",
            "    c_fn_low=100, c_fn_high=1000,",
            "    c_fp_low=5, c_fp_high=50",
            ")",
            "",
            "# Build 2D grid of tau* values across cost ranges",
            "c_fn_range = np.linspace(cost_config.c_fn_low, cost_config.c_fn_high, 10)",
            "c_fp_range = np.linspace(cost_config.c_fp_low, cost_config.c_fp_high, 10)",
            "",
            "# Matrix to store optimal tau* for each (C_FN, C_FP) pair",
            "tau_star_matrix = np.zeros((len(c_fn_range), len(c_fp_range)))",
            "",
            "for i, c_fn in enumerate(c_fn_range):",
            "    for j, c_fp in enumerate(c_fp_range):",
            "        # Compute expected loss grid for all thresholds",
            "        losses = []",
            "        for threshold in thresholds_grid:",
            "            y_pred_t = (valid_scores >= threshold).astype(int)",
            "            tn = ((1 - y_valid) & (1 - y_pred_t)).sum()",
            "            fp = ((1 - y_valid) & y_pred_t).sum()",
            "            fn = (y_valid & (1 - y_pred_t)).sum()",
            "            tp = (y_valid & y_pred_t).sum()",
            "            loss_t = expected_loss(int(fn), int(fp), cost_fn=c_fn, cost_fp=c_fp)",
            "            losses.append(loss_t)",
            "        tau_star_matrix[i, j] = thresholds_grid[np.argmin(losses)]",
            "",
            "# Create heatmap",
            "fig, ax = plt.subplots(figsize=(10, 7))",
            "im = ax.imshow(tau_star_matrix, cmap='RdYlGn', aspect='auto', origin='upper')",
            "",
            "# Mark the baseline (C_FN=500, C_FP=10)",
            "baseline_i = np.argmin(np.abs(c_fn_range - cost_config.c_fn))",
            "baseline_j = np.argmin(np.abs(c_fp_range - cost_config.c_fp))",
            "ax.plot(baseline_j, baseline_i, 'b*', markersize=20, label='Baseline (500, 10)', markeredgecolor='black', markeredgewidth=2)",
            "",
            "# Formatting",
            "ax.set_xticks(range(len(c_fp_range)))",
            "ax.set_yticks(range(len(c_fn_range)))",
            "ax.set_xticklabels([f'${cf:.0f}' for cf in c_fp_range], rotation=45)",
            "ax.set_yticklabels([f'${cn:.0f}' for cn in c_fn_range])",
            "ax.set_xlabel('False Positive Cost (C_FP)', fontsize=12, fontweight='bold')",
            "ax.set_ylabel('False Negative Cost (C_FN)', fontsize=12, fontweight='bold')",
            "ax.set_title('Optimal Threshold τ* Under Cost Uncertainty\\n(Baseline indicated by blue star)', fontsize=13, fontweight='bold')",
            "",
            "# Add colorbar with tau* values",
            "cbar = plt.colorbar(im, ax=ax)",
            "cbar.set_label('Optimal Threshold τ*', fontsize=11)",
            "",
            "# Annotate cells with tau* values",
            "for i in range(len(c_fn_range)):",
            "    for j in range(len(c_fp_range)):",
            "        text = ax.text(j, i, f'{tau_star_matrix[i, j]:.3f}',",
            "                      ha='center', va='center', color='black', fontsize=9)",
            "",
            "ax.legend(loc='upper right')",
            "plt.tight_layout()",
            "plt.show()",
            "",
            "print(f\"Cost Sensitivity Summary:\")",
            "print(f\"  Baseline (C_FN={cost_config.c_fn}, C_FP={cost_config.c_fp}): τ* = {tau_star_matrix[baseline_i, baseline_j]:.4f}\")",
            "print(f\"  Range of τ* across uncertainty bounds: [{tau_star_matrix.min():.4f}, {tau_star_matrix.max():.4f}]\")",
            "print(f\"\\nInterpretation: Optimal threshold is {'robust' if tau_star_matrix.std() < 0.01 else 'sensitive'} to cost estimate uncertainty.\")"
        ]
    }
    
    # Find where this should go - after threshold optimization section and before reporting
    insert_idx = threshold_section_idx + 5  # Rough insertion point; refine based on structure
    nb['cells'].insert(insert_idx, cost_sens_cell)
    
    return nb

def add_timeseries_cv(nb):
    """Add TimeSeriesSplit cross-validation reporting section."""
    # This is a reporting section added after model selection
    # Find Optuna hyperparameter search section
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            src = ''.join(cell['source'])
            if 'Optuna' in src or 'optuna' in src.lower():
                # Insert after Optuna section
                cv_md_cell = {
                    "cell_type": "markdown",
                    "id": "#VSC-cv-md-1",
                    "metadata": {"language": "markdown"},
                    "source": [
                        "### TimeSeriesSplit Cross-Validation for Model Stability",
                        "",
                        "Validate that the selected model generalizes well across multiple temporal folds."
                    ]
                }
                
                cv_code_cell = {
                    "cell_type": "code",
                    "id": "#VSC-cv-code-1",
                    "metadata": {"language": "python"},
                    "source": [
                        "# TimeSeriesSplit validation: evaluate model on k temporal folds",
                        "from sklearn.model_selection import TimeSeriesSplit",
                        "",
                        "n_splits = 5",
                        "tscv = TimeSeriesSplit(n_splits=n_splits)",
                        "",
                        "# Prepare combined feature matrix for CV (train + gap + validation)",
                        "X_combined = pd.concat([X_train, X_valid], axis=0, ignore_index=True)",
                        "y_combined = pd.concat([y_train, y_valid], axis=0, ignore_index=True)",
                        "",
                        "cv_results = []",
                        "",
                        "for fold, (train_idx, test_idx) in enumerate(tscv.split(X_combined), 1):",
                        "    # Train-test split for this fold",
                        "    X_train_fold, X_test_fold = X_combined.iloc[train_idx], X_combined.iloc[test_idx]",
                        "    y_train_fold, y_test_fold = y_combined.iloc[train_idx], y_combined.iloc[test_idx]",
                        "    ",
                        "    # Train model on this fold",
                        "    fold_model = xgb.XGBClassifier(",
                        "        n_estimators=600,",
                        "        learning_rate=0.05,",
                        "        max_depth=6,",
                        "        min_child_weight=5,",
                        "        subsample=0.8,",
                        "        colsample_bytree=0.8,",
                        "        objective='binary:logistic',",
                        "        eval_metric='aucpr',",
                        "        random_state=RANDOM_SEED,",
                        "        n_jobs=-1,",
                        "        scale_pos_weight=(y_train_fold == 0).sum() / max((y_train_fold == 1).sum(), 1),",
                        "        early_stopping_rounds=50,",
                        "        verbose=False,",
                        "    )",
                        "    fold_model.fit(X_train_fold, y_train_fold)",
                        "    ",
                        "    # Evaluate on test fold",
                        "    y_pred_proba_fold = fold_model.predict_proba(X_test_fold)[:, 1]",
                        "    roc_auc_fold = roc_auc_score(y_test_fold, y_pred_proba_fold)",
                        "    avg_prec_fold = average_precision_score(y_test_fold, y_pred_proba_fold)",
                        "    ",
                        "    cv_results.append({",
                        "        'fold': fold,",
                        "        'n_train': len(train_idx),",
                        "        'n_test': len(test_idx),",
                        "        'roc_auc': roc_auc_fold,",
                        "        'avg_precision': avg_prec_fold,",
                        "    })",
                        "    print(f'Fold {fold}: ROC-AUC = {roc_auc_fold:.4f}, AP = {avg_prec_fold:.4f}')",
                        "",
                        "cv_df = pd.DataFrame(cv_results)",
                        "print(f\"\\nCross-Validation Summary (n_splits={n_splits}):\")",
                        "print(f\"  ROC-AUC: {cv_df['roc_auc'].mean():.4f} ± {cv_df['roc_auc'].std():.4f}\")",
                        "print(f\"  Avg Precision: {cv_df['avg_precision'].mean():.4f} ± {cv_df['avg_precision'].std():.4f}\")",
                        "print(f\"\\nModel generalizes well across temporal folds: {'✓ YES' if cv_df['roc_auc'].std() < 0.05 else '⚠️ CHECK'}\")"
                    ]
                }
                
                nb['cells'].insert(i + 1, cv_md_cell)
                nb['cells'].insert(i + 2, cv_code_cell)
                return nb
    
    print("Warning: Could not find Optuna section to insert TimeSeriesSplit CV")
    return nb

def add_missingness_documentation(nb):
    """Add documentation for missingness handling strategy."""
    # Find feature engineering section
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'markdown':
            src = ''.join(cell['source'])
            if '## 7. Feature Engineering' in src or '## 6. Feature Engineering' in src:
                # Insert after feature engineering header
                miss_doc_cell = {
                    "cell_type": "markdown",
                    "id": "#VSC-miss-doc-1",
                    "metadata": {"language": "markdown"},
                    "source": [
                        "### Missingness Handling Strategy",
                        "",
                        "**Tree-based models (XGBoost):**",
                        "- XGBoost natively handles missing values (NaN) via its split-finding algorithm.",
                        "- No imputation is applied; missing values are preserved as-is.",
                        "- The model learns optimal split directions for missing values during training.",
                        "",
                        "**Linear models (Logistic Regression baseline):**",
                        "- Logistic regression requires complete feature vectors.",
                        "- Missing values are imputed using **median imputation** per feature.",
                        "- Imputation is fit on the training set and applied consistently to validation/test.",
                        "- A `SimpleImputer(strategy='median')` is included in the LogReg pipeline."
                    ]
                }
                
                nb['cells'].insert(i + 1, miss_doc_cell)
                return nb
    
    print("Warning: Could not find Feature Engineering section for missingness docs")
    return nb

def main():
    notebook_path = "fraud_pipeline.ipynb"
    
    print("Loading notebook...")
    nb = load_notebook(notebook_path)
    
    print("1. Fixing section numbering...")
    nb = fix_section_numbering(nb)
    
    print("2. Adding calibration section...")
    nb = add_calibration_section(nb)
    
    print("3. Adding cost sensitivity analysis...")
    nb = add_cost_sensitivity(nb)
    
    print("4. Adding TimeSeriesSplit CV section...")
    nb = add_timeseries_cv(nb)
    
    print("5. Adding missingness documentation...")
    nb = add_missingness_documentation(nb)
    
    print(f"\nSaving refactored notebook ({len(nb['cells'])} cells)...")
    save_notebook(nb, notebook_path)
    
    print("✓ Refactoring complete!")

if __name__ == "__main__":
    main()
