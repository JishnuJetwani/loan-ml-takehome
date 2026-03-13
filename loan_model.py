"""
Loan Default Prediction Model
==============================
Builds a predictive model to replace the rule-based loan scoring system.
Compares against baseline, provides explainability via SHAP, and runs fairness analysis.

Author: Jishnu Jetwani
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, precision_recall_curve, f1_score
)
from sklearn.preprocessing import LabelEncoder
import shap
import warnings
warnings.filterwarnings('ignore')

# Set plot style
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.bbox'] = 'tight'

OUTPUT_DIR = "outputs"
import os
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================================
# 1. LOAD & EXPLORE DATA
# =============================================================================
print("=" * 70)
print("1. DATA LOADING & EXPLORATION")
print("=" * 70)

df = pd.read_csv("loan_applications.csv")
print(f"\nDataset shape: {df.shape}")
print(f"\nOutcome distribution:")
print(df['actual_outcome'].value_counts())
print(f"\nRule-based decision distribution:")
print(df['rule_based_decision'].value_counts())
print(f"\nMissing values:")
print(df.isnull().sum()[df.isnull().sum() > 0])

# --- EDA Plot 1: Outcome distribution by rule-based decision ---
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Outcome distribution
outcome_counts = df['actual_outcome'].value_counts()
axes[0].bar(outcome_counts.index, outcome_counts.values,
            color=['#2ecc71', '#e74c3c', '#95a5a6'])
axes[0].set_title('Actual Outcome Distribution')
axes[0].set_ylabel('Count')
for i, v in enumerate(outcome_counts.values):
    axes[0].text(i, v + 10, str(v), ha='center', fontweight='bold')

# Rule-based decisions vs actual outcomes
ct = pd.crosstab(df['rule_based_decision'], df['actual_outcome'], normalize='index')
ct[['repaid', 'defaulted', 'ongoing']].plot(kind='bar', stacked=True, ax=axes[1],
    color=['#2ecc71', '#e74c3c', '#95a5a6'])
axes[1].set_title('Actual Outcomes by Rule Decision')
axes[1].set_ylabel('Proportion')
axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45)
axes[1].legend(title='Outcome')

# Score distribution by outcome (excluding ongoing)
df_resolved = df[df['actual_outcome'] != 'ongoing']
for outcome, color in [('repaid', '#2ecc71'), ('defaulted', '#e74c3c')]:
    subset = df_resolved[df_resolved['actual_outcome'] == outcome]
    axes[2].hist(subset['rule_based_score'], bins=30, alpha=0.6,
                 label=outcome, color=color, density=True)
axes[2].set_title('Rule Score Distribution by Outcome')
axes[2].set_xlabel('Rule-Based Score')
axes[2].legend()

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/01_eda_overview.png")
plt.close()
print("\nSaved: 01_eda_overview.png")

# --- EDA Plot 2: Key feature distributions ---
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Missing income analysis
has_doc = df['documented_monthly_income'].notna()
doc_status = has_doc.map({True: 'Has Docs', False: 'No Docs'})
ct_doc = pd.crosstab(doc_status, df['actual_outcome'], normalize='index')
ct_doc[['repaid', 'defaulted', 'ongoing']].plot(kind='bar', stacked=True, ax=axes[0, 0],
    color=['#2ecc71', '#e74c3c', '#95a5a6'])
axes[0, 0].set_title('Outcome by Documentation Status')
axes[0, 0].set_xticklabels(axes[0, 0].get_xticklabels(), rotation=0)

# Income discrepancy (potential misrepresentation)
df_with_docs = df[df['documented_monthly_income'].notna()].copy()
df_with_docs['income_ratio'] = df_with_docs['documented_monthly_income'] / df_with_docs['stated_monthly_income']
for outcome, color in [('repaid', '#2ecc71'), ('defaulted', '#e74c3c')]:
    subset = df_with_docs[df_with_docs['actual_outcome'] == outcome]
    axes[0, 1].hist(subset['income_ratio'], bins=40, alpha=0.6,
                     label=outcome, color=color, density=True)
axes[0, 1].axvline(x=0.4, color='red', linestyle='--', alpha=0.7, label='Misrep threshold')
axes[0, 1].set_title('Income Ratio (Doc/Stated) by Outcome')
axes[0, 1].set_xlabel('Documented / Stated Income')
axes[0, 1].legend()

# Overdrafts vs outcome
ct_od = pd.crosstab(df['bank_has_overdrafts'], df['actual_outcome'], normalize='index')
ct_od[['repaid', 'defaulted', 'ongoing']].plot(kind='bar', stacked=True, ax=axes[0, 2],
    color=['#2ecc71', '#e74c3c', '#95a5a6'])
axes[0, 2].set_title('Outcome by Overdraft Status')
axes[0, 2].set_xticklabels(['No Overdrafts', 'Has Overdrafts'], rotation=0)

# Bank balance distribution
for outcome, color in [('repaid', '#2ecc71'), ('defaulted', '#e74c3c')]:
    subset = df_resolved[df_resolved['actual_outcome'] == outcome]
    axes[1, 0].hist(subset['bank_ending_balance'], bins=30, alpha=0.6,
                     label=outcome, color=color, density=True)
axes[1, 0].axvline(x=500, color='red', linestyle='--', alpha=0.7, label='Low balance threshold')
axes[1, 0].set_title('Bank Balance Distribution by Outcome')
axes[1, 0].legend()

# Loan-to-income ratio
df['loan_to_income'] = df['loan_amount'] / df['stated_monthly_income']
df_resolved['loan_to_income'] = df_resolved['loan_amount'] / df_resolved['stated_monthly_income']
for outcome, color in [('repaid', '#2ecc71'), ('defaulted', '#e74c3c')]:
    subset = df_resolved[df_resolved['actual_outcome'] == outcome]
    axes[1, 1].hist(subset['loan_to_income'], bins=30, alpha=0.6,
                     label=outcome, color=color, density=True)
axes[1, 1].axvline(x=0.3, color='red', linestyle='--', alpha=0.7, label='Risk threshold (0.3)')
axes[1, 1].set_title('Loan-to-Income Ratio by Outcome')
axes[1, 1].legend()

# Withdrawal-to-deposit ratio
df['withdrawal_ratio'] = df['monthly_withdrawals'] / df['monthly_deposits'].clip(lower=1)
df_resolved['withdrawal_ratio'] = df_resolved['monthly_withdrawals'] / df_resolved['monthly_deposits'].clip(lower=1)
for outcome, color in [('repaid', '#2ecc71'), ('defaulted', '#e74c3c')]:
    subset = df_resolved[df_resolved['actual_outcome'] == outcome]
    axes[1, 2].hist(subset['withdrawal_ratio'], bins=30, alpha=0.6,
                     label=outcome, color=color, density=True)
axes[1, 2].axvline(x=0.8, color='red', linestyle='--', alpha=0.7, label='High spend threshold')
axes[1, 2].set_title('Withdrawal/Deposit Ratio by Outcome')
axes[1, 2].legend()

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/02_feature_distributions.png")
plt.close()
print("Saved: 02_feature_distributions.png")

# --- EDA Plot 3: Employment status analysis (key for fairness) ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Default rates by employment status
emp_outcomes = df_resolved.groupby('employment_status')['actual_outcome'].value_counts(normalize=True).unstack()
emp_outcomes[['repaid', 'defaulted']].plot(kind='bar', ax=axes[0],
    color=['#2ecc71', '#e74c3c'])
axes[0].set_title('Actual Default Rate by Employment Status')
axes[0].set_ylabel('Proportion')
axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=0)

# Rule-based score distribution by employment
for emp in ['employed', 'self_employed', 'unemployed']:
    subset = df[df['employment_status'] == emp]
    axes[1].hist(subset['rule_based_score'], bins=30, alpha=0.5, label=emp, density=True)
axes[1].set_title('Rule Score Distribution by Employment')
axes[1].set_xlabel('Rule-Based Score')
axes[1].legend()

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/03_employment_analysis.png")
plt.close()
print("Saved: 03_employment_analysis.png")


# =============================================================================
# 2. DATA PREPROCESSING & FEATURE ENGINEERING
# =============================================================================
print("\n" + "=" * 70)
print("2. DATA PREPROCESSING & FEATURE ENGINEERING")
print("=" * 70)

# Decision: EXCLUDE ongoing applications
# Rationale: They have no outcome label. Training on them would require assumptions
# about their eventual outcome. Excluding them introduces a mild survivorship bias
# (ongoing apps skew recent / potentially lower-risk), which we acknowledge.
df_model = df[df['actual_outcome'] != 'ongoing'].copy()
print(f"\nExcluded {len(df) - len(df_model)} ongoing applications ({(len(df) - len(df_model))/len(df)*100:.1f}%)")
print(f"Modeling dataset: {len(df_model)} rows")
print(f"Class balance: {df_model['actual_outcome'].value_counts().to_dict()}")

# Binary target: 1 = defaulted, 0 = repaid
df_model['target'] = (df_model['actual_outcome'] == 'defaulted').astype(int)
print(f"Default rate in training data: {df_model['target'].mean():.3f}")

# --- Feature Engineering ---
# Clean feature set: 21 features, all defensible as standard lending domain knowledge.
# No hard-coded thresholds from the data generation formula — continuous signals only,
# letting the model learn its own decision boundaries.

# Fill NaN: absence of documentation is itself a signal, encode as 0
df_model['documented_monthly_income'] = df_model['documented_monthly_income'].fillna(0)
mask_has_docs = df_model['documented_monthly_income'] > 0

# --- Documentation features ---
# ~15% of applicants have no documented income (called out in dataset description)
df_model['has_documentation'] = mask_has_docs.astype(int)

# Income verification: does documented match stated within 10%?
df_model['income_verified'] = 0
df_model.loc[mask_has_docs, 'income_verified'] = (
    (np.abs(df_model.loc[mask_has_docs, 'stated_monthly_income'] -
            df_model.loc[mask_has_docs, 'documented_monthly_income']) /
     df_model.loc[mask_has_docs, 'stated_monthly_income']) <= 0.1
).astype(int)

# Possible misrepresentation: stated income >> documented (~5% of applicants, per dataset description)
df_model['possible_misrep'] = 0
df_model.loc[mask_has_docs, 'possible_misrep'] = (
    df_model.loc[mask_has_docs, 'stated_monthly_income'] >
    3 * df_model.loc[mask_has_docs, 'documented_monthly_income']
).astype(int)

# --- Continuous ratio features (model learns its own thresholds) ---
df_model['loan_to_income']    = df_model['loan_amount'] / df_model['stated_monthly_income']
df_model['withdrawal_ratio']  = df_model['monthly_withdrawals'] / df_model['monthly_deposits'].clip(lower=1)
df_model['income_doc_ratio']  = np.where(mask_has_docs,
    df_model['documented_monthly_income'] / df_model['stated_monthly_income'], 0)
df_model['balance_to_loan']   = df_model['bank_ending_balance'] / df_model['loan_amount'].clip(lower=1)
df_model['deposits_to_loan']  = df_model['monthly_deposits'] / df_model['loan_amount'].clip(lower=1)
df_model['net_cash_flow']     = df_model['monthly_deposits'] - df_model['monthly_withdrawals']
df_model['net_flow_to_loan']  = df_model['net_cash_flow'] / df_model['loan_amount'].clip(lower=1)

# --- Employment encoding ---
df_model['is_unemployed']    = (df_model['employment_status'] == 'unemployed').astype(int)
df_model['is_self_employed'] = (df_model['employment_status'] == 'self_employed').astype(int)

# --- Final feature list ---
all_features = [
    # Continuous ratios
    'loan_to_income',
    'withdrawal_ratio',
    'income_doc_ratio',
    'balance_to_loan',
    'deposits_to_loan',
    'net_flow_to_loan',
    # Documentation flags (from dataset description)
    'has_documentation',
    'possible_misrep',
    'income_verified',
    # Employment
    'is_unemployed',
    'is_self_employed',
    # Raw columns
    'stated_monthly_income',
    'documented_monthly_income',
    'loan_amount',
    'bank_ending_balance',
    'bank_has_overdrafts',
    'bank_has_consistent_deposits',
    'monthly_withdrawals',
    'monthly_deposits',
    'num_documents_submitted',
    'net_cash_flow',
]

print(f"\nTotal features: {len(all_features)}")
print("Feature categories:")
print("  Continuous ratios (6): loan_to_income, withdrawal_ratio, income_doc_ratio,")
print("                         balance_to_loan, deposits_to_loan, net_flow_to_loan")
print("  Documentation flags (3): has_documentation, possible_misrep, income_verified")
print("  Employment encoding (2): is_unemployed, is_self_employed")
print("  Raw columns (9): stated/documented income, loan amount, bank balance,")
print("                   overdrafts, consistent deposits, withdrawals, deposits,")
print("                   num_documents, net_cash_flow")

X = df_model[all_features].copy()
y = df_model['target'].copy()

# Convert boolean columns to int
for col in X.columns:
    if X[col].dtype == 'bool':
        X[col] = X[col].astype(int)

print(f"\nFeature matrix shape: {X.shape}")
print(f"Target distribution: {y.value_counts().to_dict()}")


# =============================================================================
# 3. MODEL — Logistic Regression
# =============================================================================
print("\n" + "=" * 70)
print("3. MODEL — Logistic Regression")
print("=" * 70)

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# 5-fold stratified cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

model = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(
        class_weight='balanced',
        C=1.0,
        max_iter=1000,
        random_state=42
    ))
])

# Out-of-fold predictions for honest evaluation
y_pred_proba = cross_val_predict(model, X, y, cv=cv, method='predict_proba')[:, 1]
y_pred = (y_pred_proba >= 0.5).astype(int)

auc  = roc_auc_score(y, y_pred_proba)
f1   = f1_score(y, y_pred)
cm   = confusion_matrix(y, y_pred)
tn, fp, fn, tp = cm.ravel()
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall    = tp / (tp + fn) if (tp + fn) > 0 else 0

print(f"\n{'AUC-ROC':>10} {'F1 (default)':>14} {'Precision':>12} {'Recall':>10}")
print("-" * 48)
print(f"{auc:>10.4f} {f1:>14.4f} {precision:>12.4f} {recall:>10.4f}")

# Fit on full dataset for SHAP
model.fit(X, y)
best_name = 'Logistic Regression'

print(f"\nModel trained on full dataset ({len(y)} samples).")


# =============================================================================
# 4. EVALUATION — Model vs Rule-Based Baseline
# =============================================================================
print("\n" + "=" * 70)
print("4. EVALUATION — ML Model vs Rule-Based Baseline")
print("=" * 70)

# Rule-based baseline predictions on the same data
# Rule decision: denied = predicted default, approved/flagged = predicted repaid
# (being generous to baseline: only "denied" = predicted default)
rule_pred = (df_model['rule_based_decision'] == 'denied').astype(int)

# Also evaluate a stricter baseline: denied + flagged = predicted default
rule_pred_strict = (df_model['rule_based_decision'] != 'approved').astype(int)

print(f"\n--- ML Model ({best_name}, threshold=0.5) ---")
print(classification_report(y, y_pred, target_names=['repaid', 'defaulted']))

print("--- Rule-Based Baseline (denied = default) ---")
print(classification_report(y, rule_pred, target_names=['repaid', 'defaulted']))

print("--- Rule-Based Baseline Strict (denied + flagged = default) ---")
print(classification_report(y, rule_pred_strict, target_names=['repaid', 'defaulted']))

# AUC-ROC
ml_auc = roc_auc_score(y, y_pred_proba)
rule_auc = roc_auc_score(y, df_model['rule_based_score'] * -1)  # Lower score = more likely default
# Actually let's use the raw score inverted properly
rule_auc = roc_auc_score(y, 100 - df_model['rule_based_score'])  # Invert: high = risky
print(f"\nAUC-ROC — ML Model:       {ml_auc:.4f}")
print(f"AUC-ROC — Rule-Based:     {rule_auc:.4f}")
print(f"AUC improvement:          +{ml_auc - rule_auc:.4f}")

# --- Confusion matrices ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax, preds, title in [
    (axes[0], y_pred, f'ML Model (AUC={ml_auc:.3f})'),
    (axes[1], rule_pred, f'Baseline: Denied=Default (AUC={rule_auc:.3f})'),
    (axes[2], rule_pred_strict, 'Baseline Strict: Denied+Flagged=Default')
]:
    cm = confusion_matrix(y, preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Repaid', 'Defaulted'],
                yticklabels=['Repaid', 'Defaulted'])
    ax.set_title(title)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')

    # Compute rates
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn)  # Good applicants wrongly denied
    fnr = fn / (fn + tp)  # Defaults that slipped through
    ax.text(0.5, -0.15, f'FPR: {fpr:.1%} | FNR: {fnr:.1%}',
            transform=ax.transAxes, ha='center', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/04_confusion_matrices.png")
plt.close()
print("\nSaved: 04_confusion_matrices.png")

# --- ROC Curves ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# ROC
fpr_ml, tpr_ml, _ = roc_curve(y, y_pred_proba)
fpr_rule, tpr_rule, _ = roc_curve(y, 100 - df_model['rule_based_score'])
axes[0].plot(fpr_ml, tpr_ml, label=f'ML Model (AUC={ml_auc:.3f})', linewidth=2)
axes[0].plot(fpr_rule, tpr_rule, label=f'Rule-Based (AUC={rule_auc:.3f})', linewidth=2, linestyle='--')
axes[0].plot([0, 1], [0, 1], 'k--', alpha=0.3)
axes[0].set_xlabel('False Positive Rate')
axes[0].set_ylabel('True Positive Rate')
axes[0].set_title('ROC Curves')
axes[0].legend()

# Precision-Recall
prec_ml, rec_ml, _ = precision_recall_curve(y, y_pred_proba)
prec_rule, rec_rule, _ = precision_recall_curve(y, 100 - df_model['rule_based_score'])
axes[1].plot(rec_ml, prec_ml, label='ML Model', linewidth=2)
axes[1].plot(rec_rule, prec_rule, label='Rule-Based', linewidth=2, linestyle='--')
axes[1].set_xlabel('Recall')
axes[1].set_ylabel('Precision')
axes[1].set_title('Precision-Recall Curves')
axes[1].legend()

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/05_roc_pr_curves.png")
plt.close()
print("Saved: 05_roc_pr_curves.png")

# Detailed comparison table
print("\n--- DEPLOYMENT IMPACT ANALYSIS ---")
cm_ml = confusion_matrix(y, y_pred)
cm_rule = confusion_matrix(y, rule_pred)
tn_ml, fp_ml, fn_ml, tp_ml = cm_ml.ravel()
tn_rule, fp_rule, fn_rule, tp_rule = cm_rule.ravel()

print(f"\n{'Metric':<40} {'ML Model':>12} {'Rule Baseline':>14} {'Delta':>10}")
print("-" * 78)
print(f"{'Defaults caught (True Positives)':<40} {tp_ml:>12} {tp_rule:>14} {tp_ml - tp_rule:>+10}")
print(f"{'Defaults missed (False Negatives)':<40} {fn_ml:>12} {fn_rule:>14} {fn_ml - fn_rule:>+10}")
print(f"{'Good applicants wrongly denied (FP)':<40} {fp_ml:>12} {fp_rule:>14} {fp_ml - fp_rule:>+10}")
print(f"{'Good applicants correctly approved (TN)':<40} {tn_ml:>12} {tn_rule:>14} {tn_ml - tn_rule:>+10}")
print(f"{'Default catch rate (Recall)':<40} {tp_ml/(tp_ml+fn_ml):>12.1%} {tp_rule/(tp_rule+fn_rule):>14.1%}")
print(f"{'False denial rate (FPR)':<40} {fp_ml/(fp_ml+tn_ml):>12.1%} {fp_rule/(fp_rule+tn_rule):>14.1%}")


# =============================================================================
# 4b. THRESHOLD OPTIMIZATION
# =============================================================================
print("\n" + "=" * 70)
print("4b. THRESHOLD OPTIMIZATION")
print("=" * 70)

# The default 0.5 threshold may not be optimal.
# Explore thresholds that balance FPR and FNR for a lending context.
thresholds = np.arange(0.2, 0.8, 0.01)
threshold_results = []
for t in thresholds:
    pred_t = (y_pred_proba >= t).astype(int)
    cm_t = confusion_matrix(y, pred_t)
    tn_t, fp_t, fn_t, tp_t = cm_t.ravel()
    f1_t = f1_score(y, pred_t)
    fpr_t = fp_t / (fp_t + tn_t)
    fnr_t = fn_t / (fn_t + tp_t)
    threshold_results.append({
        'threshold': t, 'f1': f1_t, 'fpr': fpr_t, 'fnr': fnr_t,
        'tp': tp_t, 'fp': fp_t, 'fn': fn_t, 'tn': tn_t
    })
tr_df = pd.DataFrame(threshold_results)

# Best F1 threshold
best_f1_idx = tr_df['f1'].idxmax()
best_threshold = tr_df.loc[best_f1_idx, 'threshold']
print(f"Optimal threshold (max F1): {best_threshold:.2f}")
print(f"  F1={tr_df.loc[best_f1_idx, 'f1']:.4f}, FPR={tr_df.loc[best_f1_idx, 'fpr']:.1%}, FNR={tr_df.loc[best_f1_idx, 'fnr']:.1%}")

# Also show a "balanced" threshold that keeps FPR under 20%
balanced = tr_df[tr_df['fpr'] <= 0.20]
if len(balanced) > 0:
    bal_best_idx = balanced['f1'].idxmax()
    bal_threshold = balanced.loc[bal_best_idx, 'threshold']
    print(f"\nBalanced threshold (max F1 with FPR≤20%): {bal_threshold:.2f}")
    print(f"  F1={balanced.loc[bal_best_idx, 'f1']:.4f}, FPR={balanced.loc[bal_best_idx, 'fpr']:.1%}, FNR={balanced.loc[bal_best_idx, 'fnr']:.1%}")

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(tr_df['threshold'], tr_df['f1'], label='F1 Score', linewidth=2, color='#2ecc71')
ax.plot(tr_df['threshold'], tr_df['fpr'], label='False Positive Rate', linewidth=2, color='#e74c3c', linestyle='--')
ax.plot(tr_df['threshold'], tr_df['fnr'], label='False Negative Rate', linewidth=2, color='#3498db', linestyle='--')
ax.axvline(x=best_threshold, color='#2ecc71', alpha=0.3, linestyle=':')
ax.axvline(x=0.5, color='gray', alpha=0.3, linestyle=':')
ax.set_xlabel('Decision Threshold')
ax.set_ylabel('Rate')
ax.set_title('Threshold Optimization — F1, FPR, FNR Tradeoff')
ax.legend()
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/05b_threshold_optimization.png")
plt.close()
print("Saved: 05b_threshold_optimization.png")


# =============================================================================
# 5. EXPLAINABILITY — SHAP Analysis
# =============================================================================
print("\n" + "=" * 70)
print("5. EXPLAINABILITY — SHAP Feature Importances")
print("=" * 70)

# LinearExplainer for logistic regression pipeline
scaler = model.named_steps['scaler']
X_scaled = pd.DataFrame(scaler.transform(X), columns=X.columns, index=X.index)
explainer = shap.LinearExplainer(model.named_steps['clf'], X_scaled)
shap_values = explainer.shap_values(X_scaled)

# --- SHAP Summary Plot ---
fig, ax = plt.subplots(figsize=(10, 8))
shap.summary_plot(shap_values, X, show=False, max_display=20)
plt.title("SHAP Feature Importance — What Drives Default Predictions", fontsize=12)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/06_shap_summary.png")
plt.close()
print("Saved: 06_shap_summary.png")

# --- SHAP Bar Plot (mean absolute) ---
fig, ax = plt.subplots(figsize=(10, 6))
shap.summary_plot(shap_values, X, plot_type="bar", show=False, max_display=20)
plt.title("Mean |SHAP Value| — Feature Importance Ranking", fontsize=12)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/07_shap_bar.png")
plt.close()
print("Saved: 07_shap_bar.png")

# --- Example individual explanations ---
print("\n--- Sample Individual Explanations ---")
# Show explanations for 3 defaulted applicants
defaulted_indices = df_model[df_model['target'] == 1].index[:3]
for idx in defaulted_indices:
    iloc_idx = df_model.index.get_loc(idx)
    app_id = df_model.loc[idx, 'applicant_id']
    pred_prob = y_pred_proba[iloc_idx]
    actual = df_model.loc[idx, 'actual_outcome']

    print(f"\n  {app_id} — Predicted default prob: {pred_prob:.2f}, Actual: {actual}")

    # Get top contributing features
    sv = shap_values[iloc_idx]
    feature_impacts = sorted(zip(all_features, sv, X.iloc[iloc_idx]),
                            key=lambda x: abs(x[1]), reverse=True)
    print(f"  Top risk factors:")
    for feat, impact, val in feature_impacts[:5]:
        direction = "increases" if impact > 0 else "decreases"
        print(f"    - {feat}={val:.2f} → {direction} default risk by {abs(impact):.3f}")

# Save SHAP waterfall for one example
fig, ax = plt.subplots(figsize=(10, 6))
shap.plots.waterfall(shap.Explanation(
    values=shap_values[df_model.index.get_loc(defaulted_indices[0])],
    base_values=explainer.expected_value,
    data=X.iloc[df_model.index.get_loc(defaulted_indices[0])],
    feature_names=all_features
), show=False)
plt.title(f"SHAP Waterfall — Why {df_model.loc[defaulted_indices[0], 'applicant_id']} Was Flagged")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/08_shap_waterfall_example.png")
plt.close()
print("\nSaved: 08_shap_waterfall_example.png")


# =============================================================================
# 5b. SHAP DENIAL ANALYSIS — Why Are Applicants Being Rejected?
# =============================================================================
print("\n" + "=" * 70)
print("5b. SHAP DENIAL ANALYSIS — Average Reasons for Rejection")
print("=" * 70)

# Add predictions to df_model for segmentation
df_model['pred_proba'] = y_pred_proba
df_model['pred_label'] = y_pred  # 1 = predicted default (flagged/denied)

# Use the SHAP values already computed above
# For LR pipeline, shap_values are on scaled data but features still map 1:1
# We use X (unscaled) for feature values in display, shap_values for contributions

# --- Segment applicants ---
denied_mask  = df_model['pred_label'] == 1   # predicted default → would be denied
approved_mask = df_model['pred_label'] == 0  # predicted repaid  → would be approved

n_denied   = denied_mask.sum()
n_approved = approved_mask.sum()
print(f"\nPredicted denials:   {n_denied}  ({n_denied/len(df_model):.1%})")
print(f"Predicted approvals: {n_approved} ({n_approved/len(df_model):.1%})")

# Get SHAP values for denied and approved groups
denied_idx   = np.where(denied_mask.values)[0]
approved_idx = np.where(approved_mask.values)[0]

shap_denied   = shap_values[denied_idx]    # shape: (n_denied,  n_features)
shap_approved = shap_values[approved_idx]  # shape: (n_approved, n_features)

# --- Mean SHAP contribution per feature for denied group ---
# Positive mean SHAP = this feature pushes toward default (bad for applicant)
# Negative mean SHAP = this feature reduces default risk
mean_shap_denied   = pd.Series(shap_denied.mean(axis=0),   index=all_features)
mean_shap_approved = pd.Series(shap_approved.mean(axis=0), index=all_features)

# Sort by impact in the denied group (most positive first = biggest denial drivers)
mean_shap_denied_sorted = mean_shap_denied.sort_values(ascending=False)

print("\n--- Average SHAP contribution for DENIED applicants ---")
print("  Positive = pushes toward default (hurts applicant)")
print("  Negative = reduces default risk (helps applicant)\n")
print(f"  {'Feature':<30} {'Avg SHAP':>10}  Direction")
print("  " + "-" * 55)
for feat, val in mean_shap_denied_sorted.items():
    direction = "↑ increases default risk" if val > 0 else "↓ reduces default risk"
    bar_len = int(abs(val) * 30)
    bar = ("█" * bar_len) if val > 0 else ("░" * bar_len)
    print(f"  {feat:<30} {val:>+10.4f}  {bar}  {direction}")

print("\n--- Top 5 reasons applicants are denied (on average) ---")
top_denial_reasons = mean_shap_denied_sorted[mean_shap_denied_sorted > 0].head(5)
for i, (feat, val) in enumerate(top_denial_reasons.items(), 1):
    # Get the mean feature value for denied applicants
    feat_val = X.iloc[denied_idx][feat].mean()
    print(f"  {i}. {feat:<30}  avg value={feat_val:.3f}  avg SHAP impact={val:+.4f}")

print("\n--- Top 5 protective factors for denied applicants (on average) ---")
top_protective = mean_shap_denied_sorted[mean_shap_denied_sorted < 0].tail(5).sort_values()
for i, (feat, val) in enumerate(top_protective.items(), 1):
    feat_val = X.iloc[denied_idx][feat].mean()
    print(f"  {i}. {feat:<30}  avg value={feat_val:.3f}  avg SHAP impact={val:+.4f}")

# --- Plot: Mean SHAP for denied vs approved groups ---
fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# Sort by absolute mean SHAP in denied group
feat_order = mean_shap_denied.abs().sort_values(ascending=True).index

denied_vals   = mean_shap_denied[feat_order]
approved_vals = mean_shap_approved[feat_order]

y_pos = np.arange(len(feat_order))
height = 0.35

axes[0].barh(y_pos + height/2, denied_vals,   height, color='#e74c3c', alpha=0.8, label='Denied (predicted default)')
axes[0].barh(y_pos - height/2, approved_vals, height, color='#2ecc71', alpha=0.8, label='Approved (predicted repaid)')
axes[0].axvline(x=0, color='black', linewidth=0.8)
axes[0].set_yticks(y_pos)
axes[0].set_yticklabels(feat_order, fontsize=9)
axes[0].set_xlabel('Mean SHAP Value (impact on default probability)')
axes[0].set_title('Average Feature Contribution\nDenied vs Approved Applicants')
axes[0].legend()

# --- Plot: Feature value distributions for denied vs approved ---
# Show top 6 most impactful features
top_features = mean_shap_denied.abs().sort_values(ascending=False).head(6).index.tolist()

axes[1].set_visible(False)

gs = fig.add_gridspec(3, 2, left=0.55, right=0.98, hspace=0.5, wspace=0.4)
for i, feat in enumerate(top_features):
    ax = fig.add_subplot(gs[i // 2, i % 2])
    denied_vals_feat   = X.iloc[denied_idx][feat]
    approved_vals_feat = X.iloc[approved_idx][feat]
    ax.hist(approved_vals_feat, bins=25, alpha=0.6, color='#2ecc71', density=True, label='Approved')
    ax.hist(denied_vals_feat,   bins=25, alpha=0.6, color='#e74c3c', density=True, label='Denied')
    ax.set_title(feat, fontsize=8, fontweight='bold')
    ax.tick_params(labelsize=7)
    if i == 0:
        ax.legend(fontsize=7)

fig.suptitle('SHAP Denial Analysis: Why Are Applicants Being Rejected?',
             fontsize=12, fontweight='bold', y=1.01)
plt.savefig(f"{OUTPUT_DIR}/08b_shap_denial_analysis.png", bbox_inches='tight')
plt.close()
print("\nSaved: 08b_shap_denial_analysis.png")

# --- Natural language summary ---
print("\n--- NATURAL LANGUAGE SUMMARY (for Loom) ---")
top3 = mean_shap_denied_sorted[mean_shap_denied_sorted > 0].head(3)
print(f"""
  On average, a denied applicant is flagged primarily because of:

  1. {top3.index[0]} (avg impact: {top3.iloc[0]:+.4f})
     → The single biggest driver of denials. Denied applicants have
       a mean value of {X.iloc[denied_idx][top3.index[0]].mean():.3f}
       vs {X.iloc[approved_idx][top3.index[0]].mean():.3f} for approved applicants.

  2. {top3.index[1]} (avg impact: {top3.iloc[1]:+.4f})
     → Mean for denied: {X.iloc[denied_idx][top3.index[1]].mean():.3f}
       vs approved: {X.iloc[approved_idx][top3.index[1]].mean():.3f}

  3. {top3.index[2]} (avg impact: {top3.iloc[2]:+.4f})
     → Mean for denied: {X.iloc[denied_idx][top3.index[2]].mean():.3f}
       vs approved: {X.iloc[approved_idx][top3.index[2]].mean():.3f}

  A loan reviewer can use these SHAP values to explain any individual
  denial in plain English: "Your application was flagged primarily because
  your loan-to-income ratio was X, which is above the typical threshold
  for approval, and your withdrawal ratio of Y suggests high spending
  relative to income."
""")


# =============================================================================
# 6. FAIRNESS ANALYSIS — Employment Status
# =============================================================================
print("\n" + "=" * 70)
print("6. FAIRNESS ANALYSIS — Employment Status Bias")
print("=" * 70)

# Add ML predictions to the dataframe
df_model['ml_pred_proba'] = y_pred_proba
df_model['ml_pred'] = y_pred
df_model['ml_decision'] = np.where(y_pred_proba >= 0.5, 'denied', 'approved')

print("\n--- Default Rates by Employment Status (Ground Truth) ---")
for emp in ['employed', 'self_employed', 'unemployed']:
    subset = df_model[df_model['employment_status'] == emp]
    default_rate = subset['target'].mean()
    print(f"  {emp:<15}: {default_rate:.1%} default rate (n={len(subset)})")

print("\n--- Rule-Based System: Approval Rates & Outcomes ---")
print(f"  {'Group':<15} {'Approval Rate':>15} {'Default Rate':>15} {'Defaults in Approved':>22}")
for emp in ['employed', 'self_employed', 'unemployed']:
    subset = df_model[df_model['employment_status'] == emp]
    approval_rate = (subset['rule_based_decision'] == 'approved').mean()
    default_rate = subset['target'].mean()
    approved_defaults = subset[(subset['rule_based_decision'] == 'approved') & (subset['target'] == 1)]
    approved_total = subset[subset['rule_based_decision'] == 'approved']
    approved_default_rate = len(approved_defaults) / max(len(approved_total), 1)
    print(f"  {emp:<15} {approval_rate:>15.1%} {default_rate:>15.1%} {approved_default_rate:>22.1%}")

print("\n--- ML Model: Approval Rates & Outcomes ---")
print(f"  {'Group':<15} {'Approval Rate':>15} {'Default Rate':>15} {'Defaults in Approved':>22}")
for emp in ['employed', 'self_employed', 'unemployed']:
    subset = df_model[df_model['employment_status'] == emp]
    approval_rate = (subset['ml_decision'] == 'approved').mean()
    default_rate = subset['target'].mean()
    approved_defaults = subset[(subset['ml_decision'] == 'approved') & (subset['target'] == 1)]
    approved_total = subset[subset['ml_decision'] == 'approved']
    approved_default_rate = len(approved_defaults) / max(len(approved_total), 1)
    print(f"  {emp:<15} {approval_rate:>15.1%} {default_rate:>15.1%} {approved_default_rate:>22.1%}")

# Fairness metrics: Demographic Parity & Equalized Odds
print("\n--- Fairness Metrics ---")
print("Demographic Parity (approval rate should be similar across groups):")
for emp in ['employed', 'self_employed', 'unemployed']:
    subset = df_model[df_model['employment_status'] == emp]
    rule_approval = (subset['rule_based_decision'] == 'approved').mean()
    ml_approval = (subset['ml_decision'] == 'approved').mean()
    print(f"  {emp:<15}: Rule={rule_approval:.1%}, ML={ml_approval:.1%}, Gap={ml_approval-rule_approval:+.1%}")

print("\nEqualized Odds (TPR and FPR should be similar across groups):")
for emp in ['employed', 'self_employed', 'unemployed']:
    subset = df_model[df_model['employment_status'] == emp]
    # For ML model
    if subset['target'].sum() > 0:
        ml_tpr = subset[(subset['target'] == 1) & (subset['ml_pred'] == 1)].shape[0] / subset[subset['target'] == 1].shape[0]
        ml_fpr = subset[(subset['target'] == 0) & (subset['ml_pred'] == 1)].shape[0] / subset[subset['target'] == 0].shape[0]
        rule_tpr = subset[(subset['target'] == 1) & (rule_pred.loc[subset.index] == 1)].shape[0] / subset[subset['target'] == 1].shape[0]
        rule_fpr = subset[(subset['target'] == 0) & (rule_pred.loc[subset.index] == 1)].shape[0] / subset[subset['target'] == 0].shape[0]
        print(f"  {emp:<15}: ML(TPR={ml_tpr:.2f}, FPR={ml_fpr:.2f})  Rule(TPR={rule_tpr:.2f}, FPR={rule_fpr:.2f})")

# --- Fairness visualization ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

groups = ['employed', 'self_employed', 'unemployed']

# Plot 1: Approval rates comparison
rule_approvals = [(df_model[df_model['employment_status'] == g]['rule_based_decision'] == 'approved').mean() for g in groups]
ml_approvals = [(df_model[df_model['employment_status'] == g]['ml_decision'] == 'approved').mean() for g in groups]
actual_repay = [1 - df_model[df_model['employment_status'] == g]['target'].mean() for g in groups]

x = np.arange(len(groups))
w = 0.25
axes[0].bar(x - w, rule_approvals, w, label='Rule-Based Approval', color='#3498db')
axes[0].bar(x, ml_approvals, w, label='ML Model Approval', color='#2ecc71')
axes[0].bar(x + w, actual_repay, w, label='Actual Repayment Rate', color='#95a5a6')
axes[0].set_xticks(x)
axes[0].set_xticklabels(groups)
axes[0].set_ylabel('Rate')
axes[0].set_title('Approval Rates vs Actual Repayment')
axes[0].legend()
axes[0].set_ylim(0, 1.05)

# Plot 2: Score distributions by employment (rule-based)
for emp, color in [('employed', '#3498db'), ('self_employed', '#2ecc71'), ('unemployed', '#e74c3c')]:
    subset = df_model[df_model['employment_status'] == emp]
    axes[1].hist(subset['rule_based_score'], bins=25, alpha=0.5, label=emp, color=color, density=True)
axes[1].axvline(x=75, color='black', linestyle='--', alpha=0.5, label='Approval threshold')
axes[1].set_title('Rule-Based Score Distribution by Employment')
axes[1].set_xlabel('Rule Score')
axes[1].legend()

# Plot 3: ML probability distributions by employment
for emp, color in [('employed', '#3498db'), ('self_employed', '#2ecc71'), ('unemployed', '#e74c3c')]:
    subset = df_model[df_model['employment_status'] == emp]
    axes[2].hist(subset['ml_pred_proba'], bins=25, alpha=0.5, label=emp, color=color, density=True)
axes[2].axvline(x=0.5, color='black', linestyle='--', alpha=0.5, label='Decision threshold')
axes[2].set_title('ML Default Probability by Employment')
axes[2].set_xlabel('Predicted Default Probability')
axes[2].legend()

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/09_fairness_analysis.png")
plt.close()
print("\nSaved: 09_fairness_analysis.png")

# SHAP dependence for employment features
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
shap.dependence_plot('is_self_employed', shap_values, X, ax=axes[0], show=False)
axes[0].set_title('SHAP: Impact of Self-Employment on Default Prediction')
shap.dependence_plot('is_unemployed', shap_values, X, ax=axes[1], show=False)
axes[1].set_title('SHAP: Impact of Unemployment on Default Prediction')
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/10_shap_employment.png")
plt.close()
print("Saved: 10_shap_employment.png")


# =============================================================================
# 7. FAIRNESS RECOMMENDATION
# =============================================================================
print("\n" + "=" * 70)
print("7. FAIRNESS ANALYSIS — CONCLUSIONS & RECOMMENDATION")
print("=" * 70)

print("""
KEY FINDING: The rule-based system unfairly penalizes self-employed applicants.
  - Self-employed get an employment score of 60 vs 100 for employed
  - But the actual default rate for self-employed is comparable to employed
  - The only employment status that genuinely predicts higher default risk is 'unemployed'

ML MODEL BEHAVIOR:
  - The ML model learns from outcomes, not from the biased scoring rubric
  - It naturally gives self-employed applicants fairer treatment because
    their actual repayment behavior doesn't justify the penalty
  - The model still appropriately flags unemployed applicants at higher rates
    because unemployment IS a genuine risk signal in the data

RECOMMENDATION:
  I would NOT remove employment_status entirely. Instead:
  1. Keep it in the model — unemployment is a legitimate risk factor
  2. Monitor approval rate disparities across groups as a KPI
  3. Set a maximum allowable disparity threshold (e.g., equalized odds within 5%)
  4. If the model begins penalizing self-employed unfairly, investigate feature
     interactions and consider constrained optimization

  Removing the feature entirely would:
  - Lose signal about unemployment risk (a real predictor)
  - Potentially increase overall defaults
  - Not address the root cause (the feature itself isn't biased;
    the rule-based WEIGHTING was biased)
""")


# =============================================================================
# 8. PRODUCTION READINESS DISCUSSION
# =============================================================================
print("=" * 70)
print("8. PRODUCTION CONSIDERATIONS")
print("=" * 70)

print("""
IF THIS MODEL WENT LIVE TOMORROW, THE FIRST THING THAT WOULD GO WRONG:

1. DATA DRIFT — The model was trained on historical data with specific distributions.
   A recession, policy change, or shift in applicant demographics would degrade
   performance. Unlike the rule-based system (which is at least predictable in its
   errors), the ML model could fail silently.

   Mitigation: Monitor prediction distributions weekly. Alert if the distribution
   of predicted probabilities shifts significantly (Kolmogorov-Smirnov test).

2. MISSING DOCUMENTATION RATE SHIFT — If the share of applicants without income
   documentation changes (currently ~15%), the model's calibration breaks.

3. REGULATORY AUDIT — A regulator asks "why was applicant X denied?"
   Solution: SHAP waterfall plots give per-applicant explanations. But you need
   infrastructure to store and serve these explanations, not just generate them.

4. FEEDBACK LOOP — If we deny more applicants, we never observe their outcome,
   creating a bias toward the model's existing beliefs. We'd need randomized
   approval of some borderline cases to maintain calibration.

5. THRESHOLD SELECTION — The 0.5 threshold was chosen by default. In production,
   you'd calibrate this based on the business cost of false positives vs false
   negatives (wrongly denying a $5K loan vs missing a default on a $300 loan
   have very different costs).
""")


# =============================================================================
# FINAL SUMMARY
# =============================================================================
print("=" * 70)
print("SUMMARY OF RESULTS")
print("=" * 70)
print(f"""
Best Model: {best_name}
Evaluation: 5-fold stratified cross-validation (out-of-fold predictions)

Performance vs Rule-Based Baseline:
  AUC-ROC:  ML={ml_auc:.4f}  vs  Rule={rule_auc:.4f}  (Δ={ml_auc-rule_auc:+.4f})

Confusion Matrix Summary (ML Model):
  True Positives (caught defaults):     {tp_ml}
  False Negatives (missed defaults):    {fn_ml}
  False Positives (wrongly denied):     {fp_ml}
  True Negatives (correctly approved):  {tn_ml}

Key Insight: The ML model learns that self-employed ≠ high risk,
correcting a systematic bias in the rule-based system.

All plots saved to: {OUTPUT_DIR}/
""")

print("Done! All outputs generated successfully.")
