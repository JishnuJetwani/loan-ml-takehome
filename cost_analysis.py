"""
Business Cost Analysis
======================
Translates model predictions into dollar terms.

Cost assumptions (standard consumer lending):
  - LOAN_MARGIN: interest profit from a successfully repaid loan
                 (8% annual rate × 2-year avg term = ~16% of loan amount)
  - LOSS_GIVEN_DEFAULT: fraction of loan amount lost on default
                        (standard LGD in unsecured consumer lending is 60-70%)

At each decision threshold we compute:
  - TN (correctly approve repayer)  → +margin per loan
  - FN (approve a defaulter)        → -LGD per loan  [most expensive error]
  - FP (reject a good repayer)      → -margin per loan (lost opportunity)
  - TP (correctly reject defaulter) → $0

Then compare: our model at optimal threshold vs rule-based baseline vs approve-all.

Author: Jishnu Jetwani
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.bbox'] = 'tight'

OUTPUT_DIR = "outputs"
import os
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================================
# COST ASSUMPTIONS
# =============================================================================
LOAN_MARGIN       = 0.16   # profit earned on a repaid loan (8% × 2 years)
LOSS_GIVEN_DEFAULT = 0.65  # fraction of loan amount lost on default

print("=" * 70)
print("BUSINESS COST ANALYSIS")
print("=" * 70)
print(f"\nCost assumptions:")
print(f"  Revenue per repaid loan  = {LOAN_MARGIN*100:.0f}% of loan amount")
print(f"  Loss per defaulted loan  = {LOSS_GIVEN_DEFAULT*100:.0f}% of loan amount")

# =============================================================================
# 1. LOAD DATA & FEATURE ENGINEERING (mirrors loan_model.py exactly)
# =============================================================================
df = pd.read_csv("loan_applications.csv")
df_model = df[df['actual_outcome'] != 'ongoing'].copy()
df_model['target'] = (df_model['actual_outcome'] == 'defaulted').astype(int)

df_model['documented_monthly_income'] = df_model['documented_monthly_income'].fillna(0)
mask_has_docs = df_model['documented_monthly_income'] > 0

df_model['has_documentation'] = mask_has_docs.astype(int)
df_model['income_verified'] = 0
df_model.loc[mask_has_docs, 'income_verified'] = (
    (np.abs(df_model.loc[mask_has_docs, 'stated_monthly_income'] -
            df_model.loc[mask_has_docs, 'documented_monthly_income']) /
     df_model.loc[mask_has_docs, 'stated_monthly_income']) <= 0.1
).astype(int)
df_model['possible_misrep'] = 0
df_model.loc[mask_has_docs, 'possible_misrep'] = (
    df_model.loc[mask_has_docs, 'stated_monthly_income'] >
    3 * df_model.loc[mask_has_docs, 'documented_monthly_income']
).astype(int)

df_model['loan_to_income']   = df_model['loan_amount'] / df_model['stated_monthly_income']
df_model['withdrawal_ratio'] = df_model['monthly_withdrawals'] / df_model['monthly_deposits'].clip(lower=1)
df_model['income_doc_ratio'] = np.where(mask_has_docs,
    df_model['documented_monthly_income'] / df_model['stated_monthly_income'], 0)
df_model['balance_to_loan']  = df_model['bank_ending_balance'] / df_model['loan_amount'].clip(lower=1)
df_model['deposits_to_loan'] = df_model['monthly_deposits'] / df_model['loan_amount'].clip(lower=1)
df_model['net_cash_flow']    = df_model['monthly_deposits'] - df_model['monthly_withdrawals']
df_model['net_flow_to_loan'] = df_model['net_cash_flow'] / df_model['loan_amount'].clip(lower=1)
df_model['is_unemployed']    = (df_model['employment_status'] == 'unemployed').astype(int)
df_model['is_self_employed'] = (df_model['employment_status'] == 'self_employed').astype(int)

all_features = [
    'loan_to_income', 'withdrawal_ratio', 'income_doc_ratio', 'balance_to_loan',
    'deposits_to_loan', 'net_flow_to_loan', 'has_documentation', 'possible_misrep',
    'income_verified', 'is_unemployed', 'is_self_employed',
    'stated_monthly_income', 'documented_monthly_income', 'loan_amount',
    'bank_ending_balance', 'bank_has_overdrafts', 'bank_has_consistent_deposits',
    'monthly_withdrawals', 'monthly_deposits', 'num_documents_submitted', 'net_cash_flow',
]

X = df_model[all_features].copy()
y = df_model['target'].copy()
loan_amounts = df_model['loan_amount'].values  # keep dollar amounts aligned

for col in X.columns:
    if X[col].dtype == 'bool':
        X[col] = X[col].astype(int)

# =============================================================================
# 2. GET OUT-OF-FOLD PREDICTED PROBABILITIES (5-fold CV)
# =============================================================================
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('lr', LogisticRegression(C=1.0, class_weight='balanced', max_iter=1000, random_state=42))
])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
y_prob = cross_val_predict(pipe, X, y, cv=cv, method='predict_proba')[:, 1]

auc = roc_auc_score(y, y_prob)
print(f"\nModel AUC (5-fold CV): {auc:.4f}")

# =============================================================================
# 3. COST FUNCTION
# =============================================================================
def compute_portfolio_value(threshold, y_true, y_prob, loan_amounts,
                             margin=LOAN_MARGIN, lgd=LOSS_GIVEN_DEFAULT):
    """
    Compute portfolio P&L for a given decision threshold.
    approve  = predicted non-default (y_pred = 0)
    reject   = predicted default    (y_pred = 1)
    """
    y_pred = (y_prob >= threshold).astype(int)

    # Outcome × Decision matrix
    TN_mask = (y_true == 0) & (y_pred == 0)   # good customer, approved  → earn margin
    FP_mask = (y_true == 0) & (y_pred == 1)   # good customer, rejected  → lost margin
    FN_mask = (y_true == 1) & (y_pred == 0)   # defaulter,   approved  → lose LGD
    TP_mask = (y_true == 1) & (y_pred == 1)   # defaulter,   rejected  → $0

    revenue  = (loan_amounts[TN_mask] * margin).sum()
    opp_cost = (loan_amounts[FP_mask] * margin).sum()
    loss     = (loan_amounts[FN_mask] * lgd).sum()

    profit  = revenue - opp_cost - loss
    n_approved = TN_mask.sum() + FN_mask.sum()
    approval_rate = n_approved / len(y_true)

    return {
        'threshold':     threshold,
        'profit':        profit,
        'revenue':       revenue,
        'opp_cost':      opp_cost,
        'loss':          loss,
        'TP': TP_mask.sum(), 'TN': TN_mask.sum(),
        'FP': FP_mask.sum(), 'FN': FN_mask.sum(),
        'approval_rate': approval_rate,
        'n_approved':    n_approved,
    }

# Sweep thresholds
thresholds = np.linspace(0.05, 0.95, 200)
results = [compute_portfolio_value(t, y.values, y_prob, loan_amounts) for t in thresholds]
df_sweep = pd.DataFrame(results)

# Find optimal threshold (max profit)
best_idx   = df_sweep['profit'].idxmax()
best_row   = df_sweep.loc[best_idx]
best_thresh = best_row['threshold']

print(f"\nOptimal threshold: {best_thresh:.3f}")
print(f"  Approval rate at optimal: {best_row['approval_rate']:.1%}")
print(f"  Portfolio profit at optimal: ${best_row['profit']:,.0f}")

# =============================================================================
# 4. THREE-WAY COMPARISON AT KEY THRESHOLDS
# =============================================================================
print("\n" + "=" * 70)
print("SCENARIO COMPARISON")
print("=" * 70)

# Rule-based baseline
# The rule system has 3 tiers: 'approved', 'flagged_for_review', 'denied'
# Current policy: approve everything except 'denied' (flagged still goes through)
# This models a human-in-the-loop that reviews flagged applications.
rule_reject = (df_model['rule_based_decision'] == 'denied').astype(int).values
rule_TN = ((y.values == 0) & (rule_reject == 0)).sum()
rule_FP = ((y.values == 0) & (rule_reject == 1)).sum()
rule_FN = ((y.values == 1) & (rule_reject == 0)).sum()
rule_TP = ((y.values == 1) & (rule_reject == 1)).sum()
rule_revenue  = (loan_amounts[(y.values == 0) & (rule_reject == 0)] * LOAN_MARGIN).sum()
rule_opp_cost = (loan_amounts[(y.values == 0) & (rule_reject == 1)] * LOAN_MARGIN).sum()
rule_loss     = (loan_amounts[(y.values == 1) & (rule_reject == 0)] * LOSS_GIVEN_DEFAULT).sum()
rule_profit   = rule_revenue - rule_opp_cost - rule_loss
rule_approval = (rule_TN + rule_FN) / len(y)

# Approve-all baseline
all_revenue  = (loan_amounts[y.values == 0] * LOAN_MARGIN).sum()
all_loss     = (loan_amounts[y.values == 1] * LOSS_GIVEN_DEFAULT).sum()
all_profit   = all_revenue - all_loss
all_approval = 1.0

rows = []
for label, thresh in [('Approve All (baseline)', None),
                       ('Rule-Based System',      None),
                       ('ML Model @ 0.30',        0.30),
                       ('ML Model @ 0.50',        0.50),
                       (f'ML Model @ {best_thresh:.2f} (optimal)', best_thresh)]:
    if thresh is None:
        if label.startswith('Approve'):
            r = {'Scenario': label,
                 'Approval Rate': f'{all_approval:.1%}',
                 'Revenue ($)':   f'${all_revenue:,.0f}',
                 'FN Loss ($)':   f'${all_loss:,.0f}',
                 'Net Profit ($)':f'${all_profit:,.0f}',
                 'FP Count': 0,
                 'FN Count': int((y.values==1).sum())}
        else:
            r = {'Scenario': label,
                 'Approval Rate': f'{rule_approval:.1%}',
                 'Revenue ($)':   f'${rule_revenue:,.0f}',
                 'FN Loss ($)':   f'${rule_loss:,.0f}',
                 'Net Profit ($)':f'${rule_profit:,.0f}',
                 'FP Count': rule_FP,
                 'FN Count': rule_FN}
    else:
        res = compute_portfolio_value(thresh, y.values, y_prob, loan_amounts)
        r = {'Scenario': label,
             'Approval Rate': f'{res["approval_rate"]:.1%}',
             'Revenue ($)':   f'${res["revenue"]:,.0f}',
             'FN Loss ($)':   f'${res["loss"]:,.0f}',
             'Net Profit ($)':f'${res["profit"]:,.0f}',
             'FP Count': res['FP'],
             'FN Count': res['FN']}
    rows.append(r)

df_compare = pd.DataFrame(rows)
print(df_compare.to_string(index=False))

# How much better is the optimal ML model vs rule-based?
best_profit_ml   = compute_portfolio_value(best_thresh, y.values, y_prob, loan_amounts)['profit']
uplift_vs_rules  = best_profit_ml - rule_profit
uplift_vs_all    = best_profit_ml - all_profit
print(f"\nML Model (optimal) vs Rule-Based: ${uplift_vs_rules:+,.0f}")
print(f"ML Model (optimal) vs Approve-All: ${uplift_vs_all:+,.0f}")

# =============================================================================
# 5. SENSITIVITY: LGD assumption varies (±15pp)
# =============================================================================
lgd_values  = [0.50, 0.65, 0.80]
lgd_labels  = ['LGD = 50%', 'LGD = 65% (base)', 'LGD = 80%']
lgd_colors  = ['#3498db', '#e74c3c', '#8e44ad']

sens_results = {}
for lgd_val in lgd_values:
    s_results = [compute_portfolio_value(t, y.values, y_prob, loan_amounts,
                                          margin=LOAN_MARGIN, lgd=lgd_val)
                 for t in thresholds]
    sens_results[lgd_val] = pd.DataFrame(s_results)

# =============================================================================
# 6. PLOTS
# =============================================================================

# ── Figure 1: Main cost curve (4-panel) ──────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Business Cost Analysis — Loan Default Model', fontsize=14, fontweight='bold', y=1.01)

# Panel 1: Net profit vs threshold
ax = axes[0, 0]
ax.plot(df_sweep['threshold'], df_sweep['profit'] / 1e3, color='#2ecc71', lw=2)
ax.axvline(best_thresh, color='#e74c3c', linestyle='--', lw=1.5, label=f'Optimal: {best_thresh:.2f}')
ax.axhline(rule_profit / 1e3, color='#f39c12', linestyle=':', lw=1.5, label='Rule-based')
ax.axhline(all_profit / 1e3, color='#95a5a6', linestyle=':', lw=1.5, label='Approve-all')
ax.set_xlabel('Decision Threshold')
ax.set_ylabel('Net Portfolio Profit ($K)')
ax.set_title('Net Profit vs Decision Threshold')
ax.legend(fontsize=8)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:.0f}K'))

# Panel 2: Cost decomposition
ax = axes[0, 1]
ax.stackplot(df_sweep['threshold'],
             df_sweep['revenue'] / 1e3,
             -df_sweep['opp_cost'] / 1e3,
             -df_sweep['loss'] / 1e3,
             labels=['Revenue (TN)', 'Opportunity Cost (FP)', 'Default Loss (FN)'],
             colors=['#2ecc71', '#f39c12', '#e74c3c'],
             alpha=0.7)
ax.axvline(best_thresh, color='black', linestyle='--', lw=1.2, label=f'Optimal: {best_thresh:.2f}')
ax.set_xlabel('Decision Threshold')
ax.set_ylabel('$K')
ax.set_title('Profit Component Breakdown')
ax.legend(fontsize=8, loc='upper right')
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:.0f}K'))

# Panel 3: Approval rate & error counts
ax = axes[1, 0]
ax2 = ax.twinx()
ax.plot(df_sweep['threshold'], df_sweep['approval_rate'] * 100, color='#3498db', lw=2, label='Approval rate')
ax2.plot(df_sweep['threshold'], df_sweep['FN'], color='#e74c3c', lw=1.5, linestyle='--', label='FN count')
ax2.plot(df_sweep['threshold'], df_sweep['FP'], color='#f39c12', lw=1.5, linestyle='--', label='FP count')
ax.axvline(best_thresh, color='black', linestyle=':', lw=1.2)
ax.set_xlabel('Decision Threshold')
ax.set_ylabel('Approval Rate (%)', color='#3498db')
ax2.set_ylabel('Error Count')
ax.set_title('Approval Rate and Errors vs Threshold')
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8)

# Panel 4: LGD sensitivity
ax = axes[1, 1]
for lgd_val, label, color in zip(lgd_values, lgd_labels, lgd_colors):
    df_s = sens_results[lgd_val]
    ax.plot(df_s['threshold'], df_s['profit'] / 1e3, color=color, lw=2, label=label)
    opt_idx = df_s['profit'].idxmax()
    ax.axvline(df_s.loc[opt_idx, 'threshold'], color=color, linestyle=':', lw=1)
ax.set_xlabel('Decision Threshold')
ax.set_ylabel('Net Portfolio Profit ($K)')
ax.set_title('Sensitivity to Loss-Given-Default Assumption')
ax.legend(fontsize=9)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:.0f}K'))

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/13_cost_analysis.png")
plt.close()
print("\nSaved: 13_cost_analysis.png")

# ── Figure 2: Scenario bar chart ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))

scenario_names = [r['Scenario'] for r in rows]
profits_k = [
    all_profit / 1e3,
    rule_profit / 1e3,
    compute_portfolio_value(0.30, y.values, y_prob, loan_amounts)['profit'] / 1e3,
    compute_portfolio_value(0.50, y.values, y_prob, loan_amounts)['profit'] / 1e3,
    best_profit_ml / 1e3,
]
colors = ['#95a5a6', '#f39c12', '#3498db', '#9b59b6', '#2ecc71']
bars = ax.barh(scenario_names, profits_k, color=colors, edgecolor='white', height=0.6)

for bar, val in zip(bars, profits_k):
    ax.text(val + 0.5, bar.get_y() + bar.get_height()/2,
            f'${val:.1f}K', va='center', fontsize=9, fontweight='bold')

ax.set_xlabel('Net Portfolio Profit ($K)')
ax.set_title(f'Profit Comparison Across Decision Strategies\n(5-fold CV predictions on {len(y):,} resolved applications)',
             fontsize=11)
ax.axvline(0, color='black', lw=0.8)
ax.set_xlim(left=min(profits_k) - 5)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/14_scenario_comparison.png")
plt.close()
print("Saved: 14_scenario_comparison.png")

# =============================================================================
# 7. SUMMARY
# =============================================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
opt_res = compute_portfolio_value(best_thresh, y.values, y_prob, loan_amounts)
print(f"""
Optimal operating threshold: {best_thresh:.2f}
  Approval rate:     {opt_res['approval_rate']:.1%}  ({opt_res['n_approved']} of {len(y)} applications)
  Approved repayers: {opt_res['TN']}  → earn ${opt_res['revenue']:,.0f}
  Rejected repayers: {opt_res['FP']}  → lose ${opt_res['opp_cost']:,.0f} (opportunity cost)
  Missed defaulters: {opt_res['FN']}  → lose ${opt_res['loss']:,.0f} (default losses)
  Caught defaulters: {opt_res['TP']}  → avoided loss

  NET PROFIT: ${opt_res['profit']:,.0f}

vs Rule-Based System:
  Rule-based profit: ${rule_profit:,.0f}  (approval rate: {rule_approval:.1%})
  ML model uplift:   ${uplift_vs_rules:+,.0f}  ({uplift_vs_rules/abs(rule_profit)*100:+.1f}%)

vs Approve-All:
  Approve-all profit: ${all_profit:,.0f}
  ML model uplift:    ${uplift_vs_all:+,.0f}

Key insight: The optimal threshold ({best_thresh:.2f}) is {'below' if best_thresh < 0.5 else 'above'} 0.5,
meaning the model should be {'more lenient' if best_thresh < 0.5 else 'stricter'} than a
naive classifier. This is because FN losses ({LOSS_GIVEN_DEFAULT*100:.0f}% of loan) are
{LOSS_GIVEN_DEFAULT/LOAN_MARGIN:.1f}x larger than the revenue from a single good loan ({LOAN_MARGIN*100:.0f}%).
""")
