"""
Profit Maximisation Chart
=========================
Single-purpose plot: shows portfolio P&L vs decision threshold,
highlights the optimal operating point, and benchmarks it
against the rule-based system and approve-all baselines.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

OUTPUT_DIR = "outputs"
import os
os.makedirs(OUTPUT_DIR, exist_ok=True)

LOAN_MARGIN        = 0.16
LOSS_GIVEN_DEFAULT = 0.65

# ── Reproduce feature engineering ────────────────────────────────────────────
df = pd.read_csv("loan_applications.csv")
df_model = df[df['actual_outcome'] != 'ongoing'].copy()
df_model['target'] = (df_model['actual_outcome'] == 'defaulted').astype(int)
df_model['documented_monthly_income'] = df_model['documented_monthly_income'].fillna(0)
mask = df_model['documented_monthly_income'] > 0

df_model['has_documentation'] = mask.astype(int)
df_model['income_verified'] = 0
df_model.loc[mask, 'income_verified'] = (
    (np.abs(df_model.loc[mask, 'stated_monthly_income'] -
            df_model.loc[mask, 'documented_monthly_income']) /
     df_model.loc[mask, 'stated_monthly_income']) <= 0.1).astype(int)
df_model['possible_misrep'] = 0
df_model.loc[mask, 'possible_misrep'] = (
    df_model.loc[mask, 'stated_monthly_income'] >
    3 * df_model.loc[mask, 'documented_monthly_income']).astype(int)
df_model['loan_to_income']   = df_model['loan_amount'] / df_model['stated_monthly_income']
df_model['withdrawal_ratio'] = df_model['monthly_withdrawals'] / df_model['monthly_deposits'].clip(lower=1)
df_model['income_doc_ratio'] = np.where(mask, df_model['documented_monthly_income'] / df_model['stated_monthly_income'], 0)
df_model['balance_to_loan']  = df_model['bank_ending_balance'] / df_model['loan_amount'].clip(lower=1)
df_model['deposits_to_loan'] = df_model['monthly_deposits'] / df_model['loan_amount'].clip(lower=1)
df_model['net_cash_flow']    = df_model['monthly_deposits'] - df_model['monthly_withdrawals']
df_model['net_flow_to_loan'] = df_model['net_cash_flow'] / df_model['loan_amount'].clip(lower=1)
df_model['is_unemployed']    = (df_model['employment_status'] == 'unemployed').astype(int)
df_model['is_self_employed'] = (df_model['employment_status'] == 'self_employed').astype(int)

all_features = [
    'loan_to_income','withdrawal_ratio','income_doc_ratio','balance_to_loan',
    'deposits_to_loan','net_flow_to_loan','has_documentation','possible_misrep',
    'income_verified','is_unemployed','is_self_employed','stated_monthly_income',
    'documented_monthly_income','loan_amount','bank_ending_balance',
    'bank_has_overdrafts','bank_has_consistent_deposits','monthly_withdrawals',
    'monthly_deposits','num_documents_submitted','net_cash_flow',
]

X = df_model[all_features].copy()
y = df_model['target'].copy()
loan_amounts = df_model['loan_amount'].values
for col in X.columns:
    if X[col].dtype == 'bool':
        X[col] = X[col].astype(int)

# ── OOF probabilities ─────────────────────────────────────────────────────────
pipe = Pipeline([('scaler', StandardScaler()),
                 ('lr', LogisticRegression(C=1.0, class_weight='balanced',
                                           max_iter=1000, random_state=42))])
cv     = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
y_prob = cross_val_predict(pipe, X, y, cv=cv, method='predict_proba')[:, 1]

# ── Cost function ─────────────────────────────────────────────────────────────
def portfolio_profit(threshold, y_true, y_prob, amounts):
    pred   = (y_prob >= threshold).astype(int)
    TN     = (y_true == 0) & (pred == 0)
    FP     = (y_true == 0) & (pred == 1)
    FN     = (y_true == 1) & (pred == 0)
    revenue  = (amounts[TN] * LOAN_MARGIN).sum()
    opp_cost = (amounts[FP] * LOAN_MARGIN).sum()
    loss     = (amounts[FN] * LOSS_GIVEN_DEFAULT).sum()
    return revenue - opp_cost - loss

thresholds = np.linspace(0.05, 0.95, 300)
profits    = np.array([portfolio_profit(t, y.values, y_prob, loan_amounts) for t in thresholds])

best_idx    = profits.argmax()
best_thresh = thresholds[best_idx]
best_profit = profits[best_idx]

# ── Baselines ─────────────────────────────────────────────────────────────────
rule_reject  = (df_model['rule_based_decision'] == 'denied').astype(int).values
rule_TN      = (y.values == 0) & (rule_reject == 0)
rule_FN      = (y.values == 1) & (rule_reject == 0)
rule_FP      = (y.values == 0) & (rule_reject == 1)
rule_profit  = ((loan_amounts[rule_TN] * LOAN_MARGIN).sum()
                - (loan_amounts[rule_FP] * LOAN_MARGIN).sum()
                - (loan_amounts[rule_FN] * LOSS_GIVEN_DEFAULT).sum())
all_profit   = ((loan_amounts[y.values == 0] * LOAN_MARGIN).sum()
                - (loan_amounts[y.values == 1] * LOSS_GIVEN_DEFAULT).sum())

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5),
                          gridspec_kw={'width_ratios': [2, 1]})
fig.patch.set_facecolor('#fafafa')

# ── LEFT: Profit curve ────────────────────────────────────────────────────────
ax = axes[0]
ax.set_facecolor('#fafafa')

# Shaded region: better than rule-based
better_mask = profits > rule_profit
ax.fill_between(thresholds, profits / 1e3, rule_profit / 1e3,
                where=better_mask, alpha=0.12, color='#2ecc71',
                label='Better than rule-based')

ax.plot(thresholds, profits / 1e3, color='#2c3e50', lw=2.5, zorder=3)

# Horizontal baselines
ax.axhline(rule_profit / 1e3, color='#e67e22', lw=1.5, linestyle='--', zorder=2)
ax.axhline(all_profit  / 1e3, color='#95a5a6', lw=1.5, linestyle=':',  zorder=2)

# Optimal point
ax.scatter([best_thresh], [best_profit / 1e3],
           color='#e74c3c', s=120, zorder=5, linewidths=1.5, edgecolors='white')
ax.annotate(
    f'Optimal threshold = {best_thresh:.2f}\n${best_profit/1e3:.1f}K',
    xy=(best_thresh, best_profit / 1e3),
    xytext=(best_thresh + 0.13, best_profit / 1e3 + 12),
    fontsize=9, color='#e74c3c', fontweight='bold',
    arrowprops=dict(arrowstyle='->', color='#e74c3c', lw=1.2),
)

# Default threshold marker
default_profit = portfolio_profit(0.5, y.values, y_prob, loan_amounts)
ax.scatter([0.5], [default_profit / 1e3],
           color='#9b59b6', s=80, zorder=4, linewidths=1.2, edgecolors='white')
ax.annotate(
    f'Threshold = 0.50\n${default_profit/1e3:.1f}K',
    xy=(0.5, default_profit / 1e3),
    xytext=(0.5 - 0.18, default_profit / 1e3 - 18),
    fontsize=9, color='#9b59b6',
    arrowprops=dict(arrowstyle='->', color='#9b59b6', lw=1.0),
)

# Baseline label annotations
ax.text(0.92, rule_profit / 1e3 + 2, 'Rule-based', ha='right', va='bottom',
        fontsize=8.5, color='#e67e22', style='italic')
ax.text(0.92, all_profit / 1e3 + 2, 'Approve all', ha='right', va='bottom',
        fontsize=8.5, color='#95a5a6', style='italic')

ax.set_xlabel('Decision Threshold  (reject if predicted default probability ≥ threshold)',
              fontsize=10)
ax.set_ylabel('Net Portfolio Profit / Loss ($K)', fontsize=10)
ax.set_title('Portfolio P&L vs Decision Threshold', fontsize=12, fontweight='bold', pad=10)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:.0f}K'))
ax.set_xlim(0.05, 0.95)

green_patch = mpatches.Patch(color='#2ecc71', alpha=0.3, label='ML beats rule-based')
ax.legend(handles=[green_patch], fontsize=8, loc='lower left')
ax.grid(axis='y', alpha=0.4)
ax.spines[['top','right']].set_visible(False)

# ── RIGHT: Scenario bar chart ─────────────────────────────────────────────────
ax2 = axes[1]
ax2.set_facecolor('#fafafa')

scenarios = ['Approve All', 'Rule-Based\nSystem', 'ML Model\n@ 0.50', f'ML Model\n@ {best_thresh:.2f}\n(optimal)']
values    = [all_profit / 1e3, rule_profit / 1e3, default_profit / 1e3, best_profit / 1e3]
colors    = ['#bdc3c7', '#e67e22', '#9b59b6', '#2ecc71']

bars = ax2.barh(scenarios, values, color=colors, height=0.55,
                edgecolor='white', linewidth=0.8)

for bar, val in zip(bars, values):
    x_pos  = val - 1.5 if val < 0 else val + 0.5
    ha     = 'right' if val < 0 else 'left'
    ax2.text(x_pos, bar.get_y() + bar.get_height() / 2,
             f'${val:.1f}K', va='center', ha=ha, fontsize=9, fontweight='bold',
             color='#2c3e50')

# Uplift annotation
uplift = best_profit - rule_profit
ax2.annotate('', xy=(best_profit / 1e3, 3), xytext=(rule_profit / 1e3, 3),
             arrowprops=dict(arrowstyle='<->', color='#27ae60', lw=1.5))
ax2.text((best_profit + rule_profit) / 2 / 1e3, 3.28,
         f'+${uplift/1e3:.1f}K saved\nvs rule-based',
         ha='center', va='bottom', fontsize=8, color='#27ae60', fontweight='bold')

ax2.axvline(0, color='#7f8c8d', lw=0.8)
ax2.set_xlabel('Net Portfolio Profit / Loss ($K)', fontsize=10)
ax2.set_title('Strategy Comparison', fontsize=12, fontweight='bold', pad=10)
ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:.0f}K'))
ax2.spines[['top','right']].set_visible(False)
ax2.grid(axis='x', alpha=0.4)

plt.suptitle(
    f'Optimal threshold ({best_thresh:.2f}) saves ${uplift/1e3:.1f}K vs rule-based system  '
    f'| LGD = {LOSS_GIVEN_DEFAULT*100:.0f}%  |  Margin = {LOAN_MARGIN*100:.0f}%',
    fontsize=9, color='#7f8c8d', y=0.01
)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/15_profit_maximisation.png", dpi=150, bbox_inches='tight',
            facecolor='#fafafa')
plt.close()
print("Saved: 15_profit_maximisation.png")
