"""
Fairness Analysis Chart
=======================
Visualises approval rate disparities and default rates across employment groups
for both the rule-based system and the ML model.
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
import warnings
warnings.filterwarnings('ignore')

OUTPUT_DIR = "outputs"
import os
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Feature engineering (mirrors loan_model.py) ───────────────────────────────
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
for col in X.columns:
    if X[col].dtype == 'bool':
        X[col] = X[col].astype(int)

# ── OOF probabilities ─────────────────────────────────────────────────────────
pipe = Pipeline([('scaler', StandardScaler()),
                 ('lr', LogisticRegression(C=1.0, class_weight='balanced',
                                           max_iter=1000, random_state=42))])
cv     = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
y_prob = cross_val_predict(pipe, X, y, cv=cv, method='predict_proba')[:, 1]

df_model['ml_prob']     = y_prob
df_model['ml_approved'] = (y_prob < 0.5).astype(int)
df_model['rule_approved'] = (df_model['rule_based_decision'] == 'approved').astype(int)

# ── Per-group stats ───────────────────────────────────────────────────────────
groups       = ['employed', 'self_employed', 'unemployed']
group_labels = ['Employed', 'Self-employed', 'Unemployed']
COLORS = {'rule': '#e67e22', 'ml': '#2ecc71', 'default_rate': '#e74c3c'}

stats = {}
for g in groups:
    sub = df_model[df_model['employment_status'] == g]
    stats[g] = {
        'n':            len(sub),
        'default_rate': sub['target'].mean(),
        'rule_approval': sub['rule_approved'].mean(),
        'ml_approval':   sub['ml_approved'].mean(),
    }

# ── Plot ──────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 9))
fig.patch.set_facecolor('#fafafa')

gs = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.32,
                      left=0.08, right=0.97, top=0.88, bottom=0.08)

ax_main  = fig.add_subplot(gs[0, :])   # top: grouped bar — full width
ax_gap   = fig.add_subplot(gs[1, 0])   # bottom-left: approval gap
ax_dist  = fig.add_subplot(gs[1, 1])   # bottom-right: ML prob distributions

fig.suptitle('Fairness Analysis — Approval Rates Across Employment Groups',
             fontsize=13, fontweight='bold', y=0.97)

# ── TOP: grouped bar chart ────────────────────────────────────────────────────
x  = np.arange(len(groups))
w  = 0.22

rule_vals    = [stats[g]['rule_approval']  for g in groups]
ml_vals      = [stats[g]['ml_approval']    for g in groups]
default_vals = [stats[g]['default_rate']   for g in groups]

b1 = ax_main.bar(x - w,   rule_vals,    w, label='Rule-Based Approval Rate', color=COLORS['rule'],         alpha=0.85)
b2 = ax_main.bar(x,       ml_vals,      w, label='ML Model Approval Rate',   color=COLORS['ml'],           alpha=0.85)
b3 = ax_main.bar(x + w,   default_vals, w, label='Actual Default Rate',      color=COLORS['default_rate'], alpha=0.60,
                 hatch='///', edgecolor='white')

# Value labels
for bars in [b1, b2, b3]:
    for bar in bars:
        h = bar.get_height()
        ax_main.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                     f'{h:.0%}', ha='center', va='bottom', fontsize=8.5, fontweight='bold')

# Annotation: self-employed correction
ax_main.annotate('',
    xy=(1, ml_vals[1] + 0.02), xytext=(1 - w, rule_vals[1] + 0.02),
    arrowprops=dict(arrowstyle='->', color='#27ae60', lw=1.8))
ax_main.text(1 - w/2, max(rule_vals[1], ml_vals[1]) + 0.055,
             f'ML approves {(ml_vals[1]-rule_vals[1]):.0%} more\nself-employed applicants',
             ha='center', fontsize=8.5, color='#27ae60', fontweight='bold')

ax_main.set_xticks(x)
ax_main.set_xticklabels(group_labels, fontsize=11)
ax_main.set_ylabel('Rate', fontsize=10)
ax_main.set_ylim(0, 0.92)
ax_main.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.0%}'))
ax_main.legend(fontsize=9, loc='upper right')
ax_main.set_title('Approval Rate vs Actual Default Rate by Employment Group', fontsize=11, pad=8)
ax_main.spines[['top','right']].set_visible(False)
ax_main.set_facecolor('#fafafa')
ax_main.grid(axis='y', alpha=0.35)

# ── BOTTOM-LEFT: approval gap (disparity vs employed) ────────────────────────
employed_rule = stats['employed']['rule_approval']
employed_ml   = stats['employed']['ml_approval']

rule_gaps = [(stats[g]['rule_approval'] - employed_rule) for g in groups]
ml_gaps   = [(stats[g]['ml_approval']   - employed_ml)   for g in groups]

x2 = np.arange(len(groups))
w2 = 0.3
ax_gap.bar(x2 - w2/2, rule_gaps, w2, label='Rule-Based', color=COLORS['rule'], alpha=0.85)
ax_gap.bar(x2 + w2/2, ml_gaps,   w2, label='ML Model',   color=COLORS['ml'],   alpha=0.85)
ax_gap.axhline(0, color='#2c3e50', lw=1)

for i, (rg, mg) in enumerate(zip(rule_gaps, ml_gaps)):
    ax_gap.text(i - w2/2, rg + (0.005 if rg >= 0 else -0.018),
                f'{rg:+.0%}', ha='center', fontsize=8, fontweight='bold', color=COLORS['rule'])
    ax_gap.text(i + w2/2, mg + (0.005 if mg >= 0 else -0.018),
                f'{mg:+.0%}', ha='center', fontsize=8, fontweight='bold', color='#27ae60')

ax_gap.set_xticks(x2)
ax_gap.set_xticklabels(group_labels, fontsize=9.5)
ax_gap.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:+.0%}'))
ax_gap.set_title('Approval Gap vs Employed\n(0 = same treatment as employed)', fontsize=10, pad=6)
ax_gap.legend(fontsize=8)
ax_gap.spines[['top','right']].set_visible(False)
ax_gap.set_facecolor('#fafafa')
ax_gap.grid(axis='y', alpha=0.35)

# ── BOTTOM-RIGHT: ML probability distributions by group ──────────────────────
palette = {'employed': '#3498db', 'self_employed': '#2ecc71', 'unemployed': '#e74c3c'}
for g, label, color in zip(groups, group_labels, palette.values()):
    sub = df_model[df_model['employment_status'] == g]
    ax_dist.hist(sub['ml_prob'], bins=28, alpha=0.50, density=True,
                 color=color, label=f'{label} (n={len(sub)})')

ax_dist.axvline(0.5, color='#2c3e50', linestyle='--', lw=1.5, label='Decision threshold')
ax_dist.set_xlabel('Predicted Default Probability', fontsize=10)
ax_dist.set_ylabel('Density', fontsize=10)
ax_dist.set_title('ML Predicted Default Probability\nby Employment Group', fontsize=10, pad=6)
ax_dist.legend(fontsize=8)
ax_dist.spines[['top','right']].set_visible(False)
ax_dist.set_facecolor('#fafafa')
ax_dist.grid(axis='y', alpha=0.35)

plt.savefig(f"{OUTPUT_DIR}/16_fairness_chart.png", dpi=150,
            bbox_inches='tight', facecolor='#fafafa')
plt.close()
print("Saved: 16_fairness_chart.png")
