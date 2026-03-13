"""
False Negative Analysis
=======================
Who are the defaults that slipped through?
Compare: False Negatives vs True Positives vs All Defaulters
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import shap
import warnings
warnings.filterwarnings('ignore')

sns.set_theme(style="whitegrid", palette="muted")
OUTPUT_DIR = "outputs"

# ── Rebuild model (same as loan_model.py) ────────────────────────────────────
df = pd.read_csv("loan_applications.csv")
df_model = df[df['actual_outcome'] != 'ongoing'].copy()
df_model['target'] = (df_model['actual_outcome'] == 'defaulted').astype(int)

df_model['documented_monthly_income'] = df_model['documented_monthly_income'].fillna(0)
mask = df_model['documented_monthly_income'] > 0

df_model['has_documentation']  = mask.astype(int)
df_model['income_verified']    = 0
df_model.loc[mask, 'income_verified'] = (
    (np.abs(df_model.loc[mask,'stated_monthly_income'] -
            df_model.loc[mask,'documented_monthly_income']) /
     df_model.loc[mask,'stated_monthly_income']) <= 0.1).astype(int)
df_model['possible_misrep']    = 0
df_model.loc[mask, 'possible_misrep'] = (
    df_model.loc[mask,'stated_monthly_income'] >
    3 * df_model.loc[mask,'documented_monthly_income']).astype(int)

df_model['loan_to_income']   = df_model['loan_amount'] / df_model['stated_monthly_income']
df_model['withdrawal_ratio'] = df_model['monthly_withdrawals'] / df_model['monthly_deposits'].clip(lower=1)
df_model['income_doc_ratio'] = np.where(mask,
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
    'income_verified', 'is_unemployed', 'is_self_employed', 'stated_monthly_income',
    'documented_monthly_income', 'loan_amount', 'bank_ending_balance',
    'bank_has_overdrafts', 'bank_has_consistent_deposits', 'monthly_withdrawals',
    'monthly_deposits', 'num_documents_submitted', 'net_cash_flow',
]

X = df_model[all_features].copy()
y = df_model['target'].copy()
for col in X.columns:
    if X[col].dtype == 'bool':
        X[col] = X[col].astype(int)

cv  = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
lr  = Pipeline([('scaler', StandardScaler()),
                ('clf', LogisticRegression(class_weight='balanced',
                                           C=1.0, max_iter=1000, random_state=42))])
y_pred_proba = cross_val_predict(lr, X, y, cv=cv, method='predict_proba')[:, 1]
y_pred       = (y_pred_proba >= 0.5).astype(int)

df_model['pred_proba'] = y_pred_proba
df_model['pred_label'] = y_pred

# ── Segment groups ────────────────────────────────────────────────────────────
fn_mask = (df_model['pred_label'] == 0) & (df_model['target'] == 1)  # missed defaults
tp_mask = (df_model['pred_label'] == 1) & (df_model['target'] == 1)  # caught defaults
tn_mask = (df_model['pred_label'] == 0) & (df_model['target'] == 0)  # correctly approved
fp_mask = (df_model['pred_label'] == 1) & (df_model['target'] == 0)  # wrongly denied

fn_df = df_model[fn_mask].copy()  # False Negatives  — slipped through
tp_df = df_model[tp_mask].copy()  # True Positives   — caught defaults
all_default_df = df_model[df_model['target'] == 1].copy()

print("=" * 75)
print("FALSE NEGATIVE ANALYSIS — Defaults That Slipped Through")
print("=" * 75)
print(f"\n  Total resolved applicants: {len(df_model)}")
print(f"  All defaulters:            {df_model['target'].sum()} ({df_model['target'].mean():.1%})")
print(f"  ├─ Caught  (TP):           {tp_mask.sum()} ({tp_mask.sum()/df_model['target'].sum():.1%} of all defaults)")
print(f"  └─ Slipped through (FN):   {fn_mask.sum()} ({fn_mask.sum()/df_model['target'].sum():.1%} of all defaults)")
print(f"\n  Avg predicted probability:")
print(f"  ├─ False Negatives:  {fn_df['pred_proba'].mean():.3f}  (model thought they were safe)")
print(f"  ├─ True Positives:   {tp_df['pred_proba'].mean():.3f}  (model correctly flagged)")
print(f"  └─ All defaulters:   {all_default_df['pred_proba'].mean():.3f}")

# ── Average stats comparison ─────────────────────────────────────────────────
display_features = [
    # Most interpretable features first
    'loan_to_income',
    'withdrawal_ratio',
    'income_doc_ratio',
    'balance_to_loan',
    'net_flow_to_loan',
    'has_documentation',
    'income_verified',
    'possible_misrep',
    'is_unemployed',
    'is_self_employed',
    'bank_has_overdrafts',
    'bank_has_consistent_deposits',
    'stated_monthly_income',
    'loan_amount',
    'bank_ending_balance',
    'num_documents_submitted',
    'net_cash_flow',
]

stats = pd.DataFrame({
    'FN (slipped)':   fn_df[display_features].mean(),
    'TP (caught)':    tp_df[display_features].mean(),
    'All defaults':   all_default_df[display_features].mean(),
    'All repaid':     df_model[df_model['target']==0][display_features].mean(),
}).round(3)

# Add a diff column: how far are FNs from caught defaults?
stats['FN vs TP diff'] = (stats['FN (slipped)'] - stats['TP (caught)']).round(3)

print(f"\n{'─'*85}")
print(f"  {'Feature':<32} {'FN (slipped)':>14} {'TP (caught)':>12} {'All defaults':>13} {'All repaid':>11} {'FN-TP diff':>11}")
print(f"{'─'*85}")
for feat in display_features:
    row = stats.loc[feat]
    # Flag features where FNs look more like repaid than caught defaults
    fn_val     = row['FN (slipped)']
    tp_val     = row['TP (caught)']
    repaid_val = row['All repaid']
    diff       = row['FN vs TP diff']
    # FN looks "safer" than TP on this feature?
    closer_to_repaid = abs(fn_val - repaid_val) < abs(tp_val - repaid_val)
    flag = " ←" if closer_to_repaid and abs(diff) > 0.05 else ""
    print(f"  {feat:<32} {fn_val:>14.3f} {tp_val:>12.3f} {row['All defaults']:>13.3f} {repaid_val:>11.3f} {diff:>+11.3f}{flag}")

print(f"\n  ← = FN applicants look more like repaid than caught defaults on this feature")

# ── SHAP analysis for FNs ─────────────────────────────────────────────────────
print(f"\n{'─'*75}")
print("  SHAP ANALYSIS — Why did the model think FNs were safe?")
print(f"{'─'*75}")

lr.fit(X, y)
scaler      = lr.named_steps['scaler']
X_scaled    = pd.DataFrame(scaler.transform(X), columns=X.columns, index=X.index)
explainer   = shap.LinearExplainer(lr.named_steps['clf'], X_scaled)
shap_values = explainer.shap_values(X_scaled)

fn_idx = np.where(fn_mask.values)[0]
tp_idx = np.where(tp_mask.values)[0]

shap_fn = pd.Series(shap_values[fn_idx].mean(axis=0),  index=all_features)
shap_tp = pd.Series(shap_values[tp_idx].mean(axis=0),  index=all_features)

# For FNs, which features are NEGATIVE (pushing model toward "safe")?
shap_fn_sorted = shap_fn.sort_values()  # most negative first = strongest "safe" signals

print(f"\n  Average SHAP for False Negatives (sorted — most negative = why model said 'safe'):\n")
print(f"  {'Feature':<32} {'FN avg SHAP':>12} {'TP avg SHAP':>12} {'Difference':>12}")
print(f"  {'─'*70}")
for feat in shap_fn_sorted.index:
    fn_s  = shap_fn[feat]
    tp_s  = shap_tp[feat]
    diff  = fn_s - tp_s
    bar   = "░" * min(int(abs(fn_s) * 20), 15) if fn_s < 0 else "█" * min(int(abs(fn_s) * 20), 15)
    color_flag = " ← key factor" if fn_s < -0.1 and tp_s > 0 else ""
    print(f"  {feat:<32} {fn_s:>+12.4f} {tp_s:>+12.4f} {diff:>+12.4f}  {bar}{color_flag}")

print(f"\n  Top reasons the model said FNs were SAFE (most negative SHAP):")
top_safe_signals = shap_fn_sorted.head(5)
for i, (feat, val) in enumerate(top_safe_signals.items(), 1):
    fn_avg = X.iloc[fn_idx][feat].mean()
    tp_avg = X.iloc[tp_idx][feat].mean()
    print(f"  {i}. {feat:<30} FN avg={fn_avg:.3f}  TP avg={tp_avg:.3f}  SHAP={val:+.4f}")

print(f"\n  In other words: FNs are harder to catch because they look")
print(f"  genuinely safer on these features — the model isn't making a mistake")
print(f"  so much as these applicants are harder to distinguish from repayers.")

# ── Visualisation ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("False Negatives: Why Did Defaults Slip Through?", fontsize=13, fontweight='bold')

plot_features = [
    ('loan_to_income',   'Loan-to-Income Ratio'),
    ('withdrawal_ratio', 'Withdrawal Ratio'),
    ('income_doc_ratio', 'Income Doc Ratio'),
    ('balance_to_loan',  'Balance-to-Loan Ratio'),
    ('has_documentation','Has Documentation'),
    ('bank_has_overdrafts', 'Has Overdrafts'),
]

for ax, (feat, label) in zip(axes.flat, plot_features):
    fn_vals     = X.iloc[fn_idx][feat]
    tp_vals     = X.iloc[tp_idx][feat]
    repaid_vals = X.iloc[np.where((df_model['target']==0).values)[0]][feat]

    if feat in ('has_documentation', 'bank_has_overdrafts'):
        cats  = ['FN\n(slipped)', 'TP\n(caught)', 'All repaid']
        means = [fn_vals.mean(), tp_vals.mean(), repaid_vals.mean()]
        colors = ['#e67e22', '#e74c3c', '#2ecc71']
        bars = ax.bar(cats, means, color=colors, alpha=0.8, edgecolor='white')
        for bar, val in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.2f}', ha='center', fontsize=9, fontweight='bold')
        ax.set_ylim(0, 1.1)
    else:
        ax.hist(repaid_vals, bins=30, alpha=0.4, color='#2ecc71', density=True, label='All repaid')
        ax.hist(tp_vals,     bins=30, alpha=0.6, color='#e74c3c', density=True, label='TP (caught)')
        ax.hist(fn_vals,     bins=30, alpha=0.7, color='#e67e22', density=True, label='FN (slipped)', linewidth=1.5)
        ax.legend(fontsize=8)
        ax.axvline(fn_vals.mean(),     color='#e67e22', linestyle='--', linewidth=2)
        ax.axvline(tp_vals.mean(),     color='#e74c3c', linestyle='--', linewidth=2)

    ax.set_title(label, fontweight='bold')

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/11_false_negative_analysis.png", bbox_inches='tight')
plt.close()
print(f"\nSaved: 11_false_negative_analysis.png")

# ── SHAP comparison plot ──────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 8))
feat_order = shap_fn.abs().sort_values(ascending=True).index
y_pos = np.arange(len(feat_order))
h = 0.35
ax.barh(y_pos + h/2, shap_fn[feat_order],  h, color='#e67e22', alpha=0.8, label='FN (slipped through)')
ax.barh(y_pos - h/2, shap_tp[feat_order],  h, color='#e74c3c', alpha=0.8, label='TP (correctly caught)')
ax.axvline(x=0, color='black', linewidth=0.8)
ax.set_yticks(y_pos)
ax.set_yticklabels(feat_order, fontsize=9)
ax.set_xlabel('Mean SHAP Value (positive = pushes toward default)')
ax.set_title('Why Did FNs Slip Through?\nMean SHAP Values: Missed Defaults vs Caught Defaults')
ax.legend()
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/12_fn_vs_tp_shap.png", bbox_inches='tight')
plt.close()
print(f"Saved: 12_fn_vs_tp_shap.png")

# ── One-line summary ──────────────────────────────────────────────────────────
print(f"\n{'='*75}")
print("SUMMARY")
print(f"{'='*75}")

# Find the top 3 features where FNs look most like repaid applicants
fn_repaid_similarity = {}
for feat in display_features:
    fn_val     = fn_df[feat].mean()
    tp_val     = tp_df[feat].mean()
    repaid_val = df_model[df_model['target']==0][feat].mean()
    # How much closer is FN to repaid than TP is to repaid?
    fn_gap = abs(fn_val - repaid_val)
    tp_gap = abs(tp_val - repaid_val)
    if tp_gap > 0:
        fn_repaid_similarity[feat] = tp_gap - fn_gap  # positive = FN closer to repaid

top_camouflage = sorted(fn_repaid_similarity.items(), key=lambda x: x[1], reverse=True)[:3]

print(f"""
  The {fn_mask.sum()} defaults that slipped through are genuinely harder to detect.
  They don't just look like average defaulters — on key features they more
  closely resemble repaid applicants:

  Top 3 features where FNs "look safe":""")
for feat, gap in top_camouflage:
    fn_val     = fn_df[feat].mean()
    tp_val     = tp_df[feat].mean()
    repaid_val = df_model[df_model['target']==0][feat].mean()
    print(f"    {feat:<30}  FN={fn_val:.3f}  TP={tp_val:.3f}  Repaid={repaid_val:.3f}")

print(f"""
  This means the model's false negatives are not random errors — they are
  structurally ambiguous cases. Without better data (e.g. credit history,
  spending category breakdowns), they would be hard for ANY model to catch.
""")
