"""
Evaluation Against Baseline Charts
==================================
Creates polished visuals for the "Evaluation Against the Baseline" section:
1. Threshold-independent and threshold-dependent metric comparison
2. Side-by-side confusion matrices with deployment impact callouts
"""

import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def build_clean_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    df_model = df[df["actual_outcome"] != "ongoing"].copy()
    df_model["target"] = (df_model["actual_outcome"] == "defaulted").astype(int)

    df_model["documented_monthly_income"] = df_model["documented_monthly_income"].fillna(0)
    mask = df_model["documented_monthly_income"] > 0

    df_model["has_documentation"] = mask.astype(int)
    df_model["income_verified"] = 0
    df_model.loc[mask, "income_verified"] = (
        (
            np.abs(
                df_model.loc[mask, "stated_monthly_income"]
                - df_model.loc[mask, "documented_monthly_income"]
            )
            / df_model.loc[mask, "stated_monthly_income"]
        )
        <= 0.1
    ).astype(int)
    df_model["possible_misrep"] = 0
    df_model.loc[mask, "possible_misrep"] = (
        df_model.loc[mask, "stated_monthly_income"]
        > 3 * df_model.loc[mask, "documented_monthly_income"]
    ).astype(int)

    df_model["loan_to_income"] = df_model["loan_amount"] / df_model["stated_monthly_income"]
    df_model["withdrawal_ratio"] = (
        df_model["monthly_withdrawals"] / df_model["monthly_deposits"].clip(lower=1)
    )
    df_model["income_doc_ratio"] = np.where(
        mask,
        df_model["documented_monthly_income"] / df_model["stated_monthly_income"],
        0,
    )
    df_model["balance_to_loan"] = (
        df_model["bank_ending_balance"] / df_model["loan_amount"].clip(lower=1)
    )
    df_model["deposits_to_loan"] = (
        df_model["monthly_deposits"] / df_model["loan_amount"].clip(lower=1)
    )
    df_model["net_cash_flow"] = (
        df_model["monthly_deposits"] - df_model["monthly_withdrawals"]
    )
    df_model["net_flow_to_loan"] = (
        df_model["net_cash_flow"] / df_model["loan_amount"].clip(lower=1)
    )
    df_model["is_unemployed"] = (
        df_model["employment_status"] == "unemployed"
    ).astype(int)
    df_model["is_self_employed"] = (
        df_model["employment_status"] == "self_employed"
    ).astype(int)

    features = [
        "loan_to_income",
        "withdrawal_ratio",
        "income_doc_ratio",
        "balance_to_loan",
        "deposits_to_loan",
        "net_flow_to_loan",
        "has_documentation",
        "possible_misrep",
        "income_verified",
        "is_unemployed",
        "is_self_employed",
        "stated_monthly_income",
        "documented_monthly_income",
        "loan_amount",
        "bank_ending_balance",
        "bank_has_overdrafts",
        "bank_has_consistent_deposits",
        "monthly_withdrawals",
        "monthly_deposits",
        "num_documents_submitted",
        "net_cash_flow",
    ]

    X = df_model[features].copy()
    for col in X.columns:
        if X[col].dtype == "bool":
            X[col] = X[col].astype(int)

    return X, df_model["target"].copy(), df_model


def rates_from_cm(cm: np.ndarray) -> dict[str, float]:
    tn, fp, fn, tp = cm.ravel()
    return {
        "fpr": fp / (fp + tn),
        "fnr": fn / (fn + tp),
        "approval_rate": (tn + fn) / cm.sum(),
        "intervention_rate": (fp + tp) / cm.sum(),
    }


df = pd.read_csv("loan_applications.csv")
X, y, df_model = build_clean_features(df)

pipe = Pipeline(
    [
        ("scaler", StandardScaler()),
        (
            "lr",
            LogisticRegression(
                C=1.0,
                class_weight="balanced",
                max_iter=1000,
                random_state=42,
            ),
        ),
    ]
)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
y_prob = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba")[:, 1]
y_pred = (y_prob >= 0.5).astype(int)

rule_pred = (df_model["rule_based_decision"] == "denied").astype(int).values
rule_score = 100 - df_model["rule_based_score"].values

metrics = pd.DataFrame(
    {
        "Metric": ["Precision", "Recall", "F1", "ROC-AUC"],
        "ML Model": [
            precision_score(y, y_pred),
            recall_score(y, y_pred),
            f1_score(y, y_pred),
            roc_auc_score(y, y_prob),
        ],
        "Rule-Based": [
            precision_score(y, rule_pred),
            recall_score(y, rule_pred),
            f1_score(y, rule_pred),
            roc_auc_score(y, rule_score),
        ],
    }
)

cm_ml = confusion_matrix(y, y_pred)
cm_rule = confusion_matrix(y, rule_pred)
rates_ml = rates_from_cm(cm_ml)
rates_rule = rates_from_cm(cm_rule)


# ---------------------------------------------------------------------------
# Figure 1: metric comparison dashboard
# ---------------------------------------------------------------------------
fig = plt.figure(figsize=(14, 8))
fig.patch.set_facecolor("#fafafa")
gs = fig.add_gridspec(2, 2, hspace=0.34, wspace=0.28, left=0.07, right=0.97, top=0.9, bottom=0.08)

ax_metrics = fig.add_subplot(gs[:, 0])
ax_errors = fig.add_subplot(gs[0, 1])
ax_ops = fig.add_subplot(gs[1, 1])

ml_color = "#2ecc71"
rule_color = "#e67e22"
text_color = "#2c3e50"

fig.suptitle("Evaluation Against the Baseline", fontsize=16, fontweight="bold", y=0.965)

x = np.arange(len(metrics))
w = 0.35
b1 = ax_metrics.bar(x - w / 2, metrics["ML Model"], w, color=ml_color, alpha=0.88, label="ML Model")
b2 = ax_metrics.bar(x + w / 2, metrics["Rule-Based"], w, color=rule_color, alpha=0.88, label="Rule-Based")

for bars in (b1, b2):
    for bar in bars:
        h = bar.get_height()
        ax_metrics.text(
            bar.get_x() + bar.get_width() / 2,
            h + 0.015,
            f"{h:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            color=text_color,
        )

ax_metrics.set_xticks(x)
ax_metrics.set_xticklabels(metrics["Metric"], fontsize=11)
ax_metrics.set_ylim(0, 1.05)
ax_metrics.set_ylabel("Score", fontsize=11)
ax_metrics.set_title("Default-Class Metrics and AUC", fontsize=12, fontweight="bold", pad=10)
ax_metrics.legend(fontsize=10, loc="upper left")
ax_metrics.grid(axis="y", alpha=0.35)
ax_metrics.spines[["top", "right"]].set_visible(False)
ax_metrics.set_facecolor("#fafafa")

error_names = ["False Positive Rate", "False Negative Rate"]
ml_error_vals = [rates_ml["fpr"], rates_ml["fnr"]]
rule_error_vals = [rates_rule["fpr"], rates_rule["fnr"]]
ypos = np.arange(len(error_names))

ax_errors.barh(ypos + 0.18, ml_error_vals, 0.34, color=ml_color, alpha=0.88, label="ML Model")
ax_errors.barh(ypos - 0.18, rule_error_vals, 0.34, color=rule_color, alpha=0.88, label="Rule-Based")
for i, (ml_v, rule_v) in enumerate(zip(ml_error_vals, rule_error_vals)):
    ax_errors.text(ml_v + 0.01, i + 0.18, f"{ml_v:.1%}", va="center", fontsize=9, fontweight="bold")
    ax_errors.text(rule_v + 0.01, i - 0.18, f"{rule_v:.1%}", va="center", fontsize=9, fontweight="bold")

ax_errors.set_yticks(ypos)
ax_errors.set_yticklabels(error_names, fontsize=10)
ax_errors.set_xlim(0, max(ml_error_vals + rule_error_vals) + 0.12)
ax_errors.set_title("Error Tradeoff", fontsize=12, fontweight="bold", pad=10)
ax_errors.grid(axis="x", alpha=0.35)
ax_errors.spines[["top", "right"]].set_visible(False)
ax_errors.set_facecolor("#fafafa")

ops_names = ["Approval Rate", "Intervention Rate"]
ml_ops = [rates_ml["approval_rate"], rates_ml["intervention_rate"]]
rule_ops = [rates_rule["approval_rate"], rates_rule["intervention_rate"]]
x2 = np.arange(len(ops_names))

c1 = ax_ops.bar(x2 - w / 2, ml_ops, w, color=ml_color, alpha=0.88, label="ML Model")
c2 = ax_ops.bar(x2 + w / 2, rule_ops, w, color=rule_color, alpha=0.88, label="Rule-Based")
for bars in (c1, c2):
    for bar in bars:
        h = bar.get_height()
        ax_ops.text(
            bar.get_x() + bar.get_width() / 2,
            h + 0.015,
            f"{h:.1%}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            color=text_color,
        )

ax_ops.set_xticks(x2)
ax_ops.set_xticklabels(ops_names, fontsize=10)
ax_ops.set_ylim(0, 1.05)
ax_ops.set_title("Operating Point at Threshold = 0.50", fontsize=12, fontweight="bold", pad=10)
ax_ops.grid(axis="y", alpha=0.35)
ax_ops.spines[["top", "right"]].set_visible(False)
ax_ops.set_facecolor("#fafafa")

plt.savefig(
    f"{OUTPUT_DIR}/18_evaluation_dashboard.png",
    dpi=150,
    bbox_inches="tight",
    facecolor="#fafafa",
)
plt.close()


# ---------------------------------------------------------------------------
# Figure 2: confusion matrices + deployment impact
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 5.6), gridspec_kw={"width_ratios": [1, 1, 1.2]})
fig.patch.set_facecolor("#fafafa")


def draw_confusion(ax, cm: np.ndarray, title: str, accent: str) -> None:
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Pred Repaid", "Pred Default"], fontsize=10)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Actual Repaid", "Actual Default"], fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.set_facecolor("#fafafa")
    ax.spines[:].set_visible(False)
    for i in range(2):
        for j in range(2):
            val = cm[i, j]
            ax.text(
                j, i, f"{val:,}",
                ha="center", va="center",
                fontsize=13, fontweight="bold",
                color="white" if val > cm.max() * 0.45 else text_color,
            )
    ax.add_patch(plt.Rectangle((-0.5, -0.5), 2, 2, fill=False, edgecolor=accent, linewidth=2))


draw_confusion(axes[0], cm_ml, "ML Model", ml_color)
draw_confusion(axes[1], cm_rule, "Rule-Based Baseline", rule_color)

axes[2].set_facecolor("#fafafa")
axes[2].axis("off")

tn_ml, fp_ml, fn_ml, tp_ml = cm_ml.ravel()
tn_rule, fp_rule, fn_rule, tp_rule = cm_rule.ravel()

impact_lines = [
    ("Defaults caught", tp_ml, tp_rule, tp_ml - tp_rule),
    ("Defaults missed", fn_ml, fn_rule, fn_ml - fn_rule),
    ("Good apps denied", fp_ml, fp_rule, fp_ml - fp_rule),
    ("Good apps approved", tn_ml, tn_rule, tn_ml - tn_rule),
]

axes[2].text(
    0.02, 0.95, "Deployment Impact vs Baseline",
    fontsize=13, fontweight="bold", color=text_color, transform=axes[2].transAxes
)
axes[2].text(
    0.02, 0.88, "Counts on the same resolved dataset",
    fontsize=10, color="#7f8c8d", transform=axes[2].transAxes
)

y0 = 0.75
for idx, (label, ml_v, rule_v, delta) in enumerate(impact_lines):
    yline = y0 - idx * 0.16
    delta_color = ml_color if ("caught" in label.lower() or "approved" in label.lower()) and delta > 0 else "#e74c3c"
    if ("missed" in label.lower() or "denied" in label.lower()) and delta < 0:
        delta_color = ml_color
    axes[2].text(0.02, yline, label, fontsize=10.5, fontweight="bold", color=text_color, transform=axes[2].transAxes)
    axes[2].text(0.02, yline - 0.055, f"ML: {ml_v:,}    Rule: {rule_v:,}", fontsize=10, color=text_color, transform=axes[2].transAxes)
    axes[2].text(
        0.72, yline - 0.055, f"{delta:+,}",
        fontsize=12, fontweight="bold", color=delta_color, transform=axes[2].transAxes
    )

axes[2].text(
    0.02, 0.08,
    f"FPR: {rates_ml['fpr']:.1%} vs {rates_rule['fpr']:.1%}\n"
    f"FNR: {rates_ml['fnr']:.1%} vs {rates_rule['fnr']:.1%}",
    fontsize=11, color=text_color, transform=axes[2].transAxes,
    bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="#d5d8dc")
)

fig.suptitle("Confusion Matrices and What Changes in Deployment", fontsize=15, fontweight="bold", y=0.98)
plt.savefig(
    f"{OUTPUT_DIR}/19_confusion_and_impact.png",
    dpi=150,
    bbox_inches="tight",
    facecolor="#fafafa",
)
plt.close()

print("Saved: 18_evaluation_dashboard.png")
print("Saved: 19_confusion_and_impact.png")
