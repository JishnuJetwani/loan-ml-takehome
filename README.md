# Loan Default Prediction

## Quick Start

```bash
pip install -r requirements.txt
python generate_data.py      # Creates loan_applications.csv (2,000 rows)
python loan_model.py          # Runs full analysis, saves plots to outputs/
```

## Approach & Key Decisions

### Data Handling
- **Ongoing applications (8.2%)**: Excluded. They have no outcome label, so including them would require assumptions about their eventual status. This introduces mild survivorship bias which I acknowledge but accept given the scope.
- **Missing documented income (15%)**: Treated as a signal. Through some graphing I hypothesised that a lack of documented is a signal corresponding to higher chance of default.
- **Class imbalance (~70% repaid, 30% defaulted)**: Handled via `class_weight='balanced'`, which reweights the loss function so the model doesn't learn to predict repaid for everyone.

### Feature Engineering

| Feature | Rationale |
|---------|-----------|
| `loan_to_income` | Loan size relative to stated income — affordability signal |
| `withdrawal_ratio` | Monthly withdrawals / deposits — spending behaviour |
| `income_doc_ratio` | Documented / stated income — consistency check |
| `balance_to_loan` | Buffer relative to loan size |
| `has_documentation` | Absence of income docs is itself a risk flag |
| `possible_misrep` | Stated income > 3× documented — potential fraud signal |
| `income_verified` | Stated and documented income agree within 10% |
| `is_unemployed` | Unemployment (not self-employment) is a genuine risk signal |
| `bank_has_overdrafts` | Direct indicator of account instability |

### Key Findings

**Fairness**: The rule-based system penalizes self-employed applicants (scoring them 60 vs 100 for employed), despite their actual default rate (29.0%) being nearly identical to employed applicants (27.2%). The ML model corrects this: self-employed approval rate rises from 35.5% → 58.8%, nearly matching employed at 64.3%. Unemployment remains appropriately flagged — unemployed applicants have a genuinely higher default rate (46.7%) and are approved at lower rates by both systems.

**Performance vs Baseline**: AUC-ROC improves from 0.7064 (rule-based) → 0.7243 (ML model). The ML model catches 65.5% of defaults (vs 26.2% for rule-based), with 188 false negatives vs 402. See `outputs/05_roc_pr_curves.png`.

**Business Value**: At the profit-maximising threshold (0.52), the ML model saves **$99,740** vs the rule-based system on the evaluation portfoliom a 38.7% reduction in losses. The optimal threshold is slightly above 0.5 because a missed defaulter (65% LGD) costs 4× more than a wrongly rejected good customer (16% margin). See `outputs/15_profit_maximisation.png`.

**False Negatives**: Of the 545 actual defaults, 200 slip through at the optimal threshold. These are  deceptive cases with smaller loans (avg $1,451 vs $2,133 for caught defaults), verified income (92.6% vs 58.5%), and almost no overdrafts (4.3% vs 38.4%). The model is fooled by these strong positive signals. See `outputs/11_false_negative_analysis.png`.

## What I'd Do With More Time
1. **Survival analysis** for ongoing applications instead of excluding them.
3. **Feature interactions** — e.g., low balance + overdrafts might be more than additive (partially explored via SHAP dependence plots).
4. **Monitoring dashboard** — Drift detection, fairness metric tracking, and prediction distribution alerts for production deployment.
5. **Larger dataset** — More historical data would likely improve performance; the model's learning curve had not fully plateaued at 1,836 training examples.

## Repository Structure
```
├── README.md
├── requirements.txt
├── generate_data.py                # Dataset generation script
├── loan_applications.csv           # Generated dataset (2,000 rows)
├── loan_model.py                   # Full analysis pipeline (main deliverable)
├── cost_analysis.py                # Business cost & profit-maximisation analysis
├── false_negative_analysis.py      # Deep-dive on defaults that slipped through
└── plots/                          # Standalone chart scripts
    ├── plot_data_handling_pies.py  # EDA: outcome distribution before/after filtering
    ├── plot_evaluation_baseline.py # Evaluation dashboard vs rule-based baseline
    ├── plot_fairness.py            # Fairness analysis across employment groups
    └── plot_profit.py              # Profit-maximisation chart
└── outputs/
    ├── 01_eda_overview.png
    ├── 02_feature_distributions.png
    ├── 03_employment_analysis.png
    ├── 04_confusion_matrices.png
    ├── 05_roc_pr_curves.png
    ├── 05b_threshold_optimization.png
    ├── 06_shap_summary.png
    ├── 07_shap_bar.png
    ├── 08_shap_waterfall_example.png
    ├── 08b_shap_denial_analysis.png
    ├── 09_fairness_analysis.png
    ├── 10_shap_employment.png
    ├── 11_false_negative_analysis.png
    ├── 12_fn_vs_tp_shap.png
    ├── 13_cost_analysis.png
    ├── 14_scenario_comparison.png
    ├── 15_profit_maximisation.png
    ├── 16_fairness_chart.png
    ├── 17_outcome_split_pies.png
    ├── 18_evaluation_dashboard.png
    └── 19_confusion_and_impact.png
```
