"""
Data Handling Outcome Mix Chart
===============================
Creates a single figure with two pie charts:
1. Full dataset outcome mix
2. Outcome mix after removing ongoing applications
"""

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUTPUT_DIR = "outputs"

COLORS = {
    'repaid': '#2ecc71',
    'defaulted': '#e74c3c',
    'ongoing': '#95a5a6',
}


def autopct_with_count(values):
    total = sum(values)

    def _formatter(pct):
        count = int(round(pct / 100 * total))
        return f"{pct:.1f}%\n(n={count})"

    return _formatter


df = pd.read_csv("loan_applications.csv")
resolved = df[df['actual_outcome'] != 'ongoing']

full_counts = (
    df['actual_outcome']
    .value_counts()
    .reindex(['repaid', 'defaulted', 'ongoing'])
)

resolved_counts = (
    resolved['actual_outcome']
    .value_counts()
    .reindex(['repaid', 'defaulted'])
)

fig, axes = plt.subplots(1, 2, figsize=(13, 6))
fig.patch.set_facecolor('#fafafa')

for ax in axes:
    ax.set_facecolor('#fafafa')

wedges_1, _, autotexts_1 = axes[0].pie(
    full_counts.values,
    labels=['Repaid', 'Defaulted', 'Ongoing'],
    colors=[COLORS['repaid'], COLORS['defaulted'], COLORS['ongoing']],
    startangle=90,
    counterclock=False,
    autopct=autopct_with_count(full_counts.values),
    pctdistance=0.78,
    labeldistance=1.08,
    wedgeprops=dict(width=0.42, edgecolor='white', linewidth=2),
    textprops=dict(color='#2c3e50', fontsize=10),
)
for text in autotexts_1:
    text.set_fontsize(8.5)
    text.set_fontweight('bold')

axes[0].text(0, 0.05, 'All Apps', ha='center', va='center',
             fontsize=13, fontweight='bold', color='#2c3e50')
axes[0].text(0, -0.13, f'n = {len(df):,}', ha='center', va='center',
             fontsize=10, color='#7f8c8d')
axes[0].set_title('Raw Outcome Mix', fontsize=12, fontweight='bold', pad=14)

wedges_2, _, autotexts_2 = axes[1].pie(
    resolved_counts.values,
    labels=['Repaid', 'Defaulted'],
    colors=[COLORS['repaid'], COLORS['defaulted']],
    startangle=90,
    counterclock=False,
    autopct=autopct_with_count(resolved_counts.values),
    pctdistance=0.78,
    labeldistance=1.08,
    wedgeprops=dict(width=0.42, edgecolor='white', linewidth=2),
    textprops=dict(color='#2c3e50', fontsize=10),
)
for text in autotexts_2:
    text.set_fontsize(8.5)
    text.set_fontweight('bold')

axes[1].text(0, 0.05, 'Resolved Only', ha='center', va='center',
             fontsize=13, fontweight='bold', color='#2c3e50')
axes[1].text(0, -0.13, f'n = {len(resolved):,}', ha='center', va='center',
             fontsize=10, color='#7f8c8d')
axes[1].set_title('After Removing Ongoing', fontsize=12, fontweight='bold', pad=14)

filtered = len(df) - len(resolved)
fig.suptitle('Outcome Distribution Before and After Filtering Ongoing Applications',
             fontsize=14, fontweight='bold', y=0.97)
fig.text(0.5, 0.04,
         f'Filtered out {filtered} ongoing applications ({filtered / len(df):.1%}) before training',
         ha='center', fontsize=10, color='#7f8c8d')

plt.tight_layout(rect=[0, 0.08, 1, 0.93])
plt.savefig(f"{OUTPUT_DIR}/17_outcome_split_pies.png", dpi=150,
            bbox_inches='tight', facecolor='#fafafa')
plt.close()

print("Saved: 17_outcome_split_pies.png")
