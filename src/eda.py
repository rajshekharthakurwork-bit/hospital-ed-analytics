import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

os.makedirs('outputs/plots',   exist_ok=True)
os.makedirs('outputs/reports', exist_ok=True)

df = pd.read_csv('data/ed_patient_data.csv', parse_dates=['arrival_datetime'])
num_cols = ['patient_age','wait_time_triage_min','wait_time_doctor_min',
            'length_of_stay_hrs','satisfaction_score','billed_amount_usd']

# --- Descriptive stats ---
summary = df[num_cols].describe().T
summary['skew']     = df[num_cols].skew()
summary['kurtosis'] = df[num_cols].kurtosis()
summary['IQR']      = summary['75%'] - summary['25%']
summary.to_csv('outputs/reports/descriptive_stats.csv')
print("=== DESCRIPTIVE STATS ===")
print(summary.round(2).to_string())

# --- Outliers (IQR method) ---
print("\n=== OUTLIERS ===")
outlier_rows = []
for col in num_cols:
    Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    IQR = Q3 - Q1
    lo, hi = Q1 - 1.5*IQR, Q3 + 1.5*IQR
    n = ((df[col] < lo) | (df[col] > hi)).sum()
    print(f"  {col}: {n} outliers ({round(n/len(df)*100,1)}%)")
    outlier_rows.append({'column':col,'outliers':n,'pct':round(n/len(df)*100,1),'lo':round(lo,2),'hi':round(hi,2)})
pd.DataFrame(outlier_rows).to_csv('outputs/reports/outlier_summary.csv', index=False)

# --- Correlation ---
corr_cols = num_cols + ['triage_level','readmitted_30d']
corr = df[corr_cols].corr()
r, p = stats.pearsonr(df['wait_time_doctor_min'], df['satisfaction_score'])
print(f"\nSatisfaction vs Wait: r={r:.3f}, p={p:.2e}")

# --- Breach rates ---
benchmarks = {1:5, 2:15, 3:30, 4:60, 5:120}
breach_rows = []
print("\n=== BREACH RATES ===")
for esi, bench in benchmarks.items():
    sub = df[df['triage_level']==esi]
    b   = (sub['wait_time_doctor_min'] > bench).sum()
    pct = round(b/len(sub)*100, 1)
    print(f"  ESI {esi} (≤{bench}min): {b}/{len(sub)} = {pct}% breached")
    breach_rows.append({'esi':esi,'benchmark':bench,'breached':b,'total':len(sub),'pct':pct})
pd.DataFrame(breach_rows).to_csv('outputs/reports/breach_rates.csv', index=False)

# ===== CHART 1: Feature distributions (2×3 grid) =====
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.suptitle('Distribution of Numeric Features', fontsize=14, fontweight='bold')
colors = ['#378ADD','#534AB7','#E24B4A','#1D9E75','#BA7517','#888780']
for ax, col, color in zip(axes.flatten(), num_cols, colors):
    ax.hist(df[col], bins=40, color=color, alpha=0.8, edgecolor='none')
    ax.axvline(df[col].mean(),   color='black', ls='--', lw=1.2, label=f'Mean {df[col].mean():.1f}')
    ax.axvline(df[col].median(), color='gray',  ls=':',  lw=1.0, label=f'Median {df[col].median():.1f}')
    ax.set_title(col.replace('_',' ').title(), fontsize=10)
    ax.legend(fontsize=7)
plt.tight_layout()
plt.savefig('outputs/plots/01_feature_distributions.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved → outputs/plots/01_feature_distributions.png")

# ===== CHART 2: Box plots — wait time by triage =====
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle('Wait Time by Triage Level (ESI)', fontsize=13, fontweight='bold')
palette = {1:'#E24B4A',2:'#D4537E',3:'#534AB7',4:'#1D9E75',5:'#888780'}
for ax, col, title in zip(axes,
    ['wait_time_triage_min','wait_time_doctor_min'],
    ['Triage Wait (min)','Doctor Wait (min)']):
    sns.boxplot(data=df, x='triage_level', y=col, palette=palette, ax=ax,
                flierprops=dict(marker='o', markersize=2, alpha=0.3))
    ax.set_title(title); ax.set_xlabel('ESI Level'); ax.set_ylabel('Minutes')
plt.tight_layout()
plt.savefig('outputs/plots/02_wait_boxplots.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved → outputs/plots/02_wait_boxplots.png")

# ===== CHART 3: Correlation heatmap =====
fig, ax = plt.subplots(figsize=(9, 7))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
            vmin=-1, vmax=1, center=0, ax=ax, linewidths=0.5, annot_kws={'size':9})
ax.set_title('Pearson Correlation Matrix', fontsize=13, fontweight='bold', pad=14)
plt.tight_layout()
plt.savefig('outputs/plots/03_correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved → outputs/plots/03_correlation_heatmap.png")

# ===== CHART 4: Categorical distributions (2×2) =====
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
fig.suptitle('Categorical Feature Distributions', fontsize=13, fontweight='bold')

ic = df['insurance_type'].value_counts()
axes[0,0].bar(ic.index, ic.values, color='#378ADD', edgecolor='none')
axes[0,0].set_title('Visits by Insurance Type'); axes[0,0].set_ylabel('Count')

dc = df['disposition'].value_counts()
axes[0,1].bar(dc.index, dc.values, color=['#1D9E75','#378ADD','#BA7517','#888780','#E24B4A'], edgecolor='none')
axes[0,1].set_title('Visits by Disposition'); axes[0,1].tick_params(axis='x', rotation=20)

ac = df['age_group'].value_counts().sort_index()
axes[1,0].bar(ac.index.astype(str), ac.values, color='#534AB7', edgecolor='none')
axes[1,0].set_title('Visits by Age Group'); axes[1,0].set_ylabel('Count')

ra = df.groupby('age_group', observed=True)['readmitted_30d'].mean() * 100
axes[1,1].bar(ra.index.astype(str), ra.values, color='#E24B4A', edgecolor='none')
axes[1,1].set_title('30-Day Readmission Rate by Age Group (%)')
axes[1,1].set_ylabel('%')

plt.tight_layout()
plt.savefig('outputs/plots/04_categorical_breakdown.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved → outputs/plots/04_categorical_breakdown.png")
print("\nDay 2 complete. Check outputs/plots/ and outputs/reports/")