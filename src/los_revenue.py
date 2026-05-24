import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

os.makedirs('outputs/plots',   exist_ok=True)
os.makedirs('outputs/reports', exist_ok=True)

df = pd.read_csv('data/ed_patient_data.csv')

# --- LOS by department ---
los_dept = df.groupby('department').agg(
    visits=('visit_id','count'), mean_los=('length_of_stay_hrs','mean'),
    median_los=('length_of_stay_hrs','median'), std_los=('length_of_stay_hrs','std'),
    max_los=('length_of_stay_hrs','max')
).round(2).sort_values('mean_los', ascending=False)
los_dept.to_csv('outputs/reports/los_by_department.csv')
print("LOS BY DEPARTMENT:\n", los_dept.to_string())

# --- Revenue by insurance ---
rev = df.groupby('insurance_type').agg(
    visits=('visit_id','count'),
    total_billed=('billed_amount_usd','sum'),
    avg_bill=('billed_amount_usd','mean'),
    median_bill=('billed_amount_usd','median')
).round(2).sort_values('total_billed', ascending=False)
rev['visit_share_pct']   = (rev['visits'] / rev['visits'].sum() * 100).round(1)
rev['revenue_share_pct'] = (rev['total_billed'] / rev['total_billed'].sum() * 100).round(1)
rev.to_csv('outputs/reports/revenue_by_insurance.csv')
print("\nREVENUE BY INSURANCE:\n", rev.to_string())

# --- Revenue by department ---
rev_dept = df.groupby('department').agg(
    visits=('visit_id','count'),
    total_billed=('billed_amount_usd','sum'),
    avg_bill=('billed_amount_usd','mean')
).round(2).sort_values('total_billed', ascending=False)
rev_dept.to_csv('outputs/reports/revenue_by_department.csv')

# --- Satisfaction correlations ---
for col, label in [('wait_time_doctor_min','Doctor Wait'),
                    ('length_of_stay_hrs','LOS'),
                    ('billed_amount_usd','Bill Amount')]:
    r, p = stats.pearsonr(df[col], df['satisfaction_score'])
    print(f"Satisfaction vs {label}: r={r:.3f}, p={p:.2e}")

sat_by_triage = df.groupby('triage_level')['satisfaction_score'].agg(['mean','std']).round(2)
sat_by_triage.to_csv('outputs/reports/satisfaction_by_triage.csv')
print("\nSatisfaction by triage:\n", sat_by_triage.to_string())

# ===== CHART 1: Revenue share vs visit share (grouped bar) =====
fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(rev)); w = 0.35
ax.bar(x-w/2, rev['revenue_share_pct'], w, label='Revenue share %', color='#185FA5')
ax.bar(x+w/2, rev['visit_share_pct'],   w, label='Visit share %',   color='#B5D4F4')
ax.set_xticks(x); ax.set_xticklabels(rev.index)
ax.set_ylabel('%'); ax.legend()
ax.set_title('Revenue Share vs Visit Share by Insurance', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/plots/17_revenue_vs_visit_share.png', dpi=150, bbox_inches='tight')
plt.close(); print("Saved → outputs/plots/17_revenue_vs_visit_share.png")

# ===== CHART 2: Avg bill per visit by insurance =====
fig, ax = plt.subplots(figsize=(8, 4))
colors = ['#185FA5','#378ADD','#534AB7','#888780','#E24B4A']
bars = ax.barh(rev.index, rev['avg_bill'], color=colors, edgecolor='none')
for bar, val in zip(bars, rev['avg_bill']):
    ax.text(val+30, bar.get_y()+bar.get_height()/2, f'${val:,.0f}', va='center', fontsize=9)
ax.set_title('Average Bill per Visit by Insurance Type', fontsize=12, fontweight='bold')
ax.set_xlabel('USD'); plt.tight_layout()
plt.savefig('outputs/plots/18_avg_bill_by_insurance.png', dpi=150, bbox_inches='tight')
plt.close(); print("Saved → outputs/plots/18_avg_bill_by_insurance.png")

# ===== CHART 3: Satisfaction vs wait scatter + regression line =====
sample = df.sample(2000, random_state=42)
slope, intercept, r, p, _ = stats.linregress(sample['wait_time_doctor_min'],
                                               sample['satisfaction_score'])
fig, ax = plt.subplots(figsize=(9, 5))
palette = {1:'#E24B4A',2:'#D4537E',3:'#534AB7',4:'#1D9E75',5:'#888780'}
for esi, grp in sample.groupby('triage_level'):
    ax.scatter(grp['wait_time_doctor_min'], grp['satisfaction_score'],
               alpha=0.3, s=15, color=palette[esi], label=f'ESI {esi}')
x_line = np.array([0, 240])
ax.plot(x_line, slope*x_line+intercept, color='black', lw=2, label=f'Regression (r={r:.2f})')
ax.set_xlabel('Doctor Wait Time (min)'); ax.set_ylabel('Satisfaction Score (1–10)')
ax.set_title('Patient Satisfaction vs Doctor Wait Time', fontsize=12, fontweight='bold')
ax.legend(fontsize=8, ncol=2); plt.tight_layout()
plt.savefig('outputs/plots/19_satisfaction_vs_wait.png', dpi=150, bbox_inches='tight')
plt.close(); print("Saved → outputs/plots/19_satisfaction_vs_wait.png")

# ===== CHART 4: LOS box plots by department =====
dept_order = los_dept.index.tolist()
fig, ax = plt.subplots(figsize=(12, 5))
sns.boxplot(data=df, x='department', y='length_of_stay_hrs', order=dept_order,
            color='#534AB7', ax=ax, flierprops=dict(marker='o', markersize=2, alpha=0.3))
ax.set_xticklabels(dept_order, rotation=25, ha='right')
ax.set_title('Length of Stay Distribution by Department', fontsize=12, fontweight='bold')
ax.set_ylabel('Hours'); ax.set_xlabel('')
plt.tight_layout()
plt.savefig('outputs/plots/20_los_boxplot_by_dept.png', dpi=150, bbox_inches='tight')
plt.close(); print("Saved → outputs/plots/20_los_boxplot_by_dept.png")
print("\nDay 6 complete.")