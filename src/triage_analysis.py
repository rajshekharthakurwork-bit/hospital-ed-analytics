import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs('outputs/plots',   exist_ok=True)
os.makedirs('outputs/reports', exist_ok=True)

df  = pd.read_csv('data/ed_patient_data.csv', parse_dates=['arrival_datetime'])
ESI = {1:'Critical',2:'Emergent',3:'Urgent',4:'Less Urgent',5:'Minor'}
BENCH = {1:5, 2:15, 3:30, 4:60, 5:120}

df['benchmark_min'] = df['triage_level'].map(BENCH)
df['breach']        = (df['wait_time_doctor_min'] > df['benchmark_min']).astype(int)

# --- Breach by triage ---
bt = df.groupby('triage_level').agg(
    total=('visit_id','count'), breaches=('breach','sum'),
    avg_wait=('wait_time_doctor_min','mean'),
    median_wait=('wait_time_doctor_min','median'),
    max_wait=('wait_time_doctor_min','max')
).reset_index()
bt['breach_pct'] = (bt['breaches']/bt['total']*100).round(1)
bt['label']      = bt['triage_level'].map(ESI)
bt.to_csv('outputs/reports/breach_by_triage.csv', index=False)
print("BREACH RATES BY TRIAGE:\n", bt[['triage_level','label','total','breaches','breach_pct','avg_wait']].to_string(index=False))

# --- Breach by hour ---
bh = df.groupby('arrival_hour').agg(total=('visit_id','count'), breaches=('breach','sum')).reset_index()
bh['breach_pct'] = (bh['breaches']/bh['total']*100).round(1)
bh.to_csv('outputs/reports/breach_by_hour.csv', index=False)
print("\nWORST 5 HOURS:\n", bh.sort_values('breach_pct', ascending=False).head(5).to_string(index=False))

# --- Breach by department ---
bd = df.groupby('department').agg(total=('visit_id','count'), breaches=('breach','sum'),
                                    avg_wait=('wait_time_doctor_min','mean')).reset_index()
bd['breach_pct'] = (bd['breaches']/bd['total']*100).round(1)
bd = bd.sort_values('breach_pct', ascending=False)
bd.to_csv('outputs/reports/breach_by_department.csv', index=False)
print("\nBREACH BY DEPARTMENT:\n", bd.to_string(index=False))

# --- High-risk flag: ESI 1 or 2 + breach ---
hr = df[(df['triage_level']<=2) & (df['breach']==1)].copy()
hr['flag'] = 'CRITICAL BREACH'
hr[['visit_id','arrival_datetime','triage_level','department',
    'wait_time_doctor_min','benchmark_min','attending_physician','flag']]\
    .to_csv('outputs/reports/high_risk_breaches.csv', index=False)
print(f"\nHigh-risk breach flags: {len(hr)} visits saved to outputs/reports/high_risk_breaches.csv")

# ===== CHART 1: Breach rate bars by triage =====
fig, ax = plt.subplots(figsize=(9, 5))
colors = ['#E24B4A','#D4537E','#534AB7','#1D9E75','#888780']
bars = ax.bar(bt['label'], bt['breach_pct'], color=colors, edgecolor='none')
for bar, val in zip(bars, bt['breach_pct']):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
            f'{val}%', ha='center', fontsize=10, fontweight='bold')
ax.axhline(50, color='gray', ls='--', lw=1)
ax.set_ylim(0, 105); ax.set_ylabel('% of Visits Exceeding Benchmark')
ax.set_title('Wait Time Benchmark Breach Rate by Triage Level', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/plots/09_breach_by_triage.png', dpi=150, bbox_inches='tight')
plt.close(); print("Saved → outputs/plots/09_breach_by_triage.png")

# ===== CHART 2: Breach rate by hour =====
fig, ax = plt.subplots(figsize=(14, 4))
ax.fill_between(bh['arrival_hour'], bh['breach_pct'], alpha=0.3, color='#E24B4A')
ax.plot(bh['arrival_hour'], bh['breach_pct'], color='#E24B4A', lw=2, marker='o', ms=4)
ax.axhline(bh['breach_pct'].mean(), color='gray', ls='--', lw=1, label='Overall avg')
ax.set_xticks(range(24)); ax.set_xlabel('Hour of Day'); ax.set_ylabel('Breach Rate (%)')
ax.set_title('Benchmark Breach Rate by Hour', fontsize=12, fontweight='bold')
ax.legend(); plt.tight_layout()
plt.savefig('outputs/plots/10_breach_by_hour.png', dpi=150, bbox_inches='tight')
plt.close(); print("Saved → outputs/plots/10_breach_by_hour.png")

# ===== CHART 3: LOS by department (horizontal bar) =====
los_d = df.groupby('department')['length_of_stay_hrs'].mean().sort_values()
fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.barh(los_d.index, los_d.values, color='#534AB7', edgecolor='none')
for bar, val in zip(bars, los_d.values):
    ax.text(val+0.03, bar.get_y()+bar.get_height()/2, f'{val:.2f}h', va='center', fontsize=9)
ax.axvline(los_d.mean(), color='gray', ls='--', lw=1, label='Average')
ax.set_title('Average Length of Stay by Department', fontsize=12, fontweight='bold')
ax.set_xlabel('Hours'); ax.legend(); plt.tight_layout()
plt.savefig('outputs/plots/11_los_by_department.png', dpi=150, bbox_inches='tight')
plt.close(); print("Saved → outputs/plots/11_los_by_department.png")

# ===== CHART 4: Violin plot — doctor wait by triage =====
fig, ax = plt.subplots(figsize=(10, 5))
palette = {1:'#E24B4A',2:'#D4537E',3:'#534AB7',4:'#1D9E75',5:'#888780'}
sns.violinplot(data=df, x='triage_level', y='wait_time_doctor_min',
               palette=palette, ax=ax, inner='quartile', cut=0)
for esi, bench in BENCH.items():
    ax.scatter([esi-1],[bench], color='black', marker='D', s=50, zorder=5)
ax.set_title('Doctor Wait Distribution by Triage (◆ = benchmark)', fontsize=12, fontweight='bold')
ax.set_xlabel('ESI Level'); ax.set_ylabel('Minutes'); ax.set_ylim(0, 260)
plt.tight_layout()
plt.savefig('outputs/plots/12_wait_violin.png', dpi=150, bbox_inches='tight')
plt.close(); print("Saved → outputs/plots/12_wait_violin.png")
print("\nDay 4 complete.")