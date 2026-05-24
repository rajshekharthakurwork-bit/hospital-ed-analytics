import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs('outputs/plots',   exist_ok=True)
os.makedirs('outputs/reports', exist_ok=True)

df = pd.read_csv('data/ed_patient_data.csv', parse_dates=['arrival_datetime'])
day_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']

# --- Volume summaries ---
hourly   = df.groupby('arrival_hour').size()
by_day   = df.groupby('arrival_dayofweek').size().reindex(day_order)
by_month = df.groupby('arrival_month').size()
avg_wait_month = df.groupby('arrival_month')['wait_time_doctor_min'].mean().round(1)

print("Peak hours (top 5):", hourly.sort_values(ascending=False).head(5).to_dict())
print("Busiest days:\n", by_day.to_string())

# --- Heatmap pivot: day × hour ---
pivot = df.groupby(['arrival_dayofweek','arrival_hour']).size().unstack(fill_value=0)
pivot = pivot.reindex(day_order)

# --- Peak / off-peak classification ---
avg_per_hour   = hourly / df['arrival_datetime'].dt.date.nunique()
overall_mean   = avg_per_hour.mean()
peak_hours     = avg_per_hour[avg_per_hour > overall_mean * 1.2].index.tolist()
offpeak_hours  = avg_per_hour[avg_per_hour < overall_mean * 0.6].index.tolist()
print(f"\nPeak hours (>120% avg): {peak_hours}")
print(f"Off-peak hours (<60% avg): {offpeak_hours}")

# --- Triage mix by hour ---
triage_hour = df.groupby(['arrival_hour','triage_level']).size().unstack(fill_value=0)
triage_pct  = triage_hour.div(triage_hour.sum(axis=1), axis=0) * 100
critical_pct = triage_pct[[1,2]].sum(axis=1)
print("\nTop hours for critical cases (ESI 1+2):")
print(critical_pct.sort_values(ascending=False).head(6).to_string())

# --- Save staffing report ---
staffing = pd.DataFrame({
    'hour': avg_per_hour.index,
    'avg_arrivals_per_day': avg_per_hour.round(1).values,
    'zone': ['PEAK' if h in peak_hours else 'OFF-PEAK' if h in offpeak_hours else 'NORMAL'
             for h in avg_per_hour.index]
})
staffing.to_csv('outputs/reports/staffing_by_hour.csv', index=False)
print("Saved → outputs/reports/staffing_by_hour.csv")

monthly_df = pd.DataFrame({'visits': by_month, 'avg_wait_min': avg_wait_month})
monthly_df.to_csv('outputs/reports/monthly_volume.csv')
print("Saved → outputs/reports/monthly_volume.csv")

# ===== CHART 1: Arrival heatmap =====
fig, ax = plt.subplots(figsize=(16, 5))
sns.heatmap(pivot, cmap='YlOrRd', ax=ax, linewidths=0.3, linecolor='white',
            cbar_kws={'label':'Patient Arrivals'})
ax.set_title('ED Arrivals — Hour of Day × Day of Week', fontsize=13, fontweight='bold', pad=12)
ax.set_xlabel('Hour of Day'); ax.set_ylabel('')
plt.tight_layout()
plt.savefig('outputs/plots/05_arrival_heatmap.png', dpi=150, bbox_inches='tight')
plt.close(); print("Saved → outputs/plots/05_arrival_heatmap.png")

# ===== CHART 2: Hourly volume bar (peak highlighted) =====
fig, ax = plt.subplots(figsize=(14, 5))
bar_colors = ['#E24B4A' if h in peak_hours else '#B5D4F4' for h in hourly.index]
ax.bar(hourly.index, hourly.values, color=bar_colors, edgecolor='none')
ax.axhline(hourly.mean(), color='gray', ls='--', lw=1, label='Average')
ax.set_xticks(range(24)); ax.set_xlabel('Hour of Day'); ax.set_ylabel('Total Arrivals (year)')
ax.set_title('Hourly Arrivals — Red = Peak Hours', fontsize=13, fontweight='bold')
ax.legend(); plt.tight_layout()
plt.savefig('outputs/plots/06_hourly_volume.png', dpi=150, bbox_inches='tight')
plt.close(); print("Saved → outputs/plots/06_hourly_volume.png")

# ===== CHART 3: Monthly dual-axis =====
fig, ax1 = plt.subplots(figsize=(12, 5))
ax2 = ax1.twinx()
ax1.bar(by_month.index, by_month.values, color='#378ADD', alpha=0.7, label='Visits')
ax2.plot(avg_wait_month.index, avg_wait_month.values, color='#E24B4A',
         lw=2, marker='o', ms=5, label='Avg Wait (min)')
ax1.set_xticks(range(1,13))
ax1.set_xticklabels(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])
ax1.set_ylabel('Visits', color='#378ADD'); ax2.set_ylabel('Avg Doctor Wait (min)', color='#E24B4A')
ax1.set_title('Monthly Volume vs Average Wait Time', fontsize=13, fontweight='bold')
h1,l1 = ax1.get_legend_handles_labels(); h2,l2 = ax2.get_legend_handles_labels()
ax1.legend(h1+h2, l1+l2, loc='upper left')
plt.tight_layout()
plt.savefig('outputs/plots/07_monthly_trend.png', dpi=150, bbox_inches='tight')
plt.close(); print("Saved → outputs/plots/07_monthly_trend.png")

# ===== CHART 4: Triage mix stacked bar by hour =====
triage_colors = {1:'#E24B4A',2:'#D4537E',3:'#534AB7',4:'#1D9E75',5:'#B5D4F4'}
fig, ax = plt.subplots(figsize=(14, 5))
bottom = np.zeros(24)
for esi in [5,4,3,2,1]:
    vals = triage_pct[esi].values if esi in triage_pct.columns else np.zeros(24)
    ax.bar(triage_pct.index, vals, bottom=bottom,
           color=triage_colors[esi], label=f'ESI {esi}', edgecolor='none')
    bottom += vals
ax.set_title('Triage Level Mix by Hour (%)', fontsize=13, fontweight='bold')
ax.set_xlabel('Hour'); ax.set_ylabel('% of Arrivals')
ax.set_xticks(range(24)); ax.legend(loc='upper left', fontsize=9)
plt.tight_layout()
plt.savefig('outputs/plots/08_triage_mix_by_hour.png', dpi=150, bbox_inches='tight')
plt.close(); print("Saved → outputs/plots/08_triage_mix_by_hour.png")
print("\nDay 3 complete.")