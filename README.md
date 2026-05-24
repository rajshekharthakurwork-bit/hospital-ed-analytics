<img width="1912" height="869" alt="Screenshot 2026-05-24 151515" src="https://github.com/user-attachments/assets/f9729e27-5caa-41ef-8f36-95a103d089ca" />
# 🏥 Hospital Emergency Department — Patient Flow & Operational Analytics

> A full end-to-end data analytics project simulating real hospital ED operations.
> Built to demonstrate skills used by healthcare analytics teams at Apollo, Manipal, and AIIMS.

---

## 📌 Project Overview

This project analyses 10,000 synthetic ED patient visits across a full year.
It answers 6 real operational questions that hospital management teams care about:

| Question | Analysis |
|----------|----------|
| Which hours and days overwhelm the ED? | Time-series demand heatmaps |
| Which triage levels are waiting too long? | Clinical benchmark breach analysis |
| What factors predict 30-day readmissions? | ML classification model |
| Which departments have the longest stay? | LOS bottleneck analysis |
| Does wait time hurt satisfaction scores? | Correlation + regression |
| Which insurance types drive the most revenue? | Payer-mix financial analysis |


## 🔗 Links

| Platform | Link |
|----------|------|
| 📊 Kaggle Dataset | [Hospital ED Dataset](https://www.kaggle.com/datasets/rajshekhar43/hospital-emergency-department-patient-flow-dataset) |
| 💻 GitHub Repo | [hospital-ed-analytics](https://github.com/rajshekharthakurwork-bit/hospital-ed-analytics) |

## 🔑 Key Findings

- **92.9%** of ESI 1 (cardiac arrest, septic shock) patients exceeded the 5-minute safety benchmark
- **Satisfaction vs wait time** correlation: r = −0.77 — the strongest signal in the dataset
- **65+ patients** readmit at 20%, nearly double the rate of younger groups
- **Private insurance** generates $5,619/visit vs $3,142 for self-pay — a 79% revenue gap
- **Trauma + Cardiology** have the highest average LOS, flagged as operational bottlenecks
- Random Forest readmission model achieved **AUC ~0.75**

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python  | Core language |
| Pandas, NumPy | Data manipulation |
| Matplotlib, Seaborn | Static charts |
| Scikit-learn | ML model (Logistic Regression + Random Forest) |
| Plotly + Dash | Interactive dashboard |
| SciPy | Statistical tests |


## 📈 Dashboard Preview

The dashboard includes:
- 6 live KPI cards (visits, wait time, LOS, readmission rate, satisfaction, revenue)
- Hourly arrival volume chart
- Benchmark breach rate by triage level
- LOS by department
- Revenue by insurance type
- Satisfaction vs wait time scatter with regression line

  Dashboard http://127.0.0.1:8050/


## 💡 Skills Demonstrated

- Time-series analysis and demand forecasting
- Feature engineering for clinical risk scoring
- Binary classification (Logistic Regression + Random Forest)
- Correlation analysis and statistical significance testing
- Cohort analysis (age groups, payer mix, triage levels)
- Interactive dashboard development
- Healthcare domain knowledge (ESI triage, CTAS benchmarks, CMS readmission metrics)
