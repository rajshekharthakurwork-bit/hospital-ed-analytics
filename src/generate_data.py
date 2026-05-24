
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random, os

np.random.seed(42)
random.seed(42)

N = 10000
start_date = datetime(2023, 1, 1)
end_date   = datetime(2023, 12, 31)

# --- Arrivals with realistic hour/day weights ---
def gen_arrivals(n):
    hour_w = [0.5,0.4,0.3,0.3,0.4,0.6,1.0,1.4,1.6,1.8,2.0,2.1,
              2.0,1.9,1.8,1.7,1.8,2.0,2.2,2.1,1.9,1.6,1.2,0.8]
    hour_w = np.array(hour_w) / sum(hour_w)
    total  = (end_date - start_date).days + 1
    dates  = []
    for _ in range(n):
        d = start_date + timedelta(days=int(np.random.choice(total)))
        h = np.random.choice(24, p=hour_w)
        dates.append(d.replace(hour=h, minute=random.randint(0,59), second=random.randint(0,59)))
    return sorted(dates)

arrivals = gen_arrivals(N)

# --- Triage ESI 1 (critical) to 5 (minor) ---
triage = np.random.choice([1,2,3,4,5], size=N, p=[0.05,0.15,0.35,0.30,0.15])

# --- Wait times in minutes ---
base_triage = {1:3, 2:8,  3:25, 4:45, 5:70}
base_doctor = {1:10,2:20, 3:40, 4:70, 5:100}
def wt(t, base, spread, cap):
    return round(min(np.random.exponential(base[t]) + np.random.uniform(0, spread), cap), 1)
wait_triage = [wt(t, base_triage, 5, 30)  for t in triage]
wait_doctor = [wt(t, base_doctor, 15, 240) for t in triage]

# --- Length of stay (hours) ---
base_los = {1:6.0, 2:5.0, 3:3.5, 4:2.5, 5:1.5}
los = [round(max(0.5, np.random.gamma(2, base_los[t]/2) + np.random.normal(0,0.3)), 2) for t in triage]

# --- Department ---
depts  = ['Emergency','Cardiology','Orthopedics','Neurology',
          'Pediatrics','Trauma','Obs & Gynae','General Surgery','Psychiatry']
dept_w = [0.28,0.14,0.12,0.10,0.10,0.08,0.08,0.06,0.04]
department = np.random.choice(depts, size=N, p=dept_w)

# --- Chief complaint tied to triage ---
cx = {1:['Cardiac arrest','Respiratory failure','Stroke','Septic shock','Major trauma'],
      2:['Chest pain','Severe dyspnea','Altered consciousness','Active bleeding','Anaphylaxis'],
      3:['Abdominal pain','Moderate chest pain','Fracture','Head injury','Severe infection'],
      4:['Back pain','Laceration','Urinary symptoms','Mild fever','Sprain'],
      5:['Rash','Minor cut','Cold symptoms','Prescription refill','Ear pain']}
chief_complaint = [random.choice(cx[t]) for t in triage]

# --- Demographics ---
ages = np.clip(np.concatenate([
    np.random.normal(35,12,int(N*0.35)),
    np.random.normal(62,15,int(N*0.40)),
    np.random.normal(8, 4, N-int(N*0.35)-int(N*0.40))
]), 0, 99).astype(int)
np.random.shuffle(ages)
gender    = np.random.choice(['Male','Female','Non-binary'], size=N, p=[0.48,0.50,0.02])
insurance = np.random.choice(['Medicare','Medicaid','Private','Self-pay','Government'],
                              size=N, p=[0.28,0.20,0.35,0.10,0.07])

# --- Billed amount ---
base_bill = {1:12000,2:8000,3:4500,4:2000,5:800}
dm = {'Emergency':1.0,'Cardiology':1.4,'Orthopedics':1.3,'Neurology':1.35,
      'Pediatrics':1.1,'Trauma':1.5,'Obs & Gynae':1.2,'General Surgery':1.3,'Psychiatry':0.9}
im = {'Medicare':0.85,'Medicaid':0.70,'Private':1.10,'Self-pay':0.60,'Government':0.80}
billed = [round(base_bill[triage[i]]*dm[department[i]]*im[insurance[i]]*np.random.uniform(0.7,1.4),2)
          for i in range(N)]

# --- Disposition ---
do = {1:(['Admitted','Transferred','Expired'],[0.65,0.25,0.10]),
      2:(['Admitted','Transferred','Discharged home'],[0.55,0.15,0.30]),
      3:(['Admitted','Discharged home','AMA'],[0.30,0.65,0.05]),
      4:(['Discharged home','AMA'],[0.90,0.10]),
      5:(['Discharged home','AMA'],[0.96,0.04])}
disposition = [np.random.choice(do[t][0], p=do[t][1]) for t in triage]

# --- Readmission (risk-based) ---
def readmit(i):
    p = 0.08
    if triage[i] <= 2:             p += 0.12
    if ages[i] >= 65:              p += 0.08
    if insurance[i]=='Self-pay':   p += 0.05
    if disposition[i]=='AMA':      p += 0.10
    return min(p, 0.45)
readmitted = [int(random.random() < readmit(i)) for i in range(N)]

# --- Satisfaction (penalised by wait time) ---
def satisfaction(i):
    s = 7.5 - (wait_triage[i]+wait_doctor[i])/60 + np.random.normal(0, 0.8)
    return round(float(np.clip(s, 1, 10)), 1)
sat = [satisfaction(i) for i in range(N)]

physicians = [f"Dr.{chr(65+i%26)}{chr(65+(i//26)%26)}" for i in range(40)]

df = pd.DataFrame({
    'visit_id':             [f"VT{str(i+1).zfill(6)}" for i in range(N)],
    'patient_id':           [f"PT{str(i+1).zfill(6)}" for i in range(N)],
    'arrival_datetime':     arrivals,
    'triage_level':         triage,
    'chief_complaint':      chief_complaint,
    'department':           department,
    'attending_physician':  [random.choice(physicians) for _ in range(N)],
    'patient_age':          ages,
    'patient_gender':       gender,
    'insurance_type':       insurance,
    'wait_time_triage_min': wait_triage,
    'wait_time_doctor_min': wait_doctor,
    'length_of_stay_hrs':   los,
    'disposition':          disposition,
    'readmitted_30d':       readmitted,
    'satisfaction_score':   sat,
    'billed_amount_usd':    billed,
})

df['arrival_datetime']  = pd.to_datetime(df['arrival_datetime'])
df['arrival_hour']      = df['arrival_datetime'].dt.hour
df['arrival_dayofweek'] = df['arrival_datetime'].dt.day_name()
df['arrival_month']     = df['arrival_datetime'].dt.month
df['age_group']         = pd.cut(df['patient_age'], bins=[0,17,34,49,64,99],
                                  labels=['0-17','18-34','35-49','50-64','65+'],
                                  include_lowest=True)

os.makedirs('data', exist_ok=True)
df.to_csv('data/ed_patient_data.csv', index=False)
print(f"Saved  →  data/ed_patient_data.csv")
print(f"Rows   :  {len(df):,}")
print(f"Columns:  {len(df.columns)}")
print(f"Nulls  :  {df.isnull().sum().sum()}")