import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve, classification_report, ConfusionMatrixDisplay
from sklearn.pipeline import Pipeline
import os

os.makedirs('outputs/plots',   exist_ok=True)
os.makedirs('outputs/reports', exist_ok=True)

df = pd.read_csv('data/ed_patient_data.csv')

# --- Feature engineering ---
fe = df.copy()
fe['is_elderly']       = (fe['patient_age'] >= 65).astype(int)
fe['is_self_pay']      = (fe['insurance_type'] == 'Self-pay').astype(int)
fe['is_ama']           = (fe['disposition'] == 'AMA').astype(int)
fe['is_admitted']      = (fe['disposition'] == 'Admitted').astype(int)
fe['critical_triage']  = (fe['triage_level'] <= 2).astype(int)
fe['total_wait_min']   = fe['wait_time_triage_min'] + fe['wait_time_doctor_min']
fe['low_satisfaction'] = (fe['satisfaction_score'] < 5).astype(int)
fe = pd.get_dummies(fe, columns=['insurance_type','department'], drop_first=True)

feature_cols = (
    ['triage_level','patient_age','is_elderly','is_self_pay','is_ama',
     'is_admitted','critical_triage','total_wait_min','length_of_stay_hrs',
     'satisfaction_score','low_satisfaction','billed_amount_usd'] +
    [c for c in fe.columns if c.startswith('insurance_type_') or c.startswith('department_')]
)

X = fe[feature_cols].fillna(0)
y = fe['readmitted_30d']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Train: {len(X_train)}  Test: {len(X_test)}")
print(f"Target balance: {y.value_counts(normalize=True).round(3).to_dict()}")

# --- Logistic Regression ---
lr = Pipeline([('sc', StandardScaler()), ('m', LogisticRegression(max_iter=1000, random_state=42))])
lr.fit(X_train, y_train)
lr_prob = lr.predict_proba(X_test)[:,1]
lr_pred = lr.predict(X_test)
print(f"\nLogistic Regression  AUC: {roc_auc_score(y_test, lr_prob):.4f}")
print(classification_report(y_test, lr_pred))

# --- Random Forest ---
rf = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
rf_prob = rf.predict_proba(X_test)[:,1]
rf_pred = rf.predict(X_test)
rf_auc  = roc_auc_score(y_test, rf_prob)
print(f"Random Forest  AUC: {rf_auc:.4f}")
print(classification_report(y_test, rf_pred))

# --- Feature importance ---
fi = pd.DataFrame({'feature': feature_cols, 'importance': rf.feature_importances_})\
       .sort_values('importance', ascending=False).head(15)
fi.to_csv('outputs/reports/feature_importance.csv', index=False)
print("\nTop 10 features:\n", fi.head(10).to_string(index=False))

# --- Risk scoring ---
df['risk_score'] = rf.predict_proba(X)[:,1].round(4)
df['risk_tier']  = pd.cut(df['risk_score'], bins=[0,0.10,0.20,0.35,1.0],
                           labels=['Low','Medium','High','Very High'])
df[['visit_id','patient_id','triage_level','patient_age','insurance_type',
    'disposition','risk_score','risk_tier','readmitted_30d']]\
    .to_csv('outputs/reports/readmission_risk_scores.csv', index=False)
print("\nRisk tier counts:\n", df['risk_tier'].value_counts().to_string())
print("Saved → outputs/reports/readmission_risk_scores.csv")

# ===== CHART 1: ROC curve =====
fig, ax = plt.subplots(figsize=(7, 6))
for prob, label, color, auc in [
    (lr_prob,'Logistic Regression','#378ADD',roc_auc_score(y_test,lr_prob)),
    (rf_prob,'Random Forest',       '#E24B4A', rf_auc)]:
    fpr, tpr, _ = roc_curve(y_test, prob)
    ax.plot(fpr, tpr, color=color, lw=2, label=f'{label} (AUC={auc:.3f})')
ax.plot([0,1],[0,1],'--',color='gray',lw=1,label='Baseline')
ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curve — 30-Day Readmission', fontsize=12, fontweight='bold')
ax.legend(loc='lower right'); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('outputs/plots/13_roc_curve.png', dpi=150, bbox_inches='tight')
plt.close(); print("Saved → outputs/plots/13_roc_curve.png")

# ===== CHART 2: Feature importance =====
fig, ax = plt.subplots(figsize=(9, 6))
ax.barh(fi['feature'][::-1], fi['importance'][::-1], color='#534AB7', edgecolor='none')
ax.set_title('Top 15 Feature Importances — Random Forest', fontsize=12, fontweight='bold')
ax.set_xlabel('Importance Score')
plt.tight_layout()
plt.savefig('outputs/plots/14_feature_importance.png', dpi=150, bbox_inches='tight')
plt.close(); print("Saved → outputs/plots/14_feature_importance.png")

# ===== CHART 3: Confusion matrix =====
fig, ax = plt.subplots(figsize=(5, 4))
ConfusionMatrixDisplay.from_predictions(y_test, rf_pred, ax=ax,
    display_labels=['Not Readmitted','Readmitted'], colorbar=False, cmap='Blues')
ax.set_title('Confusion Matrix — Random Forest', fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/plots/15_confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.close(); print("Saved → outputs/plots/15_confusion_matrix.png")

# ===== CHART 4: Risk score distribution by actual outcome =====
fig, ax = plt.subplots(figsize=(9, 4))
df[df['readmitted_30d']==0]['risk_score'].hist(bins=40, ax=ax, alpha=0.6, color='#378ADD', label='Not readmitted')
df[df['readmitted_30d']==1]['risk_score'].hist(bins=40, ax=ax, alpha=0.6, color='#E24B4A', label='Readmitted')
ax.set_xlabel('Predicted Risk Score'); ax.set_ylabel('Count')
ax.set_title('Risk Score Distribution by Actual Outcome', fontsize=12, fontweight='bold')
ax.legend(); plt.tight_layout()
plt.savefig('outputs/plots/16_risk_score_distribution.png', dpi=150, bbox_inches='tight')
plt.close(); print("Saved → outputs/plots/16_risk_score_distribution.png")
print("\nDay 5 complete.")