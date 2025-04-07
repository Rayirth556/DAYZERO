import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from category_encoders import TargetEncoder
import xgboost as xgb

# === Step 1: Load and Preprocess Data ===
data = pd.read_csv('bank_data.csv')
for col in ['Income', 'Credit Score', 'Credit History Length', 'Outstanding Loans', 'Balance', 
            'NumOfProducts', 'NumComplaints', 'Customer Tenure']:
    data[col] = data[col].fillna(data[col].median())

data['Date of Birth'] = pd.to_datetime(data['Date of Birth'])
data['Age'] = ((datetime(2025, 4, 7) - data['Date of Birth']).dt.days / 365.25).astype(int)

y = data['Churn Flag']
X = data.drop(columns=['Churn Flag'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

X_test_original = X_test[['Gender', 'Marital Status', 'Occupation']].copy()


# Encode categorical features
drop_cols = ['RowNumber', 'CustomerId', 'Surname', 'First Name', 'Address', 'Contact Information', 
             'Date of Birth', 'Churn Date', 'Churn Reason']
cat_cols = ['Gender', 'Marital Status', 'Customer Segment', 'Preferred Communication Channel']
X_train = pd.get_dummies(X_train, columns=cat_cols, drop_first=True)
X_test = pd.get_dummies(X_test, columns=cat_cols, drop_first=True)
X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)

encoder = TargetEncoder()
for col in ['Occupation', 'Education Level']:
    X_train[col] = encoder.fit_transform(X_train[col], y_train)
    X_test[col] = encoder.transform(X_test[col])

X_train.drop(columns=[col for col in drop_cols if col in X_train.columns], inplace=True)
X_test.drop(columns=[col for col in drop_cols if col in X_test.columns], inplace=True)

scaler = StandardScaler()
numerical_cols = ['Income', 'Credit Score', 'Credit History Length', 'Outstanding Loans', 'Balance',
                  'NumOfProducts', 'NumComplaints', 'Customer Tenure', 'Age']
X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

# Add engineered features
X_train['Debt_to_Income'] = X_train['Outstanding Loans'] / X_train['Income'].replace(0, 1e-6)
X_test['Debt_to_Income'] = X_test['Outstanding Loans'] / X_test['Income'].replace(0, 1e-6)
X_train['Complaint_Rate'] = X_train['NumComplaints'] / X_train['Customer Tenure'].replace(0, 1e-6)
X_test['Complaint_Rate'] = X_test['NumComplaints'] / X_test['Customer Tenure'].replace(0, 1e-6)
X_train['Engagement_Score'] = X_train['NumOfProducts'] * X_train['Balance']
X_test['Engagement_Score'] = X_test['NumOfProducts'] * X_test['Balance']

# === Step 2: Train Model ===
model = xgb.XGBClassifier(
    objective='binary:logistic', eval_metric='auc', max_depth=6,
    learning_rate=0.1, n_estimators=100, subsample=0.8, colsample_bytree=0.8,
    scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(), random_state=42
)
model.fit(X_train, y_train)

# === Step 3: Predictions and Metrics ===
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

print("\n=== Model Performance Metrics ===")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_pred_proba))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec

# --- Visualization Dashboard ---
# Data prep
plot_data = pd.concat([
    X_test[numerical_cols].reset_index(drop=True),
    y_test.reset_index(drop=True)
], axis=1).melt(id_vars='Churn Flag', var_name='Feature', value_name='Value')

pred_data = pd.DataFrame({
    'Predicted Probability': y_pred_proba,
    'Churn Flag': y_test.reset_index(drop=True)
})

feature_importance = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': model.feature_importances_
}).nlargest(10, 'Importance')

# Prepare top 10 occupations by highest churn rate
cat_pred_data = pd.concat([
    X_test_original.reset_index(drop=True),
    pd.DataFrame({'Predicted Churn': y_pred})
], axis=1)

occupation_churn = cat_pred_data.groupby('Occupation')['Predicted Churn'].mean().reset_index()
top_churn_occupations = occupation_churn.sort_values(by='Predicted Churn', ascending=False).head(10)

# Plot layout
fig = plt.figure(figsize=(18, 12), constrained_layout=True)
spec = gridspec.GridSpec(ncols=2, nrows=2, figure=fig)

# Plot 1: Predicted Probabilities KDE
ax1 = fig.add_subplot(spec[0, 0])
sns.kdeplot(
    data=pred_data, x='Predicted Probability', hue='Churn Flag',
    fill=True, common_norm=False, palette='Set2', ax=ax1
)
ax1.set_title('Predicted Probability Distribution by Churn')
ax1.axvline(0.5, color='gray', linestyle='--')

# Plot 2: Feature Importances
ax2 = fig.add_subplot(spec[0, 1])
sns.barplot(x='Importance', y='Feature', data=feature_importance, palette='viridis', ax=ax2)
ax2.set_title('Top 10 Feature Importances')

# Plot 3: Boxplot of Key Numerical Features
ax3 = fig.add_subplot(spec[1, 0])
sns.boxplot(x='Feature', y='Value', hue='Churn Flag', data=plot_data, palette='Set3', ax=ax3)
ax3.set_title('Numerical Feature Distributions by Churn')
ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45)
ax3.set_ylim(-3, 3)

# Plot 4: Highest Churn Rate Occupations
ax4 = fig.add_subplot(spec[1, 1])
sns.barplot(
    x='Occupation', y='Predicted Churn', data=top_churn_occupations,
    palette='Set1', ax=ax4
)
ax4.set_title('Top 10 Occupations by Churn Rate')
ax4.set_ylim(0, 1)
ax4.set_xticklabels(ax4.get_xticklabels(), rotation=45, ha='right')

plt.suptitle('Churn Prediction Dashboard', fontsize=18, y=1.02)
plt.show()
