import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load Data
print("Loading data...")
df = pd.read_csv('c:/Task_02/Churn_Modelling.csv')

# 2. Preprocessing
print("Preprocessing data...")
# Drop irrelevant columns
df = df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)

# Encode categorical variables
le = LabelEncoder()
df['Gender'] = le.fit_transform(df['Gender'])
# Save label encoder for Gender if needed, but for now we'll just remember 0=Female, 1=Male usually (check later)

# One-hot encode Geography
df = pd.get_dummies(df, columns=['Geography'], drop_first=True)

# Split features and target
X = df.drop('Exited', axis=1)
y = df['Exited']

# Split into Train and Test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save the scaler
joblib.dump(scaler, 'c:/Task_02/scaler.pkl')
print("Scaler saved.")

# 3. Model Training and Evaluation
models = {
    "Logistic Regression": LogisticRegression(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
}

best_model = None
best_auc = 0

print("\nTraining and Evaluating Models:")
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    
    print(f"\n{name}:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  ROC-AUC: {auc:.4f}")
    
    if auc > best_auc:
        best_auc = auc
        best_model = model

# 4. Save the best model
print(f"\nBest Model: {type(best_model).__name__} with AUC: {best_auc:.4f}")
joblib.dump(best_model, 'c:/Task_02/best_model.pkl')
print("Best model saved to c:/Task_02/best_model.pkl")

# 5. Save Feature Importance (if applicable)
if hasattr(best_model, 'feature_importances_'):
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': best_model.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    feature_importance.to_csv('c:/Task_02/feature_importance.csv', index=False)
    print("Feature importance saved.")

# 6. Save Test Predictions for App Demo
# We'll save a subset of the test data with predictions
test_df = X_test.copy()
test_df['Actual_Exited'] = y_test
test_df['Predicted_Exited'] = best_model.predict(X_test_scaled)
test_df['Churn_Probability'] = best_model.predict_proba(X_test_scaled)[:, 1]

test_df.to_csv('c:/Task_02/test_predictions.csv', index=False)
print("Test predictions saved to c:/Task_02/test_predictions.csv")
