"""
Trains a RandomForest classifier on the placement dataset and saves
the model + scaler + metrics for the Streamlit app.
Run: python model/train_model.py
"""
import pickle
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix

df = pd.read_csv("data/placement_data.csv")

FEATURES = [
    "CGPA", "IQ", "Aptitude_Score", "Communication_Skill",
    "Internships", "Projects", "Technical_Skill", "Backlogs",
    "Extra_Curricular", "Certifications",
]
TARGET = "Placement_Status"

X = df[FEATURES]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=8,
    min_samples_leaf=3,
    random_state=42,
    class_weight="balanced",
)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
y_proba = model.predict_proba(X_test_scaled)[:, 1]

metrics = {
    "accuracy": round(accuracy_score(y_test, y_pred), 4),
    "f1_score": round(f1_score(y_test, y_pred), 4),
    "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
    "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    "feature_importance": dict(zip(FEATURES, np.round(model.feature_importances_, 4).tolist())),
    "n_train": len(X_train),
    "n_test": len(X_test),
}

with open("model/placement_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("model/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

with open("model/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

with open("model/features.json", "w") as f:
    json.dump(FEATURES, f)

print("Training complete.")
print(json.dumps(metrics, indent=2))
