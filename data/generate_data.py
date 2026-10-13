"""
Generates a synthetic but realistic student placement dataset.
Run: python data/generate_data.py
"""
import numpy as np
import pandas as pd

np.random.seed(42)
N = 1200

cgpa = np.clip(np.random.normal(7.2, 1.0, N), 4.0, 10.0)
iq = np.clip(np.random.normal(100, 15, N), 70, 150)
aptitude_score = np.clip(np.random.normal(65, 15, N), 20, 100)
communication_skill = np.clip(np.random.normal(6.5, 1.8, N), 1, 10)
internships = np.random.choice([0, 1, 2, 3], N, p=[0.35, 0.35, 0.2, 0.1])
projects = np.random.choice([0, 1, 2, 3, 4], N, p=[0.1, 0.25, 0.3, 0.2, 0.15])
technical_skill = np.clip(np.random.normal(6.5, 1.7, N), 1, 10)
backlogs = np.random.choice([0, 1, 2, 3], N, p=[0.6, 0.25, 0.1, 0.05])
extra_curricular = np.random.choice([0, 1], N, p=[0.5, 0.5])
certifications = np.random.choice([0, 1, 2, 3, 4], N, p=[0.2, 0.3, 0.25, 0.15, 0.1])

# Latent "placement score" combining weighted factors + noise
score = (
    cgpa * 1.1
    + (iq - 70) / 8
    + aptitude_score / 12
    + communication_skill * 0.9
    + internships * 1.3
    + projects * 0.8
    + technical_skill * 1.0
    + certifications * 0.5
    + extra_curricular * 0.6
    - backlogs * 1.8
    + np.random.normal(0, 2.5, N)
)

threshold = np.percentile(score, 42)  # roughly 58% placement rate
placed = (score > threshold).astype(int)

df = pd.DataFrame({
    "CGPA": np.round(cgpa, 2),
    "IQ": np.round(iq).astype(int),
    "Aptitude_Score": np.round(aptitude_score).astype(int),
    "Communication_Skill": np.round(communication_skill, 1),
    "Internships": internships,
    "Projects": projects,
    "Technical_Skill": np.round(technical_skill, 1),
    "Backlogs": backlogs,
    "Extra_Curricular": extra_curricular,
    "Certifications": certifications,
    "Placement_Status": placed,
})

df.to_csv("data/placement_data.csv", index=False)
print(f"Generated {len(df)} records -> data/placement_data.csv")
print(df["Placement_Status"].value_counts(normalize=True))
