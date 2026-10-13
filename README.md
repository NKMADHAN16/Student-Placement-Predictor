# 🎓 Student Placement Predictor

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Plotly](https://img.shields.io/badge/Plotly-5.20%2B-3F4F75?logo=plotly&logoColor=white)](https://plotly.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Student Placement Predictor** is an end-to-end machine learning web application that predicts campus recruitment outcomes based on academic performance, cognitive aptitude, and technical capabilities. Built with **Streamlit** and **Scikit-learn**, it provides real-time placement probability estimates, explainable feature importance insights, and an interactive dataset exploration suite to help students evaluate and optimize their placement readiness.

---

## 📌 Table of Contents
- [Project Overview](#-project-overview)
- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [Machine Learning Pipeline](#-machine-learning-pipeline)
  - [Model Specifications](#model-specifications)
  - [Evaluation Metrics](#evaluation-metrics)
  - [Feature Importance Breakdown](#feature-importance-breakdown)
- [Evaluated Student Parameters](#-evaluated-student-parameters)
- [Repository Structure](#-repository-structure)
- [Getting Started](#-getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation & Execution](#installation--execution)
- [Using Custom College / University Data](#-using-custom-college--university-data)
- [Future Enhancements](#-future-enhancements)
- [License](#-license)

---

## 💡 Project Overview

Campus placements are a critical career milestone for engineering and college students. This system bridges data science and career readiness by:
1. Training a balanced **Random Forest Classifier** on 1,200 multi-dimensional student records.
2. Standardizing academic and non-academic parameters to remove feature bias.
3. Providing an intuitive, dark-themed web dashboard for students to test profiles and receive actionable probability feedback.

---

## ✨ Key Features

- **🔮 Interactive Prediction Dashboard**: Real-time prediction engine accepting 10 distinct student profile inputs (CGPA, skills, internships, projects, backlogs).
- **⏱️ Dynamic Probability Gauge**: Visual representation of placement probability percentage with color-coded outcome badges (Placed vs. Unlikely).
- **📈 Decision Explainability (XAI)**: Immediate bar charts showing which factors most heavily influenced the specific model verdict.
- **📊 Dataset Exploration Suite**:
  - Donut chart of overall campus placement ratios.
  - CGPA vs. Placement distribution box plots.
  - Model-wide feature weights and raw dataset table viewer.
- **🎨 Custom Modern Dark Theme**: Glassmorphism cards, glowing indicator metrics, and responsive multi-column layouts crafted with custom CSS.

---

## 🏛️ System Architecture

```text
┌───────────────────────┐       ┌───────────────────────┐
│  data/generate_data.py│ ────> │data/placement_data.csv│
│  (1,200 Synthesized   │       │(Multi-factor records) │
│   Student Profiles)   │       └───────────┬───────────┘
└───────────────────────┘                   │
                                            v
┌───────────────────────┐       ┌───────────────────────┐
│   model/train_model.py│ <──── │ StandardScaler +      │
│   (RandomForest,      │       │ Stratified Train-Test │
│    300 Estimators)    │       └───────────────────────┘
└───────────┬───────────┘
            │
            ├──> model/placement_model.pkl & scaler.pkl
            └──> model/metrics.json & features.json
                        │
                        v
        ┌───────────────────────────────┐
        │            app.py             │
        │ Streamlit Application & Web UI│
        └───────────────────────────────┘
```

---

## 🧠 Machine Learning Pipeline

### Model Specifications
- **Algorithm**: `RandomForestClassifier` (Scikit-learn)
- **Number of Estimators**: 300 Trees
- **Max Depth**: 8
- **Min Samples Leaf**: 3
- **Class Weights**: `balanced` (mitigates class imbalance)
- **Scaler**: `StandardScaler` fitted on training split

### Evaluation Metrics

Evaluated on a held-out 20% stratified test split (240 unseen student profiles):

| Metric | Score | Details |
| :--- | :---: | :--- |
| **Accuracy** | **~75.8%** | Correct overall classification rate |
| **ROC-AUC** | **~0.868** | High discriminative ability across decision thresholds |
| **F1 Score** | **~0.793** | Harmonic mean of precision and recall |
| **Training Records** | 960 | 80% stratified train partition |
| **Testing Records** | 240 | 20% stratified test partition |

### Feature Importance Breakdown

The model's decision tree ensembles assign the highest weight to applied technical aptitude and problem-solving:

| Rank | Feature | Importance | Significance |
| :---: | :--- | :---: | :--- |
| 1 | **Technical Skill** | `18.6%` | Coding fluency & technical interview rating (1–10) |
| 2 | **IQ Score** | `18.1%` | Logical reasoning & cognitive capability (70–150) |
| 3 | **Communication Skill** | `15.5%` | Interpersonal & HR interview rating (1–10) |
| 4 | **Aptitude Score** | `11.1%` | Quantitative & analytical test score (20–100) |
| 5 | **Backlogs** | `11.0%` | Penalty factor for ongoing uncleared papers |
| 6 | **CGPA** | `10.3%` | Academic consistency across semesters (4.0–10.0) |
| 7 | **Internships** | `7.1%` | Hands-on industry exposure (0–3) |
| 8 | **Projects** | `4.5%` | Deployed software/engineering projects (0–4) |
| 9 | **Certifications** | `2.9%` | Relevant technical credentials (0–4) |
| 10 | **Extra-Curricular** | `1.0%` | Active campus engagement & leadership (Yes/No) |

---

## 📋 Evaluated Student Parameters

| Feature | Variable Name | Type | Allowed Range |
| :--- | :--- | :--- | :--- |
| **CGPA** | `CGPA` | Float | 4.0 – 10.0 |
| **IQ Score** | `IQ` | Integer | 70 – 150 |
| **Aptitude Test Score** | `Aptitude_Score` | Integer | 20 – 100 |
| **Communication Rating** | `Communication_Skill` | Float | 1.0 – 10.0 |
| **Technical Rating** | `Technical_Skill` | Float | 1.0 – 10.0 |
| **Internships Completed** | `Internships` | Integer | 0, 1, 2, 3 |
| **Projects Completed** | `Projects` | Integer | 0, 1, 2, 3, 4 |
| **Active Backlogs** | `Backlogs` | Integer | 0, 1, 2, 3 |
| **Certifications** | `Certifications` | Integer | 0, 1, 2, 3, 4 |
| **Extra-Curricular Activities** | `Extra_Curricular` | Binary | Yes (`1`) / No (`0`) |

---

## 📂 Repository Structure

```
placement_predictor/
│
├── app.py                      # Streamlit application UI & interactive charts
├── requirements.txt            # Project dependencies
├── README.md                   # Full documentation & project guide
│
├── data/
│   ├── generate_data.py        # Dataset synthesizer (generates 1,200 records)
│   └── placement_data.csv      # Generated dataset with 10 features + target
│
└── model/
    ├── train_model.py          # Trains Random Forest and exports artifacts
    ├── placement_model.pkl     # Trained RandomForestClassifier model
    ├── scaler.pkl              # Fitted StandardScaler instance
    ├── features.json           # Ordered feature column names
    └── metrics.json            # Model performance metrics & feature importances
```

---

## 🚀 Getting Started

### Prerequisites
- Python **3.10+** installed
- Git installed

### Installation & Execution

1. **Clone the repository:**
   ```bash
   git clone https://github.com/NKMADHAN16/Student-Placement-Predictor.git
   cd Student-Placement-Predictor
   ```

2. **Set up a virtual environment:**
   - **Windows (PowerShell):**
     ```powershell
     python -m venv venv
     .\venv\Scripts\activate
     ```
   - **macOS / Linux:**
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **(Optional) Regenerate dataset and retrain model:**
   ```bash
   python data/generate_data.py
   python model/train_model.py
   ```

5. **Start the Streamlit application:**
   ```bash
   streamlit run app.py
   ```
   Open **`http://localhost:8501`** in your browser.

---

## 🔄 Using Custom College / University Data

To train this model on your own institution's placement history:

1. Prepare your dataset in CSV format with the same 10 column names plus a target column `Placement_Status` (`1` for Placed, `0` for Not Placed).
2. Replace `data/placement_data.csv` with your data file.
3. Run the training script:
   ```bash
   python model/train_model.py
   ```
4. Rerun `streamlit run app.py` — the application will automatically load your new model and updated performance metrics.

---

## 🔮 Future Enhancements

- [ ] **Multi-Model Comparison**: Add XGBoost, LightGBM, and Logistic Regression with side-by-side ROC curves.
- [ ] **Resume Scanner**: Automatic skill extraction from uploaded PDF resumes using NLP.
- [ ] **Student Profile History**: Persistent SQLite/PostgreSQL storage for saving semester-by-semester progress.
- [ ] **Cloud Deployment**: One-click deployment to Streamlit Community Cloud and Hugging Face Spaces.

---

## 📄 License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT). Feel free to use, modify, and distribute for academic and personal projects.
