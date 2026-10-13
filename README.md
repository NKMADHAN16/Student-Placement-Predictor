# 🎓 Student Placement Prediction System

A machine learning web app that predicts whether a student is likely to be
placed in campus recruitment, based on academics, aptitude, and skills —
built with **Streamlit** and a **Random Forest** classifier.

## Features
- Interactive prediction form (CGPA, IQ, aptitude, communication, internships,
  projects, technical skill, backlogs, certifications, extracurriculars)
- Animated probability gauge and color-coded result card
- Feature-importance chart explaining what drives the prediction
- Dataset insights tab: placement distribution, CGPA vs placement boxplot,
  raw data browser
- Clean, dark, custom-styled UI (not default Streamlit theme)

## Project Structure
```
placement_predictor/
├── app.py                    # Streamlit application
├── requirements.txt
├── data/
│   ├── generate_data.py      # Synthetic dataset generator
│   └── placement_data.csv    # Generated dataset (1200 records)
└── model/
    ├── train_model.py        # Trains & saves the model
    ├── placement_model.pkl   # Trained RandomForest model
    ├── scaler.pkl            # Feature scaler
    ├── features.json         # Feature list used by the model
    └── metrics.json          # Accuracy / F1 / ROC-AUC / feature importance
```

## Setup

```bash
# 1. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) Regenerate data and retrain the model
python data/generate_data.py
python model/train_model.py

# 4. Run the app
streamlit run app.py
```

Then open the URL Streamlit prints (usually `http://localhost:8501`).

## Model
- **Algorithm**: Random Forest Classifier (300 trees, max depth 8, balanced class weights)
- **Accuracy**: ~76% | **ROC-AUC**: ~0.87 | **F1 Score**: ~0.79
- Trained on a synthetic dataset of 1,200 student records designed to mimic
  realistic placement patterns (CGPA, IQ, aptitude score, communication,
  technical skill, internships, projects, backlogs, certifications, extracurriculars).

## Using Your Own Data
Replace `data/placement_data.csv` with your own CSV (same column names/target
`Placement_Status`), then rerun `python model/train_model.py` to retrain on
real data — the app will automatically use the new model.

## Next Steps / Ideas to Extend
- Swap in a real placement dataset from your institution
- Add login + student history tracking (e.g., with SQLite)
- Try other models (XGBoost, Logistic Regression) and compare via the metrics tab
- Deploy for free on [Streamlit Community Cloud](https://streamlit.io/cloud)
