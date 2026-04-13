<<<<<<< HEAD
# IntellgientProg
=======
# 🫀 Heart Disease Detection System

A simple, well-structured project that detects heart disease risk using:
- A **Rule-Based Expert System** (Experta — 10 clinical rules)
- A **Decision Tree Classifier** (Scikit-Learn — with hyperparameter tuning)
- A **Streamlit Web UI** for interactive predictions

---

## 📁 Project Structure

```
Heart_Disease_Detection/
├── data/
│   ├── raw_data.csv          ← original dataset
│   └── cleaned_data.csv      ← generated after running main.py
├── rule_based_system/
│   └── expert_system.py      ← 10-rule Experta engine
├── ml_model/
│   ├── train_model.py        ← standalone training script
│   ├── predict.py            ← standalone prediction script
│   └── decision_tree_model.pkl  ← generated after training
├── utils/
│   └── data_processing.py    ← cleaning & normalisation helpers
├── reports/
│   ├── accuracy_comparison.md     ← generated after main.py
│   ├── correlation_heatmap.png
│   ├── feature_importance.png
│   ├── feature_histograms.png
│   └── target_distribution.png
├── ui/
│   └── app.py                ← Streamlit UI
├── main.py                   ← run everything end-to-end
└── requirements.txt
```

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the full pipeline
```bash
python main.py
```
This will:
- Clean and preprocess the dataset
- Generate all visualisation plots
- Run the expert system on a sample patient
- Train + evaluate the Decision Tree
- Write the comparison report

### 3. Launch the web UI
```bash
streamlit run ui/app.py
```
Open http://localhost:8501 in your browser.

---

## 🗂️ Dataset Features

| Column   | Description                          |
|----------|--------------------------------------|
| age      | Patient age in years                 |
| sex      | 1 = Male, 0 = Female                 |
| cp       | Chest pain type (0–3)                |
| trestbps | Resting blood pressure (mmHg)        |
| chol     | Serum cholesterol (mg/dL)            |
| fbs      | Fasting blood sugar > 120 mg/dL      |
| restecg  | Resting ECG results (0/1/2)          |
| thalach  | Maximum heart rate achieved          |
| exang    | Exercise induced angina (0/1)        |
| oldpeak  | ST depression induced by exercise    |
| slope    | Slope of peak exercise ST segment    |
| ca       | Number of major vessels (0–3)        |
| thal     | Thalassemia type (1/2/3)             |
| target   | 1 = Heart disease, 0 = No disease    |

---

## 📋 Expert System Rules (10 rules)

1. High cholesterol (>240) AND age >50
2. Resting BP >140 mmHg
3. Chest pain type 1, 2, or 3
4. Exercise-induced angina
5. ST depression >2.0
6. Fasting blood sugar >120 mg/dL
7. 2+ major vessels blocked
8. Thalassemia defect (type 2 or 3)
9. Max heart rate <100 bpm
10. Age >60 + cholesterol >200 + BP >130 (compound rule)

**Risk scoring:** 0 rules = LOW, 1–3 = MODERATE, 4+ = HIGH
>>>>>>> 139ba1e (Initial commit)
