# 🫀 Heart Disease Predictor

## 📌 Overview
This project predicts the likelihood of heart disease in patients based on clinical parameters such as age, sex, cholesterol levels, blood pressure, and more.  
It uses **Machine Learning** (Scikit-learn) models trained on a public heart disease dataset.

---

## 📂 Project Structure
```
heart-disease-predictor/
│
├── heart-disease-classification.ipynb   # Jupyter Notebook with full analysis and modeling
├── heart-disease (2).csv                # Dataset
├── heart-disease.joblib                  # Saved trained model (Joblib format)
├── heart-diseasepkl.pkl                  # Saved trained model (Pickle format)
└── README.md                             # Project documentation
```

---

## 📊 Dataset
The dataset contains patient health attributes including:
- Age
- Sex (1 = male, 0 = female)
- Chest pain type
- Resting blood pressure
- Serum cholesterol
- Fasting blood sugar
- Resting electrocardiographic results
- Maximum heart rate achieved
- Exercise-induced angina
- ST depression induced by exercise
- Slope of peak exercise ST segment
- Number of major vessels colored by fluoroscopy
- Thalassemia test result

📌 **Source:** [UCI Machine Learning Repository - Heart Disease Dataset](https://archive.ics.uci.edu/ml/datasets/Heart+Disease)

---

## ⚙️ Installation
1. **Clone the Repository**
```bash
git clone https://github.com/tejaschavan724/heart_disease_project.git
cd heart_disease_project/heart-disease-predictor
```

2. **Create a Virtual Environment**
```bash
python -m venv venv
source venv/bin/activate   # On Mac/Linux
venv\Scripts\activate      # On Windows
```

3. **Install Dependencies**
```bash
pip install -r requirements.txt
```
*(If `requirements.txt` not available, manually install: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `joblib`)*

---

## 🚀 Usage
### Run Jupyter Notebook
```bash
jupyter notebook
```
Open **`heart-disease-classification.ipynb`** to explore the analysis and run predictions.

### Load Model and Predict in Python
```python
import joblib
import pandas as pd

# Load model
model = joblib.load("heart-disease.joblib")

# Example patient data
patient = pd.DataFrame([{
    "age": 52,
    "sex": 1,
    "cp": 0,
    "trestbps": 125,
    "chol": 212,
    "fbs": 0,
    "restecg": 1,
    "thalach": 168,
    "exang": 0,
    "oldpeak": 1.0,
    "slope": 2,
    "ca": 2,
    "thal": 3
}])

# Predict
prediction = model.predict(patient)
print("Heart Disease" if prediction[0] == 1 else "No Heart Disease")
```

---

## 📈 Model Performance
The notebook includes:
- **Exploratory Data Analysis (EDA)**
- **Feature Engineering**
- **Model Training & Comparison**
- **Accuracy, Precision, Recall, F1-score**
- **Confusion Matrix & ROC Curve**

---

## 📌 Future Improvements
- Deploy as a **Streamlit web app**
- Add **explainability** using SHAP
- Perform **hyperparameter tuning** for better accuracy

---

## ⚠️ Disclaimer
This model is for **educational purposes only** and **not for real medical diagnosis**.

---

## 📜 License
MIT License © 2025 [Tejas Chavan](https://github.com/tejaschavan724)
