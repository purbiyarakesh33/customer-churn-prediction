#  Customer Churn Prediction

An end-to-end Machine Learning project to predict whether a telecom customer will churn (leave the service), built with Python, Scikit-learn, XGBoost, TensorFlow, and deployed as an interactive Streamlit web app.

---

##  Problem Statement

Customer churn is one of the biggest challenges in the telecom industry. Acquiring a new customer costs **5x more** than retaining an existing one. This project builds a machine learning system that predicts which customers are likely to leave — enabling the company to take proactive retention actions.

---

##  Dataset

- **Source:** IBM Telco Customer Churn Dataset (Kaggle)
- **Size:** 7,043 customers, 20 features
- **Target:** Churn (Yes/No)
- **Class Distribution:** 73% No Churn, 27% Churn (imbalanced)

---

##  Key Insights from EDA

- Customers with **Month-to-month contracts** churn the most
- **New customers** (tenure < 5 months) have the highest churn rate
- **Fiber optic** internet users churn more than DSL users
- Customers with **higher monthly charges** are more likely to churn
- **Two-year contract** customers almost never churn

---

## ⚙️ Project Pipeline

```
Raw Data
    ↓
Data Cleaning (TotalCharges fix, duplicates)
    ↓
EDA & Visualization
    ↓
Correlation Analysis & Feature Selection
    ↓
Encoding (Label + One Hot)
    ↓
Train / Validation / Test Split (60/20/20)
    ↓
Feature Scaling (StandardScaler)
    ↓
Model Building & Hyperparameter Tuning
    ↓
Bias/Variance Analysis
    ↓
Threshold Tuning
    ↓
Final Test Evaluation
    ↓
Streamlit Deployment
```

---

## Models Built

| Model | Val F1 | Threshold | Status |
|---|---|---|---|
| Logistic Regression | 64.28% | 0.52 | High Bias |
| Random Forest | 64.41% | 0.48 | High Variance |
| **XGBoost** | **64.72%** | **0.67** | **Best Model**  |
| Neural Network | 64.43% | 0.69 | High Bias |

---

##  Best Model — XGBoost

### Hyperparameters
```
n_estimators:     300
max_depth:        5
learning_rate:    0.01
min_child_weight: 5
scale_pos_weight: 2.77 (handles class imbalance)
```

### Final Test Set Results
| Metric | Score |
|---|---|
| Accuracy | 81.12% |
| Precision | 64.89% |
| Recall | 64.55% |
| **F1 Score** | **64.72%** |
| ROC-AUC | 85.43% |

---

##  ML Techniques Applied

- **Bias/Variance Analysis** — diagnosed underfitting/overfitting for every model
- **Threshold Tuning** — optimized decision threshold for best F1
- **GridSearchCV** — systematic hyperparameter tuning
- **scale_pos_weight** — handled class imbalance in XGBoost
- **class_weight='balanced'** — handled class imbalance in Random Forest
- **Early Stopping** — prevented overfitting in Neural Network
- **Correlation Analysis** — removed redundant features (TotalCharges)

---

##  Project Structure

```
customer-churn-prediction/
│
├── app.py                          ← Streamlit web app
├── 3.py                            ← Model training script
├── README.md                       ← Project documentation
├── requirements.txt                ← Dependencies
│
├── plots/
│   ├── churn_distribution.png
│   ├── tenure_vs_churn.png
│   ├── monthly_charges_vs_churn.png
│   ├── contract_vs_churn.png
│   ├── correlation_heatmap.png
│   ├── xgb_confusion_matrix.png
│   ├── rf_feature_importance.png
│   └── nn_training_curves.png
```

---

##  How To Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/rakeshpurbiya/customer-churn-prediction.git
cd customer-churn-prediction
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Train the model
```bash
python 3.py
```

### 4. Run the Streamlit app
```bash
streamlit run app.py
```

---

## Requirements

```
pandas
numpy
scikit-learn
xgboost
tensorflow
streamlit
matplotlib
seaborn
pickle
```

---

##  Live Demo

 [Click here to try the app](https://your-streamlit-url.streamlit.app)

---

##  Future Improvements

- Implement **SMOTE** for better handling of class imbalance
- Add **SHAP values** for model explainability
- Collect more features like customer complaints and call center history
- Build a **REST API** using FastAPI for production deployment

---

##  Author

**Rakesh Purbiya**
- B.Tech Mechanical Engineering — SVNIT Surat
- Connect on [LinkedIn](https://linkedin.com/in/rakeshpurbiya)
-  More projects on [GitHub](https://github.com/rakeshpurbiya)

---

