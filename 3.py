import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, roc_auc_score,
                             roc_curve)
from xgboost import XGBClassifier
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

# ─────────────────────────────────────────────
# 1. LOAD AND PREPROCESS RAW DATA
# ─────────────────────────────────────────────
print("=" * 60)
print("LOADING AND PREPROCESSING DATA")
print("=" * 60)

df = pd.read_csv("Telco_Cusomer_Churn.csv")
print(f"Data loaded: {df.shape}")

# Fix TotalCharges
df['TotalCharges'] = df['TotalCharges'].replace(' ', np.nan)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'])
df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())

# Drop customerID
df = df.drop('customerID', axis=1)

# Label Encoding
binary_cols = ['gender', 'Partner', 'Dependents',
               'PhoneService', 'PaperlessBilling', 'Churn']
le = LabelEncoder()
for col in binary_cols:
    df[col] = le.fit_transform(df[col])

# One Hot Encoding
multi_cols = ['MultipleLines', 'InternetService', 'OnlineSecurity',
              'OnlineBackup', 'DeviceProtection', 'TechSupport',
              'StreamingTV', 'StreamingMovies', 'Contract', 'PaymentMethod']
df = pd.get_dummies(df, columns=multi_cols, drop_first=True)

# Convert bool to int
df = df.astype(int)

# Drop TotalCharges (correlated with MonthlyCharges)
df = df.drop('TotalCharges', axis=1)

print(f"Encoding done: {df.shape}")
print(f"Churn rate: {df['Churn'].mean():.2%}\n")

# ─────────────────────────────────────────────
# 2. TRAIN / VAL / TEST SPLIT (60 / 20 / 20)
# ─────────────────────────────────────────────
X = df.drop('Churn', axis=1)
y = df['Churn']

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val,   X_test, y_val,   y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# ─────────────────────────────────────────────
# 3. SCALING AFTER SPLIT (no leakage)
# ─────────────────────────────────────────────
scale_cols = ['tenure', 'MonthlyCharges']
scaler = StandardScaler()
X_train[scale_cols] = scaler.fit_transform(X_train[scale_cols])
X_val[scale_cols]   = scaler.transform(X_val[scale_cols])
X_test[scale_cols]  = scaler.transform(X_test[scale_cols])

print(f"Train:      {X_train.shape}  |  churn rate: {y_train.mean():.2%}")
print(f"Validation: {X_val.shape}  |  churn rate: {y_val.mean():.2%}")
print(f"Test:       {X_test.shape}  |  churn rate: {y_test.mean():.2%}")
print(f"Scaler fitted on: {scaler.feature_names_in_}")

# Class imbalance ratio (used by XGBoost)
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
print(f"scale_pos_weight: {scale_pos_weight:.2f}\n")

# ─────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────
def evaluate(name, y_true, y_pred, y_prob=None):
    print(f"\n{'─'*40}")
    print(f"  {name}")
    print(f"{'─'*40}")
    print(f"  Accuracy : {accuracy_score(y_true, y_pred):.4f}")
    print(f"  Precision: {precision_score(y_true, y_pred):.4f}")
    print(f"  Recall   : {recall_score(y_true, y_pred):.4f}")
    print(f"  F1 Score : {f1_score(y_true, y_pred):.4f}")
    if y_prob is not None:
        print(f"  ROC-AUC  : {roc_auc_score(y_true, y_prob):.4f}")

def bias_variance_check(name, train_f1, val_f1):
    gap = train_f1 - val_f1
    print(f"\n  Bias/Variance — Train F1: {train_f1:.4f}  |  Val F1: {val_f1:.4f}  |  Gap: {gap:.4f}")
    if train_f1 < 0.75 and gap > 0.05:
        print("  → High Bias + High Variance!")
    elif train_f1 < 0.75:
        print("  → High Bias (Underfitting)")
    elif gap > 0.05:
        print("  → High Variance (Overfitting)")
    else:
        print("  → Good Fit ✓")

def tune_threshold(y_true, y_prob):
    best_f1, best_t = 0, 0.5
    for t in np.arange(0.1, 0.9, 0.01):
        pred = (y_prob >= t).astype(int)
        f1 = f1_score(y_true, pred)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t, best_f1

def save_confusion_matrix(y_true, y_pred, title, filename):
    plt.figure(figsize=(5, 4))
    sns.heatmap(confusion_matrix(y_true, y_pred),
                annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Churn', 'Churn'],
                yticklabels=['No Churn', 'Churn'])
    plt.title(title)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"  Confusion matrix saved → {filename}")

# # ─────────────────────────────────────────────
# # MODEL 1 — LOGISTIC REGRESSION
# # ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("MODEL 1: LOGISTIC REGRESSION")
print("=" * 60)

print("\nTuning C value...")
C_values = [0.001, 0.01, 0.1, 1, 10, 100]
best_lr_f1, best_lr_c = 0, 1

for c in C_values:
    m = LogisticRegression(C=c, random_state=42, max_iter=1000)
    m.fit(X_train, y_train)
    vf1 = f1_score(y_val, m.predict(X_val))
    print(f"  C={c:<8}  Val F1: {vf1:.4f}")
    if vf1 > best_lr_f1:
        best_lr_f1, best_lr_c = vf1, c

print(f"\nBest C: {best_lr_c}  |  Best Val F1: {best_lr_f1:.4f}")

lr = LogisticRegression(C=best_lr_c, random_state=42, max_iter=1000)
lr.fit(X_train, y_train)

lr_prob_val  = lr.predict_proba(X_val)[:, 1]
lr_thresh, _ = tune_threshold(y_val, lr_prob_val)
lr_pred_val  = (lr_prob_val >= lr_thresh).astype(int)

evaluate("Logistic Regression — Validation", y_val, lr_pred_val, lr_prob_val)
print(f"  Best threshold: {lr_thresh:.2f}")

train_f1_lr = f1_score(y_train, (lr.predict_proba(X_train)[:, 1] >= lr_thresh).astype(int))
bias_variance_check("LR", train_f1_lr, f1_score(y_val, lr_pred_val))
save_confusion_matrix(y_val, lr_pred_val, "Logistic Regression — Confusion Matrix", "lr_confusion_matrix.png")

# ─────────────────────────────────────────────
# MODEL 2 — RANDOM FOREST
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("MODEL 2: RANDOM FOREST")
print("=" * 60)

print("\nRunning GridSearchCV...")
rf_param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [7, 10, 15, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

rf_grid = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42, class_weight='balanced'),
    param_grid=rf_param_grid,
    scoring='f1',
    cv=5,
    verbose=0,
    n_jobs=-1
)
rf_grid.fit(X_train, y_train)

print(f"Best params: {rf_grid.best_params_}")
print(f"Best CV F1 : {rf_grid.best_score_:.4f}")

best_rf = rf_grid.best_estimator_

rf_prob_val  = best_rf.predict_proba(X_val)[:, 1]
rf_thresh, _ = tune_threshold(y_val, rf_prob_val)
rf_pred_val  = (rf_prob_val >= rf_thresh).astype(int)

evaluate("Random Forest — Validation", y_val, rf_pred_val, rf_prob_val)
print(f"  Best threshold: {rf_thresh:.2f}")

train_f1_rf = f1_score(y_train, (best_rf.predict_proba(X_train)[:, 1] >= rf_thresh).astype(int))
bias_variance_check("RF", train_f1_rf, f1_score(y_val, rf_pred_val))
save_confusion_matrix(y_val, rf_pred_val, "Random Forest — Confusion Matrix", "rf_confusion_matrix.png")

feat_imp = pd.Series(best_rf.feature_importances_, index=X_train.columns)
top15 = feat_imp.nlargest(15)
plt.figure(figsize=(8, 5))
top15.sort_values().plot(kind='barh', color='steelblue')
plt.title("Random Forest — Top 15 Feature Importances")
plt.tight_layout()
plt.savefig("rf_feature_importance.png")
plt.close()
print("  Feature importance saved → rf_feature_importance.png")

# ─────────────────────────────────────────────
# MODEL 3 — XGBOOST
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("MODEL 3: XGBOOST")
print("=" * 60)

print("\nRunning GridSearchCV...")
xgb_param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'min_child_weight': [1, 3, 5]
}

xgb_grid = GridSearchCV(
    estimator=XGBClassifier(
        random_state=42,
        scale_pos_weight=scale_pos_weight,
        eval_metric='logloss',
        verbosity=0
    ),
    param_grid=xgb_param_grid,
    scoring='f1',
    cv=5,
    verbose=0,
    n_jobs=-1
)
xgb_grid.fit(X_train, y_train)

print(f"Best params: {xgb_grid.best_params_}")
print(f"Best CV F1 : {xgb_grid.best_score_:.4f}")

best_xgb = xgb_grid.best_estimator_

xgb_prob_val  = best_xgb.predict_proba(X_val)[:, 1]
xgb_thresh, _ = tune_threshold(y_val, xgb_prob_val)
xgb_pred_val  = (xgb_prob_val >= xgb_thresh).astype(int)

evaluate("XGBoost — Validation", y_val, xgb_pred_val, xgb_prob_val)
print(f"  Best threshold: {xgb_thresh:.2f}")

train_f1_xgb = f1_score(y_train, (best_xgb.predict_proba(X_train)[:, 1] >= xgb_thresh).astype(int))
bias_variance_check("XGB", train_f1_xgb, f1_score(y_val, xgb_pred_val))
save_confusion_matrix(y_val, xgb_pred_val, "XGBoost — Confusion Matrix", "xgb_confusion_matrix.png")

# ─────────────────────────────────────────────
# MODEL 4 — NEURAL NETWORK
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("MODEL 4: NEURAL NETWORK")
print("=" * 60)

n_features = X_train.shape[1]

def build_nn(dropout_rate=0.3, learning_rate=0.001):
    model = Sequential([
        Dense(128, activation='relu', input_shape=(n_features,)),
        BatchNormalization(),
        Dropout(dropout_rate),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(dropout_rate),
        Dense(32, activation='relu'),
        Dropout(dropout_rate / 2),
        Dense(1, activation='sigmoid')
    ])
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

class_weight = {0: 1.0, 1: scale_pos_weight}

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=0
)

print("\nTraining neural network...")
nn_model = build_nn(dropout_rate=0.3, learning_rate=0.001)

history = nn_model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=32,
    class_weight=class_weight,
    callbacks=[early_stop],
    verbose=1
)

print(f"\nTraining stopped at epoch {len(history.history['loss'])}")

nn_prob_val  = nn_model.predict(X_val, verbose=0).flatten()
nn_thresh, _ = tune_threshold(y_val, nn_prob_val)
nn_pred_val  = (nn_prob_val >= nn_thresh).astype(int)

evaluate("Neural Network — Validation", y_val, nn_pred_val, nn_prob_val)
print(f"  Best threshold: {nn_thresh:.2f}")

nn_prob_train = nn_model.predict(X_train, verbose=0).flatten()
train_f1_nn   = f1_score(y_train, (nn_prob_train >= nn_thresh).astype(int))
bias_variance_check("NN", train_f1_nn, f1_score(y_val, nn_pred_val))
save_confusion_matrix(y_val, nn_pred_val, "Neural Network — Confusion Matrix", "nn_confusion_matrix.png")

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Neural Network — Loss Curves')
plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Neural Network — Accuracy Curves')
plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend()

plt.tight_layout()
plt.savefig("nn_training_curves.png")
plt.close()
print("  Training curves saved → nn_training_curves.png")

# ─────────────────────────────────────────────
# 5. FINAL TEST SET EVALUATION
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("FINAL TEST SET EVALUATION")
print("=" * 60)

models_eval = {
    "Logistic Regression": (lr,       lr_thresh),
    "Random Forest":       (best_rf,  rf_thresh),
    "XGBoost":             (best_xgb, xgb_thresh),
}

results = {}

for name, (model, thresh) in models_eval.items():
    prob = model.predict_proba(X_test)[:, 1]
    pred = (prob >= thresh).astype(int)
    results[name] = {
        'Accuracy':  accuracy_score(y_test, pred),
        'Precision': precision_score(y_test, pred),
        'Recall':    recall_score(y_test, pred),
        'F1':        f1_score(y_test, pred),
        'ROC-AUC':   roc_auc_score(y_test, prob),
    }
    evaluate(f"{name} — TEST", y_test, pred, prob)

nn_prob_test = nn_model.predict(X_test, verbose=0).flatten()
nn_pred_test = (nn_prob_test >= nn_thresh).astype(int)
results["Neural Network"] = {
    'Accuracy':  accuracy_score(y_test, nn_pred_test),
    'Precision': precision_score(y_test, nn_pred_test),
    'Recall':    recall_score(y_test, nn_pred_test),
    'F1':        f1_score(y_test, nn_pred_test),
    'ROC-AUC':   roc_auc_score(y_test, nn_prob_test),
}
evaluate("Neural Network — TEST", y_test, nn_pred_test, nn_prob_test)

# ─────────────────────────────────────────────
# 6. MODEL COMPARISON TABLE
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("MODEL COMPARISON (Test Set)")
print("=" * 60)

results_df = pd.DataFrame(results).T
results_df = results_df.sort_values('F1', ascending=False)
print(results_df.round(4).to_string())

results_df[['Accuracy', 'Precision', 'Recall', 'F1', 'ROC-AUC']].plot(
    kind='bar', figsize=(10, 5), colormap='Set2', edgecolor='black'
)
plt.title("Model Comparison — Test Set Metrics")
plt.ylabel("Score")
plt.xticks(rotation=15, ha='right')
plt.ylim(0, 1)
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig("model_comparison.png")
plt.close()
print("\nComparison chart saved → model_comparison.png")

#─────────────────────────────────────────────
# 7. ROC CURVES
# ─────────────────────────────────────────────
plt.figure(figsize=(8, 6))

for name, (model, _) in models_eval.items():
    prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, prob)
    auc = roc_auc_score(y_test, prob)
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")

fpr_nn, tpr_nn, _ = roc_curve(y_test, nn_prob_test)
auc_nn = roc_auc_score(y_test, nn_prob_test)
plt.plot(fpr_nn, tpr_nn, label=f"Neural Network (AUC={auc_nn:.3f})")

plt.plot([0, 1], [0, 1], 'k--', label='Random Baseline')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves — All Models")
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig("roc_curves.png")
plt.close()
print("ROC curves saved → roc_curves.png")

# ─────────────────────────────────────────────
# 8. SAVE BEST MODEL (XGBoost)
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("SAVING BEST MODEL")
print("=" * 60)

with open('xgb_model.pkl', 'wb') as f:
    pickle.dump(best_xgb, f)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

with open('threshold.pkl', 'wb') as f:
    pickle.dump(float(xgb_thresh), f)

print(f" Model saved:     xgb_model.pkl")
print(f" Scaler saved:    scaler.pkl  (features: {scaler.feature_names_in_})")
print(f"Threshold saved: threshold.pkl  (value: {xgb_thresh:.2f})")

print("\n" + "=" * 60)
print("ALL DONE!")
print("=" * 60)


import pickle
import numpy as np

model = pickle.load(open('xgb_model.pkl', 'rb'))
threshold = pickle.load(open('threshold.pkl', 'rb'))

print(f"Threshold: {threshold}")

# Check what probabilities model is actually outputting
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# Quick reload
df = pd.read_csv("Telco_Cusomer_Churn.csv")
df['TotalCharges'] = df['TotalCharges'].replace(' ', np.nan)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'])
df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())
df = df.drop('customerID', axis=1)

binary_cols = ['gender', 'Partner', 'Dependents',
               'PhoneService', 'PaperlessBilling', 'Churn']
le = LabelEncoder()
for col in binary_cols:
    df[col] = le.fit_transform(df[col])

multi_cols = ['MultipleLines', 'InternetService', 'OnlineSecurity',
              'OnlineBackup', 'DeviceProtection', 'TechSupport',
              'StreamingTV', 'StreamingMovies', 'Contract', 'PaymentMethod']
df = pd.get_dummies(df, columns=multi_cols, drop_first=True)
df = df.astype(int)
df = df.drop('TotalCharges', axis=1)

X = df.drop('Churn', axis=1)
y = df['Churn']

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

scaler = pickle.load(open('scaler.pkl', 'rb'))
scale_cols = ['tenure', 'MonthlyCharges']
X_val[scale_cols] = scaler.transform(X_val[scale_cols])

# Check probability distribution
probs = model.predict_proba(X_val)[:, 1]
print(f"\nProbability Stats:")
print(f"Min:  {probs.min():.4f}")
print(f"Max:  {probs.max():.4f}")
print(f"Mean: {probs.mean():.4f}")
print(f"\nHow many above threshold {threshold:.2f}: {(probs >= threshold).sum()}")
print(f"How many below threshold {threshold:.2f}: {(probs < threshold).sum()}")