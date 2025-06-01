# Import libraries and data
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold, LeaveOneOut
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('data/alzheimers.csv')

# Drop PatientID and DoctorInCharge as specified
df = df.drop(['PatientID', 'DoctorInCharge'], axis=1)

# Define function for model comparison
def collect_metrics(y_true, y_pred, method_name):
    return {
        'method': method_name,
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1_score': f1_score(y_true, y_pred)
    }

# Prepare data for model fitting
# Separate features and target
X = df.drop('Diagnosis', axis=1)
y = df['Diagnosis']

# One-hot encoding for categorical features
X = pd.get_dummies(X, drop_first=True)

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Hyper-parameter tuning using grid search
param_grid = {
    'hidden_layer_sizes': [
        (50,), (100,), (150,),
        (100, 50), (150, 75),
        (100, 50, 25)
    ],
    'activation': ['relu', 'tanh'],
    'alpha': [0.00001, 0.0001, 0.001, 0.01, 0.1],
    'solver': ['adam'],
    'learning_rate_init': [0.0001, 0.001, 0.01, 0.1],
    'max_iter': [5000]
}

ann = MLPClassifier(random_state=49)

grid_search = GridSearchCV(
    ann,
    param_grid,
    cv=5,
    scoring='f1',
    n_jobs=-1,
    verbose=1
)

print("Starting grid search...")
grid_search.fit(X_scaled, y)

# Use best estimator
best_ann = grid_search.best_estimator_
print("Best Parameters:", grid_search.best_params_)
print("Best F1 Score:", grid_search.best_score_)

# Initialize metrics list for comparing different validation methods
metrics_list = []

# Fit model using test train split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=49)
best_ann.fit(X_train, y_train)
y_pred = best_ann.predict(X_test)

metrics_list.append(collect_metrics(y_test, y_pred, method_name='Train/Test Split'))

# Fit model using K-fold
skf = StratifiedKFold(n_splits=10)
y_true_all, y_pred_all = [], []

for train_idx, test_idx in skf.split(X_scaled, y):
    X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    best_ann.fit(X_train, y_train)
    y_pred = best_ann.predict(X_test)

    y_true_all.extend(y_test)
    y_pred_all.extend(y_pred)

metrics_list.append(collect_metrics(y_true_all, y_pred_all, method_name='Stratified K-Fold'))

# Fit model using LOOCV
loo = LeaveOneOut()
y_true_all, y_pred_all = [], []

for train_idx, test_idx in loo.split(X_scaled):
    X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    best_ann.fit(X_train, y_train)
    y_pred = best_ann.predict(X_test)

    y_true_all.append(y_test.values[0])
    y_pred_all.append(y_pred[0])

metrics_list.append(collect_metrics(y_true_all, y_pred_all, method_name='LOOCV'))

# Compare model performance
metrics_df = pd.DataFrame(metrics_list)
print("\nComparison of different validation methods:")
print(metrics_df)

# Re-fit the model using train/test split for final evaluation
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=49)
best_ann.fit(X_train, y_train)

# Generate predictions
y_pred = best_ann.predict(X_test)
y_proba = best_ann.predict_proba(X_test)[:, 1]  # Probability estimates for ROC AUC

# Collect comprehensive performance metrics
metrics = {
    'Accuracy': accuracy_score(y_test, y_pred),
    'Precision': precision_score(y_test, y_pred),
    'Recall': recall_score(y_test, y_pred),
    'F1 Score': f1_score(y_test, y_pred),
    'ROC AUC': roc_auc_score(y_test, y_proba)
}

# Convert to DataFrame for display
metrics_df = pd.DataFrame(metrics, index=['ANN - Train/Test Split'])
print("\nFinal model performance metrics:")
print(metrics_df)

# Compute confusion matrix
cm = confusion_matrix(y_test, y_pred)
labels = ['No Alzheimer\'s', 'Alzheimer\'s']

# Plot confusion matrix
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=labels, yticklabels=labels, cbar=False)

plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix - ANN Model Performance')
plt.tight_layout()
plt.savefig('alzheimers_confusion_matrix.png')
plt.show() 