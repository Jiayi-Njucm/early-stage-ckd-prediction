# 2. Model Training
# 2.1 Definition of Models and Their Parameters (For hyperparameter tuning, see Section 4.2)

models = {
    'Decision Tree':DecisionTreeClassifier(),
    'Random Forest':RandomForestClassifier(),
    'MLP': MLPClassifier(),
    'AdaBoost':AdaBoostClassifier(),
    'XGBoost':xgb.XGBClassifier(),
    'LightGBM': lgb.LGBMClassifier()
}

# 2.2 Model Training and Related Evaluation Metrics

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import roc_curve, auc, confusion_matrix, average_precision_score
from sklearn.metrics import precision_recall_curve
import os
sns.set(style="whitegrid")
plt.rcParams["font.family"] = "Times New Roman"
output_dir = 'pic_train'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

model_scores = {}

def plot_confusion_matrix(cm, classes, title='Confusion Matrix', model_name='', cmap=plt.cm.Blues):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap=cmap, xticklabels=classes, yticklabels=classes, cbar=True, annot_kws={"size": 20})
    plt.title(f'{title} ({model_name})', fontsize=20)
    plt.ylabel('True Label', fontsize=16)
    plt.xlabel('Predicted Label', fontsize=16)
    plt.savefig(f'{output_dir}/confusion_matrix_{model_name}.png', dpi=300)
    plt.close()

def plot_roc_curve(models_fpr_tpr_auc, title='ROC Curve'):
    plt.figure(figsize=(8, 8))
    colors = plt.cm.tab10.colors
    for i, (name, (fpr, tpr, auc_score)) in enumerate(models_fpr_tpr_auc.items()):
        plt.plot(fpr, tpr, label=f'{name} (AUC = {auc_score:.4f})', linewidth=2, color=colors[i])
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--', linewidth=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.title(title, fontsize=18)
    plt.legend(loc='lower right', fontsize=16)
    plt.savefig(f'{output_dir}/roc_curve.png', dpi=300)
    plt.close()

def plot_prc_curve(models_prc, title='Precision-Recall Curve'):
    plt.figure(figsize=(8, 8))
    colors = plt.cm.tab10.colors
    for i, (name, (precision, recall, pr_auc)) in enumerate(models_prc.items()):
        plt.plot(recall, precision, label=f'{name} (PR-AUC = {pr_auc:.4f})', linewidth=2, color=colors[i])
    plt.xlabel('Recall', fontsize=16)
    plt.ylabel('Precision', fontsize=16)
    plt.title(title, fontsize=18)
    plt.legend(loc='lower left', fontsize=16)
    plt.savefig(f'{output_dir}/prc_curve.png', dpi=300)
    plt.close()

models_fpr_tpr_auc = {}
models_prc = {}

for i, (name, model) in enumerate(models.items()):
    if not hasattr(model, 'predict_proba'):
        print(f"Model {name} does not support `predict_proba`, skipping.")
        continue

    print(f"Training and evaluating model: {name}")
    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X_valid)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_valid, y_prob)
    auc_score = auc(fpr, tpr)
    models_fpr_tpr_auc[name] = (fpr, tpr, auc_score)
    precision, recall, _ = precision_recall_curve(y_valid, y_prob)
    pr_auc = average_precision_score(y_valid, y_prob)
    models_prc[name] = (precision, recall, pr_auc)
    y_pred_best = (y_prob >= thresholds[np.argmax(tpr - fpr)]).astype(int)
    cm = confusion_matrix(y_valid, y_pred_best)
    plot_confusion_matrix(cm, classes=["Negative", "Positive"], model_name=name)
    tn, fp, fn, tp = cm.ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    specificity = tn / (tn + fp)
    NPV=tn / (tn + fn)
    f1 = 2 * (precision * recall) / (precision + recall)
    model_scores[name] = {
        "Accuracy": accuracy,
        "F1 Score": f1,
        "Sensitivity": recall,
        "Specificity": specificity,
        "PPV":precision,
        "NPV":NPV,
        "AUC": auc_score,
        "PR-AUC":pr_auc
    }
plot_roc_curve(models_fpr_tpr_auc, title='ROC Curve Comparison')
plot_prc_curve(models_prc, title='Precision-Recall Curve Comparison')
df_model_scores= pd.DataFrame.from_dict(model_scores, orient='index')
df_model_scores.to_csv("df_model_scores.csv", index=True)



# 2.3 Different Feature Combinations

import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_score, recall_score, f1_score,
    accuracy_score, confusion_matrix, average_precision_score,
    precision_recall_curve
)
import joblib
import os

output_dir = "combination"
os.makedirs(output_dir, exist_ok=True)
models_fpr_tpr_auc = {}
models_prc = {}


def plot_and_save_confusion_matrix(cm, feature_set_name):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=True, annot_kws={"size": 14})
    plt.title(f'Confusion Matrix - {feature_set_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    cm_path = os.path.join(output_dir, f'confusion_matrix_{feature_set_name}.png')
    plt.savefig(cm_path, dpi=300, bbox_inches="tight")
    plt.close()


xueChangGui_all = []
niaoChangGui_all = []
xueShengHua_all = []
basci_info = []
feature_sets = {
    'XCG+Basic': xueChangGui_all + basci_info,
    'NCG+Basic': niaoChangGui_all + basci_info,
    'XSH+Basic': xueShengHua_all + basci_info,
    'XCG+NCG+Basic': xueChangGui_all + niaoChangGui_all + basci_info,
    'XCG+XSH+Basic': xueChangGui_all + xueShengHua_all + basci_info,
    'NCG+XSH+Basic': niaoChangGui_all + xueShengHua_all + basci_info,
    'XCG+NCG+XSH+Basic': xueChangGui_all + niaoChangGui_all + xueShengHua_all + basci_info
}
feature_combination_results = {}
for feature_set_name, feature_list in feature_sets.items():
    print(f"\nEvaluating feature combination: {feature_set_name}")
    X_train_subset = X_train[feature_list]
    X_valid_subset = X_valid[feature_list]

    model = xgb.XGBClassifier()  # Use the optimized hyperparameters in this section
    model.fit(X_train_subset, y_train)
    y_prob = model.predict_proba(X_valid_subset)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_valid, y_prob)
    auc_score = roc_auc_score(y_valid, y_prob)
    precision, recall, _ = precision_recall_curve(y_valid, y_prob)
    pr_auc = average_precision_score(y_valid, y_prob)
    models_fpr_tpr_auc[feature_set_name] = (fpr, tpr, auc_score)
    models_prc[feature_set_name] = (precision, recall, pr_auc)
    youden_index = tpr - fpr
    best_threshold = thresholds[np.argmax(youden_index)]
    best_youden = np.max(youden_index)
    y_pred_best = (y_prob >= best_threshold).astype(int)
    cm = confusion_matrix(y_valid, y_pred_best)
    tn, fp, fn, tp = cm.ravel()
    accuracy = accuracy_score(y_valid, y_pred_best)
    recall = recall_score(y_valid, y_pred_best)
    specificity = tn / (tn + fp)
    ppv = precision_score(y_valid, y_pred_best)
    npv = tn / (tn + fn)
    f1 = f1_score(y_valid, y_pred_best)
    feature_combination_results[feature_set_name] = {
        "Youden’s J": best_youden,
        "Optimal Threshold": best_threshold,
        "Accuracy": accuracy,
        "F1 Score": f1,
        "Sensitivity": recall,
        "Specificity": specificity,
        "PPV": ppv,
        "NPV": npv,
        "AUC": auc_score,
        "PR-AUC": pr_auc
    }
    plot_and_save_confusion_matrix(cm, feature_set_name)

plt.figure(figsize=(8, 8))
colors = plt.cm.tab10.colors
for i, (name, (fpr, tpr, auc_score)) in enumerate(models_fpr_tpr_auc.items()):
    plt.plot(fpr, tpr, label=f'{name} (AUC = {auc_score:.4f})', linewidth=2, color=colors[i])
plt.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=2)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
roc_path = os.path.join(output_dir, "roc_curve_all.png")
plt.savefig(roc_path, dpi=300, bbox_inches="tight")
plt.close()

plt.figure(figsize=(8, 8))
for i, (name, (precision, recall, pr_auc)) in enumerate(models_prc.items()):
    plt.plot(recall, precision, label=f'{name} (PR-AUC = {pr_auc:.4f})', linewidth=2, color=colors[i])
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve Comparison")
plt.legend()
prc_path = os.path.join(output_dir, "prc_curve_all.png")
plt.savefig(prc_path, dpi=300, bbox_inches="tight")
plt.close()

df_feature_results = pd.DataFrame.from_dict(feature_combination_results, orient='index')
df_feature_results.to_csv("feature_combination_performance.csv", index=True)


# 2.4 Internal Testing
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_score, recall_score, f1_score,
    accuracy_score, confusion_matrix, average_precision_score
)
feature_list = feature_sets['XCG+NCG+Basic']
X_train_subset = X_train[feature_list]
X_valid_subset = X_valid[feature_list]
X_test_subset = X_test[feature_list]
model = xgb.XGBClassifier() # Use the optimized hyperparameters in this section
model.fit(X_train_subset, y_train)
y_prob = model.predict_proba(X_test_subset)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
auc_score = roc_auc_score(y_test, y_prob)
youden_index = tpr - fpr
best_threshold = thresholds[np.argmax(youden_index)]
best_youden = np.max(youden_index)
y_pred_best = (y_prob >= best_threshold).astype(int)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred_best).ravel()
accuracy = accuracy_score(y_test, y_pred_best)
precision = precision_score(y_test, y_pred_best)
recall = recall_score(y_test, y_pred_best)
specificity = tn / (tn + fp)
ppv = precision
npv = tn / (tn + fn)
f1 = f1_score(y_test, y_pred_best)
pr_auc = average_precision_score(y_test, y_prob)

evaluation_results = {
    "AUC": auc_score,
    "Youden’s J": best_youden,
    "Optimal Threshold": best_threshold,
    "Accuracy": accuracy,
    "Precision": precision,
    "F1 Score": f1,
    "Sensitivity": recall,
    "Specificity": specificity,
    "PPV": ppv,
    "NPV": npv,
    "PR-AUC": pr_auc
}
df_evaluation = pd.DataFrame([evaluation_results], index=['XCG+NCG+Basic'])
model_path = "model.pkl"
joblib.dump(model, model_path)
