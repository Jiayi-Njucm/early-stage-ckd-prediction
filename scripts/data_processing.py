"""
Note:
The dataset used in this study contains clinical data and is not publicly available due to privacy constraints.
All code in this notebook is provided for demonstration and reproducibility purposes.

Author: Jiadong Xie
Date: 2025-06-10
Project: early-stage CKD Prediction
"""

# 1. Data Processing
# 1.1 Import Required Packages

import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.model_selection import train_test_split

## 1.2 Feature Definition and Data Loading
continuous_features = []
categorical_features = []
features = continuous_features + categorical_features
processed_data_file = ""
processed_labels_file = ""
X = pd.read_csv(processed_data_file)
y = pd.read_csv(processed_labels_file)['CKD']

# 1.3 (Optional) Feature Name Conversion (Chinese to English)
# This step can be skipped if the original dataset already uses English feature names.
feature_mapping = {}
xueChangGui_all = []
niaoChangGui_all = []
xueShengHua_all = []
selected_features =  xueChangGui_all + niaoChangGui_all  + xueShengHua_all + ['SEX', 'AGE']
selected_features_en = [feature_mapping[feature] for feature in selected_features]
X.columns = selected_features_en

# 1.4 Dataset Splitting

from imblearn.under_sampling import RandomUnderSampler
import numpy as np
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42, stratify=y)
X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

print(f"Number of training samples: {len(y_train)}")
print(f"Number of validation samples: {len(y_valid)}")
print(f"Number of test samples: {len(y_test)}")

print("Number of positive samples in training set:", np.sum(y_train == 1))
print("Number of negative samples in training set:", np.sum(y_train == 0))
print("Number of positive samples in validation set:", np.sum(y_valid == 1))
print("Number of negative samples in validation set:", np.sum(y_valid == 0))
print("Number of positive samples in test set:", np.sum(y_test == 1))
print("Number of negative samples in test set:", np.sum(y_test == 0))
