# 4. Other Sections

# 4.1 DeLong Test Function (Example Code)

def delong_test(y_true, prob1, prob2):
    auc1 = roc_auc_score(y_true, prob1)
    auc2 = roc_auc_score(y_true, prob2)
    n1 = np.sum(y_true == 1)
    n2 = np.sum(y_true == 0)
    q1 = auc1 * (1 - auc1)
    q2 = auc2 * (1 - auc2)
    var = (q1 / n1) + (q2 / n2)
    z = (auc1 - auc2) / np.sqrt(var)
    p_value = 2 * norm.sf(abs(z))
    return auc1, auc2, z, p_value

# 4.2 Cross-Validation (Example Code)

from sklearn.model_selection import GridSearchCV
xgb = XGBClassifier()
param_grid = {
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'min_child_weight': [1, 3, 5],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'n_estimators': [50, 100, 200]
}
grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid, cv=5, verbose=1, n_jobs=-1)
grid_search.fit(X_train, y_train)
print("Best parameters found: ", grid_search.best_params_)
print("Best cross-validation score: ", grid_search.best_score_)