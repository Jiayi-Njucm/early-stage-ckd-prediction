# 3. Interpretability Analysis

# This section uses SHAP. Below is an example code for generating bar plots.
# For more details, please refer to: https://shap.readthedocs.io/en/latest/api_examples.html#plots

import shap
import matplotlib.pyplot as plt
import joblib
explainer = shap.Explainer(model, data_shap)
shap_values = explainer(data_shap)

plt.figure(figsize=(10, 6))
shap.plots.bar(shap_values, max_display=20, show=False)
shap_bar_path = "SHAP_Feature_Importance_Bar_TestSet.png"
plt.savefig(shap_bar_path, dpi=300, bbox_inches="tight")
plt.close()