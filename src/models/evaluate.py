import pandas as pd
import pickle
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np
from src.tools.config import load_config
import json
import os

# Paramètres
data_config = load_config('./params.yaml')
split_path = data_config['split_path']
normalize_path = data_config['normalize_path']
model_path = data_config['model_path']
prediction_path = data_config['prediction_path']
metrics_path = data_config['metrics_path']

# Chargement du modèle
with open(model_path + "model.pkl", "rb") as f:
    model = pickle.load(f)

# Chargement des données de test
X_test = pd.read_csv(normalize_path + 'X_test_scaled.csv')
y_test = pd.read_csv(split_path + 'y_test.csv')

# Prédictions
y_pred = model.predict(X_test)

# Scores : coefficient de détermination, erreur absolue moyenne, racine de l'erreur quadratique moyenne
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# sauvegarde des résultats en json
results = {
    "r2": r2,
    "mae": mae,
    "rmse": rmse
}

with open(metrics_path + "scores.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=4, ensure_ascii=False)

# Sauvegarde des prédictions
os.makedirs(os.path.dirname(prediction_path + "predictions.csv"), exist_ok=True)
y_pred_df = pd.DataFrame(y_pred, columns=["predictions"])
y_pred_df.to_csv(prediction_path + "predictions.csv", index=False)