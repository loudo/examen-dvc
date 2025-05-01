import pandas as pd
import pickle
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np

# Paramètres
processed_path = './data/processed_data/'
model_path = './models/'

# Chargement du modèle
with open(model_path + "model.pkl", "rb") as f:
    model = pickle.load(f)

# Chargement des données de test
X_test = pd.read_csv(processed_path + 'X_test_scaled.csv')
y_test = pd.read_csv(processed_path + 'y_test.csv')

# Prédictions
y_pred = model.predict(X_test)

# Scores : coefficient de détermination, erreur absolue moyenne, racine de l'erreur quadratique moyenne
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# sauvegarde des résultats en json
results = {
    'r2': r2,
    'mae': mae,
    'rmse': rmse
}

with open(model_path + "scores.json", "w") as f:
    f.write(str(results))
