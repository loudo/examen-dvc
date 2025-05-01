import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle
from src.tools.config import load_config

# Paramètres
data_config = load_config('./params.yaml')
split_path = data_config['split_path']
normalize_path = data_config['normalize_path']
model_path = data_config['model_path']

# Création du modèle
model = LinearRegression()

# Chargement des données
X = pd.read_csv(normalize_path + 'X_train_scaled.csv')
y = pd.read_csv(split_path + 'y_train.csv')

# Best parameters
with open(model_path + "best_params.pkl", "rb") as f:
    best_params = pickle.load(f)

# Application des meilleurs paramètres
model.set_params(**best_params)

# Entraînement du modèle
model.fit(X, y)

# Sauvegarde du modèle
with open(model_path + "model.pkl", "wb") as f:
    pickle.dump(model, f)