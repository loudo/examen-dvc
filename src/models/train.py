import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

# Paramètres
processed_path = './data/processed_data/'
model_path = './models/'

# Création du modèle
model = LinearRegression()

# Chargement des données
X = pd.read_csv(processed_path + 'X_train_scaled.csv')
y = pd.read_csv(processed_path + 'y_train.csv')

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