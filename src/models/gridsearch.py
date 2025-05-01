import pandas as pd
from sklearn.model_selection import GridSearchCV 
from sklearn.linear_model import LinearRegression
import pickle
from sklearn import ensemble
from src.tools.config import load_config

# Paramètres
data_config = load_config('./params.yaml')

split_path = data_config['split_path']
model_path = data_config['model_path']
normalize_path = data_config['normalize_path']
split_path = data_config['split_path']

# Création du modèle
model = LinearRegression()

# Chargement des données
X = pd.read_csv(normalize_path + 'X_train_scaled.csv')
y = pd.read_csv(split_path + 'y_train.csv')

# Paramètres pour la recherche 
param_grid = { 
    'fit_intercept': [True, False], 
    'positive': [True, False],
    'copy_X': [True, False]
} 

# Création de la recherche par grille
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error') 

# Lancer la recherche
grid_search.fit(X, y) 

# Sauvegarde des meilleurs paramètres
best_params = grid_search.best_params_


with open(model_path + "best_params.pkl", "wb") as f:
    pickle.dump(best_params, f)