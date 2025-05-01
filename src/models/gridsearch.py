import pandas as pd
from sklearn.model_selection import GridSearchCV 
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