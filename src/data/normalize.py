from sklearn.preprocessing import StandardScaler
import pandas as pd
import os

# Paramètres
raw_path = './data/raw_data/'
split_path = './data/split_data/'
normalize_path = './data/normalize_data/'

# Normaliser les données
scaler = StandardScaler()

# Chargement des données
X_train = pd.read_csv(split_path + 'X_train.csv')
X_test = pd.read_csv(split_path + 'X_test.csv')

# Normalisation
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Création du répertoire si il n'existe pas
os.makedirs(os.path.dirname(normalize_path  + 'X_train_scaled.csv'), exist_ok=True)

# Enregistrer les données normalisées
pd.DataFrame(X_train_scaled, columns=X_train.columns).to_csv(normalize_path + 'X_train_scaled.csv', index=False)
pd.DataFrame(X_test_scaled, columns=X_test.columns).to_csv(normalize_path + 'X_test_scaled.csv', index=False)