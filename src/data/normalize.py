from sklearn.preprocessing import StandardScaler
import pandas as pd

# Paramètres
raw_path = './data/raw_data/'
processed_path = './data/processed_data/'

# Normaliser les données
scaler = StandardScaler()

# Chargement des données
X_train = pd.read_csv(processed_path + 'X_train.csv')
X_test = pd.read_csv(processed_path + 'X_test.csv')

# Normalisation
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Enregistrer les données normalisées
pd.DataFrame(X_train_scaled, columns=X_train.columns).to_csv(processed_path + 'X_train_scaled.csv', index=False)
pd.DataFrame(X_test_scaled, columns=X_test.columns).to_csv(processed_path + 'X_test_scaled.csv', index=False)