import pandas as pd
import os
from src.tools.config import load_config
from sklearn.model_selection import train_test_split

# Paramètres
data_config = load_config('./params.yaml')


url_data = data_config['url_data']
raw_path = data_config['raw_path']  
split_path = data_config['split_path']
fic_data = 'raw.csv'

# Charger les données dans pandas
df = pd.read_csv(raw_path + fic_data)

# Séparer les données en features et target
y = df['silica_concentrate']
X = df.drop(columns=['silica_concentrate', 'date'], axis=1)

# Création des des jeux d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)

# Création du répertoire si il n'existe pas
os.makedirs(os.path.dirname(split_path  + 'X_train.csv'), exist_ok=True)

# Enregistrer les données dans processed_data
X_train.to_csv(split_path + 'X_train.csv', index=False)
X_test.to_csv(split_path + 'X_test.csv', index=False)
y_train.to_csv(split_path + 'y_train.csv', index=False)
y_test.to_csv(split_path + 'y_test.csv', index=False)