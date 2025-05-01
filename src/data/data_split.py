import pandas as pd
import requests

from sklearn.model_selection import train_test_split

# Paramètres
url_data = 'https://datascientest-mlops.s3.eu-west-1.amazonaws.com/mlops_dvc_fr/raw.csv'
raw_path = './data/raw_data/'
processed_path = './data/processed_data/'
fic_data = 'raw.csv'

# Téléchargement du fichier
response = requests.get(url_data)

# Récupération du contenu
content = response.text  

# Ecrtiure du contenu dans un fichier
with open(raw_path + fic_data, 'w', encoding='utf-8') as fichier:
    fichier.write(content)

# Charger les données dans pandas
df = pd.read_csv(raw_path + fic_data)

# Séparer les données en features et target
y = df['silica_concentrate']
X = df.drop(columns=['silica_concentrate', 'date'], axis=1)

# Création des des jeux d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)

# Enregistrer les données dans processed_data
X_train.to_csv(processed_path + 'X_train.csv', index=False)
X_test.to_csv(processed_path + 'X_test.csv', index=False)
y_train.to_csv(processed_path + 'y_train.csv', index=False)
y_test.to_csv(processed_path + 'y_test.csv', index=False)