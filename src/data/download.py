import pandas as pd
import requests
import os

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

# Création du répertoire si il n'existe pas
os.makedirs(os.path.dirname(raw_path + fic_data), exist_ok=True)

# Ecrtiure du contenu dans un fichier
with open(raw_path + fic_data, 'w', encoding='utf-8') as fichier:
    fichier.write(content)