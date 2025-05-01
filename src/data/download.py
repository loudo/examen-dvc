import pandas as pd
import requests
import os
from src.tools.config import load_config

# Paramètres
data_config = load_config('./params.yaml')

url_data = data_config['url_data']
raw_path = data_config['raw_path']  
split_path = data_config['split_path']
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