import pandas as pd
import requests

url_data = 'https://datascientest-mlops.s3.eu-west-1.amazonaws.com/mlops_dvc_fr/raw.csv'
fic_data = './data/raw_data/raw.csv'

# Téléchargement du fichier
response = requests.get(url_data)

# Récupération du contenu
content = response.text  

# Ouvrir un fichier en mode écriture ('w')
with open(fic_data, 'w', encoding='utf-8') as fichier:
    fichier.write(content)


# Charger les données dans pandas

# Faire les variables X et Y

# Faire le Train Test Split

# Enregistrer les données
