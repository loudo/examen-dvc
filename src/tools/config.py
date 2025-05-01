import yaml

# Chargement d'un fichier de configuration YAML
def load_config(config_file):
  with open(config_file, "r") as fichier:
    data = yaml.safe_load(fichier)
  
  return data