"""
paths.py - Aqui generamos las rutas absolutas del proyecto
Es decir todas las rutas se van a calcular a partir de la ubicacion de este archivo
Funcionan igual en todas las maquinas, y no importa la carpeta desde la cual se ejecute el script
Se importan desde aqui las rutas

Ejemplo: from src.utils.paths.py
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] #raiz del proyecto

#DATOS
DATA_DIR = ROOT / "data"
DATASET_10_DIR = DATA_DIR /"dataset_res_144_192_10classes"
DATASET_60_DIR = DATA_DIR /"dataset_res_144_192" #60 clases 
DATASET_DIR = DATASET_60_DIR #por defecto

#CONFIGURACION COMPARTIDA
CONFIG_DIR = ROOT / "config"

#SALIDAS DE LOS EXPERIMENTOS
REPORTS_DIR = ROOT /"reports"
MODELS_DIR = REPORTS_DIR /"models"
RESULTS_DIR = REPORTS_DIR /"results"
RUNS_DIR = REPORTS_DIR /"runs"


def split_dirs(dataset_dir, split):
    """
    Devuelve (carpeta_imagenes, carpeta_mascaras) de una partición.
    split: "train" o "test".
    Ejemplo: split_dirs(DATASET_DIR, "train") -> (.../train/images, .../train/masks)
    """
    return dataset_dir / split / "images", dataset_dir / split / "masks"
 
 
def label_mapping_path(dataset_dir):
    """
    Ruta del JSON con los nombres de las clases.
    Se llama distinto según la versión del dataset (label_mapping.json o
    label_mapping_10classes.json), así que devolvemos el que haya en la raíz
    del dataset.
    """
    candidatos = sorted(dataset_dir.glob("label_mapping*.json"))
    if not candidatos:
        raise FileNotFoundError(f"No hay ningún label_mapping*.json en {dataset_dir}")
    return candidatos[0]
 
 
if __name__ == "__main__":
    # Imprime cada ruta y si existe en esta máquina.
    for nombre, ruta in [("ROOT", ROOT), ("DATA_DIR", DATA_DIR), ("DATASET_DIR", DATASET_DIR),
                         ("CONFIG_DIR", CONFIG_DIR), ("MODELS_DIR", MODELS_DIR),
                         ("RESULTS_DIR", RESULTS_DIR), ("RUNS_DIR", RUNS_DIR)]:
        estado = "OK" if ruta.exists() else "NO EXISTE"
        print(f"{nombre:<12} {estado:<10} {ruta}")



