"""
explore_data.py - Aqui vamos a mirar un poco por encima los datos que tenemos
Lo hemos implementado para ejecutarlo desde el Run and Debug con "Explorar datos" como el "Lanzar Proyecto"
"""

import os
import sys
sys.path.append(os.path.abspath(os.path.join("..", ".."))) #para importar src

import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

#En dataset_semseg tenemos un archivo que lee las imagenes y masks y las devuelve como tensores de pytorch
#Entonces basicamente hacemos SupermarketSemSeg(carpeta_imagenes, carpeta_marcaras) y empareja cada imagen con su mascara por nombre de archivo
#Si hacemos train[i] lee la pareja 1 -> abre imagen y la convierte en tensor[3,144,192] entre 0 y 1. Abre la masj y tenso[144,192] de enteros 0-10
#Devolveria las dos juntas     
from src.data.dataset_semseg import SupermarketSemSeg
from src.utils.paths import DATASET_DIR, RESULTS_DIR

#Abrimos el json asociado a las imagenes, tiene la lista de las 11 clases, y que numero es cada clase 
with open(DATASET_DIR / "label_mapping_10classes.json") as f:
    NOMBRES = [clase["name"] for clase in json.load(f)]
N_CLASES = len(NOMBRES)

#CARGAMOS LOS DATOS
train = SupermarketSemSeg(DATASET_DIR / "train" / "images", DATASET_DIR / "train" / "masks")
test = SupermarketSemSeg(DATASET_DIR / "test" / "images", DATASET_DIR / "test" / "masks")
print("Imagenes en train: ", len(train))
print("Imagenes en test: ", len(test))

#MIRAMOS UN EJEMPLO 
img, mask = train[0]
print("\nImagen :", tuple(img.shape), img.dtype, "| valores entre", img.min().item(), "y", img.max().item())
print("Máscara:", tuple(mask.shape), mask.dtype, "| clases presentes:", mask.unique().tolist())
 

def contar(dataset):
    """
    Recorre todas las máscaras y cuenta:
       - pixeles[c]: cuántos píxeles hay de la clase c
       - imagenes[c]: en cuántas imágenes aparece la clase c
       - clases_por_imagen[i]: cuántos productos distintos tiene la imagen i
    """
    pixeles = torch.zeros(N_CLASES)
    imagenes = torch.zeros(N_CLASES)
    clases_por_imagen = []
    for _, mask in dataset:
        pixeles += torch.bincount(mask.flatten(), minlength=N_CLASES)
        presentes = mask.unique().long()#clases que salen en esta máscara
        imagenes[presentes] += 1
        clases_por_imagen.append(len(presentes) - 1)#-1 para no contar el fondo
    return pixeles, imagenes, clases_por_imagen
 
 
for nombre, dataset in [("TRAIN", train), ("TEST", test)]:
    pixeles, imagenes, clases_por_imagen = contar(dataset)
    porcentaje = 100 * pixeles / pixeles.sum()
 
    print(f"\n{nombre}")
    print(f"  {'clase':<32} {'% píxeles':>9} {'imágenes':>9}")
    for c in range(N_CLASES):
        print(f"  {c:>2} {NOMBRES[c]:<29} {porcentaje[c]:8.2f}% {int(imagenes[c]):>9}")
    for k in sorted(set(clases_por_imagen)):
        print(f"  Imágenes con {k} producto(s): {clases_por_imagen.count(k)}")
 
    if nombre == "TEST":
        indices_varias_clases = [i for i, k in enumerate(clases_por_imagen) if k >= 2]
 
 
#NOS INTERESA VER ALGUN EJEMPLO ASI QUE GUARDAMOS ALGUNOS EN RESULTS 
def guardar_ejemplos(dataset, indices, archivo):
    """
    Una fila por ejemplo: imagen a la izquierda, máscara a la derecha.
    """
    fig, axes = plt.subplots(len(indices), 2, figsize=(6, 2.3 * len(indices)), squeeze=False)
    for fila, i in enumerate(indices):
        img, mask = dataset[i]
        axes[fila, 0].imshow(img.permute(1, 2, 0))  # PyTorch [3,H,W] -> matplotlib [H,W,3]
        # cmap da un color a cada clase; si no, la máscara (valores 0-10) se ve negra
        axes[fila, 1].imshow(mask, cmap="tab20", vmin=0, vmax=19, interpolation="nearest")
        productos = [NOMBRES[c] for c in mask.unique().tolist() if c != 0]
        axes[fila, 1].set_title(", ".join(productos), fontsize=8)
        axes[fila, 0].axis("off")
        axes[fila, 1].axis("off")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / archivo)
    plt.close(fig)
 
 
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
guardar_ejemplos(train, [0, 150, 300, 450, 600], "ejemplos_train.png")
guardar_ejemplos(test, indices_varias_clases[:4], "ejemplos_test_varias_clases.png")
print("\nFiguras guardadas en", RESULTS_DIR)