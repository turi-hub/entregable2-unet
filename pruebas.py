"""
Aqui comparamos la version del dataset de 10 clases con la de 60

Lo hicimos porque el modelo de 10 clases iba bien en validacion pero muy mal en test,
y en test predecia como producto mucho pixel que la mascara decia que era fondo.
Queriamos saber si de verdad era fondo o si habia algo mas.

La idea es que las imagenes de la version de 10 clases tambien estan en la de 60, asi que
podemos poner sus dos mascaras una al lado de la otra. Si un pixel es fondo en la de 10
pero producto en la de 60, es un producto de otra clase que la version de 10 marca como fondo.

Que hace el archivo:
  1. Comprueba que las imagenes son las mismas en las dos versiones y que nuestras 10 clases
     estan etiquetadas igual
  2. Cuenta cuantas imagenes tienen productos de otras clases marcados como fondo y que parte
     del fondo son en realidad productos (en train y en test)
  3. Dice que productos son y cuantos se parecen a los nuestros (otras manzanas, otros tes...)
  4. Saca una tabla resumen de las dos versiones (imagenes, % de fondo, productos por imagen...)
  5. Guarda la figura comparacion_10_60.png en reports/results: imagen, nuestra mascara en verde
     y en rojo los productos que la version de 10 marca como fondo

Los numeros que salen son los que usamos en la seccion de datos y de resultados del informe.
"""


import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Buscamos la raiz del proyecto (la carpeta que contiene src) para poder importar
ROOT = Path.cwd()
while not (ROOT / "src").exists() and ROOT != ROOT.parent:
    ROOT = ROOT.parent
sys.path.append(str(ROOT))

from src.utils.paths import DATASET_10_DIR, DATASET_60_DIR, RESULTS_DIR, label_mapping_path

def leer_nombres(dataset_dir):
    with open(label_mapping_path(dataset_dir)) as f:
        return [clase["name"] for clase in json.load(f)]  # posicion en la lista = id de la clase

NOMBRES10 = leer_nombres(DATASET_10_DIR)
NOMBRES60 = leer_nombres(DATASET_60_DIR)

# TABLA[id_60] = id_10 si es una de nuestras clases, 0 (fondo) si no
TABLA = np.zeros(256, dtype=np.uint8)
for id60, nombre in enumerate(NOMBRES60):
    if nombre in NOMBRES10:
        TABLA[id60] = NOMBRES10.index(nombre)

print(len(NOMBRES10), "clases en la version de 10 y", len(NOMBRES60), "en la de 60")
print("Clases de producto emparejadas:", int((TABLA > 0).sum()))  # tiene que salir 10


def cargar(split):
    d = {"nombres": [], "m10": [], "m60": [], "dif_imagen": []}
    faltan = 0
    for p10 in sorted((DATASET_10_DIR / split / "masks").glob("*.png")):
        p60 = DATASET_60_DIR / split / "masks" / p10.name
        if not p60.exists():
            faltan += 1
            continue
        nombre = p10.name.replace("_mask.png", "")
        img10 = np.asarray(Image.open(DATASET_10_DIR / split / "images" / f"{nombre}.jpg").convert("RGB"), dtype=np.int16)
        img60 = np.asarray(Image.open(DATASET_60_DIR / split / "images" / f"{nombre}.jpg").convert("RGB"), dtype=np.int16)
        d["nombres"].append(nombre)
        d["m10"].append(np.asarray(Image.open(p10).convert("L")))
        d["m60"].append(np.asarray(Image.open(p60).convert("L")))
        d["dif_imagen"].append(np.abs(img10 - img60).mean())
    d["m10"] = np.stack(d["m10"])
    d["m60"] = np.stack(d["m60"])
    print(f"{split}: {len(d['nombres'])} imagenes emparejadas, {faltan} sin pareja en la version de 60")
    return d

DATOS = {split: cargar(split) for split in ["train", "test"]}

for split, d in DATOS.items():
    # pixeles que en la de 60 son fondo o una de nuestras 10 clases
    nuestras = (d["m60"] == 0) | (TABLA[d["m60"]] > 0)
    iguales = (TABLA[d["m60"]][nuestras] == d["m10"][nuestras]).mean()
    # pixeles de las otras 50 clases: en la de 10 deberian ser fondo
    otras_como_fondo = (d["m10"][~nuestras] == 0).mean()
    print(f"{split}: diferencia media entre imagenes = {np.mean(d['dif_imagen']):.3f} | "
          f"nuestras clases etiquetadas igual = {100 * iguales:.2f}% | "
          f"pixeles de otras clases que en la de 10 son fondo = {100 * otras_como_fondo:.2f}%")


for split, d in DATOS.items():
    otros = (d["m10"] == 0) & (d["m60"] != 0)   # fondo en la de 10, producto en la de 60
    fondo10 = d["m10"] == 0
    afectada = otros.any(axis=(1, 2))
    n_etiquetados = np.array([len(np.unique(m)) - 1 for m in d["m10"]])
    n_ocultas = np.array([len(np.unique(m60[o])) for m60, o in zip(d["m60"], otros)])
    total = d["m10"].size

    print(f"\n{split.upper()}")
    print(f"  Imagenes con productos de otras clases como fondo: {afectada.sum()} de {len(afectada)} ({100 * afectada.mean():.1f}%)")
    print(f"  Fondo (version de 10) que es producto en la de 60: {100 * otros.sum() / fondo10.sum():.1f}%")
    print(f"  Fondo segun la version de 10: {100 * fondo10.sum() / total:.2f}% | fondo real: {100 * (fondo10.sum() - otros.sum()) / total:.2f}%")
    for k in sorted(set(n_etiquetados)):
        sel = n_etiquetados == k
        print(f"  Con {k} producto(s) etiquetado(s): {sel.sum()} imagenes | afectadas {afectada[sel].sum()} "
              f"({100 * afectada[sel].mean():.1f}%) | fondo que es producto {100 * otros[sel].sum() / fondo10[sel].sum():.1f}%")
    print(f"  Clases ocultas por imagen: min {n_ocultas.min()}, media {n_ocultas.mean():.1f}, max {n_ocultas.max()}")
"""  """

# Clases de las otras 50 que se parecen a cada una de las nuestras (elegidas a mano por el nombre)
HERMANOS = {
    "coca_cola_05": ["coca_cola_light_05"],
    "koelln_muesli_schoko": ["koelln_muesli_fruechte", "dr_oetker_vitalis_knuspermuesli_klassisch"],
    "apple_granny_smith": ["apple_braeburn_bundle", "apple_golden_delicious", "apple_red_boskoop"],
    "banana_single": ["banana_bundle"],
    "oranges": ["orange_single", "clementine", "clementine_single"],
    "pasta_reggia_spaghetti": ["pasta_reggia_elicoidali", "pasta_reggia_fusilli"],
    "gepa_bio_und_fair_fencheltee": ["gepa_bio_und_fair_kamillentee", "gepa_bio_und_fair_kraeuterteemischung",
                                     "gepa_bio_und_fair_pfefferminztee", "gepa_bio_und_fair_rooibostee"],
    "cucumber": ["zucchini"],
    "carrot": [],
    "lettuce": ["salad_iceberg", "corn_salad", "rocket"],
}
nombres_hermanos = {h for lista in HERMANOS.values() for h in lista}

for split, top in [("train", 5), ("test", 15)]:
    d = DATOS[split]
    otros = (d["m10"] == 0) & (d["m60"] != 0)
    ids, pixeles = np.unique(d["m60"][otros], return_counts=True)
    print(f"\n{split.upper()}: productos marcados como fondo (porcentaje de esos pixeles)")
    for i in np.argsort(-pixeles)[:top]:
        print(f"  {NOMBRES60[ids[i]]:<45} {100 * pixeles[i] / pixeles.sum():5.1f}%")

    if split == "test":
        es_hermano = np.array([NOMBRES60[i] in nombres_hermanos for i in ids])
        print(f"\n  Parecidos a alguna de nuestras clases: {100 * pixeles[es_hermano].sum() / pixeles.sum():.1f}% de esos pixeles")
        for nuestra, lista in HERMANOS.items():
            px = sum(pixeles[ids == NOMBRES60.index(h)].sum() for h in lista)
            print(f"    {nuestra:<32} {100 * px / pixeles.sum():5.1f}%")

        # imagenes de test con mas "fondo" que en realidad es producto (para elegir la figura)
        frac = otros.sum(axis=(1, 2)) / (d["m10"] == 0).sum(axis=(1, 2))
        print("\n  Imagenes con mas fondo que es producto:")
        for i in np.argsort(-frac)[:10]:
            print(f"    {d['nombres'][i]}  {100 * frac[i]:.1f}%")

def estadisticas(dataset_dir, split):
    n_clases = len(leer_nombres(dataset_dir))
    pixeles = np.zeros(n_clases)
    imagenes = np.zeros(n_clases)
    productos = []
    for p in sorted((dataset_dir / split / "masks").glob("*.png")):
        m = np.asarray(Image.open(p).convert("L"))
        pixeles += np.bincount(m.ravel(), minlength=n_clases)[:n_clases]
        presentes = np.unique(m)
        imagenes[presentes] += 1
        productos.append(len(presentes) - 1)
    productos = np.array(productos)
    return [len(productos), 100 * pixeles[0] / pixeles.sum(), int(imagenes[1:].min()), int(imagenes[1:].max()),
            100 * (productos == 1).mean(), int(productos.max())]

print(f"{'':<16}{'imagenes':>9}{'% fondo':>9}{'img/clase':>11}{'% 1 prod.':>11}{'max prod.':>10}")
for version, ddir in [("10 clases", DATASET_10_DIR), ("60 clases", DATASET_60_DIR)]:
    for split in ["train", "test"]:
        n, fondo, cmin, cmax, uno, maxp = estadisticas(ddir, split)
        print(f"{version + ' ' + split:<16}{n:>9}{fondo:>9.2f}{f'{cmin}-{cmax}':>11}{uno:>11.1f}{maxp:>10}")

EJEMPLOS = ["D2S_043709", "D2S_047118", "D2S_068215"]  # se pueden cambiar por las que salgan en la celda 5
VERDE, ROJO, ALFA = np.array([0.0, 1.0, 0.0]), np.array([1.0, 0.0, 0.0]), 0.55

def pintar(img, mascara, color):
    out = img.copy()
    out[mascara] = (1 - ALFA) * out[mascara] + ALFA * color
    return out

d = DATOS["test"]
titulos = ["Imagen", "Máscara 10 clases", "Otros productos (fondo)"]
fig, axes = plt.subplots(len(EJEMPLOS), 3, figsize=(9, 2.4 * len(EJEMPLOS)), squeeze=False)
for fila, nombre in enumerate(EJEMPLOS):
    i = d["nombres"].index(nombre)
    img = np.asarray(Image.open(DATASET_10_DIR / "test" / "images" / f"{nombre}.jpg").convert("RGB")) / 255
    etiquetado = d["m10"][i] != 0
    otros = (d["m10"][i] == 0) & (d["m60"][i] != 0)
    paneles = [img, pintar(img, etiquetado, VERDE), pintar(pintar(img, etiquetado, VERDE), otros, ROJO)]
    for col in range(3):
        axes[fila, col].imshow(paneles[col])
        axes[fila, col].axis("off")
        if fila == 0:
            axes[fila, col].set_title(titulos[col], fontsize=9)
    axes[fila, 0].text(3, 12, nombre, color="yellow", fontsize=7)
fig.tight_layout()
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
fig.savefig(RESULTS_DIR / "comparacion_10_60.png", dpi=200, bbox_inches="tight")
plt.show()
