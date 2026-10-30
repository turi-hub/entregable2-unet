from pathlib import Path
import matplotlib.pyplot as plt

# Raiz del proyecto (la carpeta que contiene src) para guardar en reports/results
ROOT = Path.cwd()
while not (ROOT / "src").exists() and ROOT != ROOT.parent:
    ROOT = ROOT.parent
SALIDA = ROOT / "reports" / "results"
SALIDA.mkdir(parents=True, exist_ok=True)

# mIoU por epoca, copiado de la salida de los dos entrenamientos (40 epocas cada uno)
# "test" es el mIoU en test del mejor modelo, que solo se evalua una vez al final
MODELOS = {
    "10": {
        "train": [0.080, 0.082, 0.089, 0.115, 0.149, 0.181, 0.206, 0.216, 0.233, 0.216,
                  0.225, 0.244, 0.250, 0.306, 0.350, 0.409, 0.457, 0.457, 0.480, 0.502,
                  0.465, 0.490, 0.527, 0.573, 0.548, 0.561, 0.596, 0.625, 0.628, 0.623,
                  0.649, 0.654, 0.689, 0.731, 0.715, 0.697, 0.742, 0.753, 0.800, 0.768],
        "val":   [0.081, 0.081, 0.116, 0.118, 0.177, 0.224, 0.237, 0.242, 0.262, 0.220,
                  0.256, 0.266, 0.266, 0.344, 0.374, 0.427, 0.377, 0.463, 0.521, 0.535,
                  0.505, 0.556, 0.540, 0.584, 0.546, 0.590, 0.640, 0.670, 0.613, 0.643,
                  0.576, 0.614, 0.741, 0.740, 0.707, 0.703, 0.742, 0.804, 0.810, 0.800],
        "mejor": 39,
        "test": 0.192,
    },
    "60": {
        "train": [0.017, 0.022, 0.030, 0.032, 0.051, 0.076, 0.102, 0.133, 0.178, 0.213,
                  0.243, 0.181, 0.158, 0.239, 0.271, 0.305, 0.321, 0.341, 0.388, 0.403,
                  0.451, 0.489, 0.506, 0.532, 0.520, 0.593, 0.623, 0.593, 0.656, 0.653,
                  0.729, 0.699, 0.759, 0.712, 0.788, 0.706, 0.786, 0.755, 0.784, 0.822],
        "val":   [0.018, 0.028, 0.016, 0.039, 0.060, 0.082, 0.105, 0.162, 0.180, 0.236,
                  0.267, 0.085, 0.218, 0.273, 0.284, 0.297, 0.353, 0.332, 0.388, 0.478,
                  0.406, 0.469, 0.497, 0.527, 0.564, 0.604, 0.550, 0.492, 0.619, 0.696,
                  0.641, 0.710, 0.660, 0.693, 0.770, 0.666, 0.765, 0.750, 0.765, 0.837],
        "mejor": 40,
        "test": 0.155,
    },
}

for nombre, d in MODELOS.items():
    epocas = range(1, len(d["train"]) + 1)
    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.plot(epocas, d["train"], label="Entrenamiento")
    ax.plot(epocas, d["val"], label="Validación")
    ax.axvline(d["mejor"], color="grey", linestyle=":", label=f"Mejor modelo (época {d['mejor']})")
    # El test solo se evalua una vez, con el mejor modelo: lo pintamos como un punto
    ax.plot(d["mejor"], d["test"], "o", color="C3", markersize=7,
            label=f"Test del mejor modelo ({d['test']:.3f})")
    ax.set_xlabel("Época")
    ax.set_ylabel("mIoU")
    ax.set_xlim(1, len(d["train"]) + 1)  # un poco de margen para que el punto no quede pegado al borde
    ax.set_ylim(0, 1)                    # la misma escala en las dos para poder compararlas
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(SALIDA / f"curvas_{nombre}.png", dpi=200)
    plt.show()

print("Figuras guardadas en", SALIDA)
