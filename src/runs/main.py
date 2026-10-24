import os
import sys
sys.path.append(os.path.abspath(os.path.join('..', '..')))  # Esto permite importar funciones definidas localmente en otras carpetas

from src.models.fake_model import FakeModel  # Ejemplo de importación de librerías externas
from src.data.loaders import crear_loaders
from src.utils.paths import DATASET_DIR

if __name__ == '__main__':
    pass
    # Este script os puede servir de punto de entrada
    train_loader, val_loader, test_loader = crear_loaders(DATASET_DIR)
    print("train: ", len(train_loader.dataset), "- val: ", len(val_loader.dataset), "- test: ", len(test_loader.dataset))

    imgs, masks = next(iter(train_loader))
    print("imgs: ", imgs.shape, imgs.dtype)
    print("masks:", masks.shape, masks.dtype)