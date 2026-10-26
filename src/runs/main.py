import os
import sys
import torch
sys.path.append(os.path.abspath(os.path.join('..', '..')))  # Esto permite importar funciones definidas localmente en otras carpetas

from src.models.unet_model import UNetModel  # Ejemplo de importación de librerías externas
from src.data.loaders import crear_loaders
from src.utils.paths import DATASET_DIR
from src.utils.train import train
from src.utils.test import test

if __name__ == '__main__':
    pass
    # Este script os puede servir de punto de entrada
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Se cargan los datos 
    train_loader, val_loader, test_loader = crear_loaders(DATASET_DIR)
    print("train: ", len(train_loader.dataset), "- val: ", len(val_loader.dataset), "- test: ", len(test_loader.dataset))

    imgs, masks = next(iter(train_loader))
    print("imgs: ", imgs.shape, imgs.dtype)
    print("masks:", masks.shape, masks.dtype)

    # ENTRENAMIENTO
    model = UNetModel() 

    # Usamos CrossEntropyLoss como función de pérdida
    criterion = torch.nn.CrossEntropyLoss()

    # Usamos Adam como optimizador
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
    
    model, history = train(model = model, train_loader = train_loader, criterion = criterion, 
                        optimizer = optimizer, val_loader = val_loader, num_epochs = 20, device = device)

    print("Última pérdida de entrenamiento:", history["train_loss"][-1])
    print("Última accuracy de entrenamiento:", history["train_acc"][-1])

    if val_loader is not None:
        print("Última pérdida de validación:", history["val_loss"][-1])
        print("Última accuracy de validación:", history["val_acc"][-1])
