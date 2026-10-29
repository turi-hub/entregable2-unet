import os
import sys
import torch
import argparse
sys.path.append(os.path.abspath(os.path.join('..', '..')))  # Esto permite importar funciones definidas localmente en otras carpetas

from src.models.unet_model import UNetModel  # Ejemplo de importación de librerías externas
from src.data.loaders import crear_loaders
from src.utils.paths import DATASET_DIR, MODELS_DIR, RUNS_DIR
from src.utils.train import train
from src.utils.test import test
from torch.utils.tensorboard import SummaryWriter

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    torch.manual_seed(args.seed)

    # Este script os puede servir de punto de entrada
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Se cargan los datos 
    train_loader, val_loader, test_loader = crear_loaders(DATASET_DIR, batch_size=args.batch_size)
    print("train: ", len(train_loader.dataset), "- val: ", len(val_loader.dataset), "- test: ", len(test_loader.dataset))

    imgs, masks = next(iter(train_loader))
    print("imgs: ", imgs.shape, imgs.dtype)
    print("masks:", masks.shape, masks.dtype)

    num_classes = 11
    
    # ENTRENAMIENTO
    model = UNetModel() 

    # Usamos CrossEntropyLoss como función de pérdida
    criterion = torch.nn.CrossEntropyLoss()

    # Usamos Adam como optimizador
    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)

    nombre = f"base_seed{args.seed}"
    writer = SummaryWriter(str(RUNS_DIR / nombre))
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    ruta_best = MODELS_DIR / f"{nombre}_best.pth"
    
    model, history = train(model = model, train_loader = train_loader, criterion = criterion,
                        optimizer = optimizer, num_classes=num_classes,val_loader = val_loader, num_epochs = args.epochs, device = device,
                        writer = writer, ruta_best = ruta_best)

    writer.close()

    print("Ultimo mIoU de entrenamiento: ", history["train_miou"][-1])
    print("Ultimo mIoU de validacion: ", history["val_miou"][-1])

    # Test con el mejor modelo, no con el de la última época
    model.load_state_dict(torch.load(ruta_best, map_location=device))
    test(model=model, test_loader=test_loader, criterion=criterion, num_classes=num_classes, device=device)

    # print("Última pérdida de entrenamiento:", history["train_loss"][-1])
    # print("Última accuracy de entrenamiento:", history["train_acc"][-1])

if __name__ == '__main__':
    main()