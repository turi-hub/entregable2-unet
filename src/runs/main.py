import os
import sys
import json
import torch
import argparse
from datetime import datetime
sys.path.append(os.path.abspath(os.path.join('..', '..')))  # Esto permite importar funciones definidas localmente en otras carpetas

from src.models.unet_model import UNetModel
from src.data.loaders import crear_loaders
from src.utils.paths import DATASET_DIR, MODELS_DIR, RUNS_DIR, RESULTS_DIR
from src.utils.train import train
from src.utils.test import test
from torch.utils.tensorboard import SummaryWriter

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--nombre", type=str, default="base")  # para distinguir los experimentos de las mejoras
    args = parser.parse_args()
    torch.manual_seed(args.seed)

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
    n_params = sum(p.numel() for p in model.parameters())
    print("Parametros del modelo:", n_params)

    # Usamos CrossEntropyLoss como función de pérdida
    criterion = torch.nn.CrossEntropyLoss()

    # Usamos Adam como optimizador
    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)

    # Nombre unico por ejecucion, asi no se mezclan los logs ni se pisa el best.pth
    nombre = f"{args.nombre}_seed{args.seed}_{datetime.now():%Y%m%d_%H%M}"
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
    m_test, cm_test = test(model=model, test_loader=test_loader, criterion=criterion, num_classes=num_classes, device=device)

    # Guardamos todo en un json para sacar las figuras del informe
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / f"{nombre}.json", "w") as f:
        json.dump({"args": vars(args), "n_params": n_params, "history": history,
                   "test": m_test, "cm_test": cm_test.cpu().tolist()}, f, indent=2)
    print("Resultados guardados en", RESULTS_DIR / f"{nombre}.json")

if __name__ == '__main__':
    main()
