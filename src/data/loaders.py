"""
loaders.py - Aqui basicamente creamos la carga de los datos que tenemos
No se nos oolvide que solo tenemos train y test entonces lo que haremos es crear tambien validacion 
"""

import torch
from torch.utils.data import DataLoader, random_split
from src.data.dataset_semseg import SupermarketSemSeg

def crear_loaders(dataset_dir, batch_size = 16):
    train_completo = SupermarketSemSeg(dataset_dir / "train" / "images", dataset_dir /"train" /"masks")
    test = SupermarketSemSeg(dataset_dir / "test" / "images", dataset_dir /"test" /"masks")

    generador = torch.Generator().manual_seed(42)
    train, val = random_split(train_completo, [0.8, 0.2], generator = generador)

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val, batch_size=batch_size)
    test_loader = DataLoader(test, batch_size=batch_size)
    return train_loader, val_loader, test_loader