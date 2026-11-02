"""
Hemos visto que hay orden en las fotos y que hacer particion aleatoria no es l omejor. Esto es para el objetivo 2
Las fotos de train vienen en series de 30 fotos consecutivas de la misma escena
(mismo producto y misma colocacion), casi identicas entre si.
Con random_split caen fotos de cada serie en train y en validacion, y la validacion
mide fotos casi iguales a las ya vistas.
Aqui mandamos SERIES ENTERAS a validacion, para validar con escenas que el modelo no ha visto.
Estratificado por producto: de cada producto con 3 o mas series, una va a validacion.
Asi ningun producto se queda sin fotos en train (hay productos con una sola serie).
"""
from torch.utils.data import DataLoader, Subset
from src.data.dataset_semseg import SupermarketSemSeg

FOTOS_POR_SERIE = 30

def crear_loaders_series(dataset_dir, batch_size = 16):
    train_completo = SupermarketSemSeg(dataset_dir / "train" / "images", dataset_dir / "train" / "masks")
    test = SupermarketSemSeg(dataset_dir / "test" / "images", dataset_dir / "test" / "masks")

    # Agrupamos las series por producto. Las fotos estan ordenadas por nombre, asi que
    # la serie s son las fotos s*30 ... s*30+29. Miramos la mascara de su primera foto.
    series_por_producto = {}
    for s in range(len(train_completo) // FOTOS_POR_SERIE):
        mascara = train_completo[s * FOTOS_POR_SERIE][1]
        producto = mascara.max().item()  # en las pocas series con 2 productos, el de mayor numero
        series_por_producto.setdefault(producto, []).append(s)

    # La ultima serie de cada producto con 3 o mas series va a validacion
    series_val = [series[-1] for series in series_por_producto.values() if len(series) >= 3]

    val_idx = [i for i in range(len(train_completo)) if i // FOTOS_POR_SERIE in series_val]
    train_idx = [i for i in range(len(train_completo)) if i // FOTOS_POR_SERIE not in series_val]

    train_loader = DataLoader(Subset(train_completo, train_idx), batch_size = batch_size, shuffle = True)
    val_loader = DataLoader(Subset(train_completo, val_idx), batch_size = batch_size)
    test_loader = DataLoader(test, batch_size = batch_size)
    return train_loader, val_loader, test_loader
