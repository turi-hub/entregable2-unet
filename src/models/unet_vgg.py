"""
Vamos a probar con tranfer learning
Es nuestra misma U-Net (UNetModel), pero el encoder empieza con los pesos de VGG16
entrenada en ImageNet en vez de con pesos aleatorios.
Nuestro encoder tiene la misma estructura que los 4 primeros bloques de VGG16
(convoluciones 3x3 de 64, 128, 256 y 512 filtros con max pooling entre medias),
asi que podemos copiar esos bloques directamente. El decoder se entrena desde cero.
"""
import torch
import torch.nn as nn
from torchvision.models import vgg16_bn, VGG16_BN_Weights

from src.models.unet_model import UNetModel


def bloque_bn(entrada, salida):
    # Igual que nuestro ConvBlock (2 convoluciones 3x3 + ReLU), pero con batch norm tras cada convolucion.
    # Lo usamos en el decoder: los bloques de VGG16 llevan batch norm y el decoder tiene que ir a la par
    # (sin el, el decoder aprende mucho peor). Con batch norm el sesgo de la convolucion sobra.
    return nn.Sequential(
        nn.Conv2d(entrada, salida, kernel_size = 3, padding = 1, bias = False), nn.BatchNorm2d(salida), nn.ReLU(),
        nn.Conv2d(salida, salida, kernel_size = 3, padding = 1, bias = False), nn.BatchNorm2d(salida), nn.ReLU())


class UNetVGG(UNetModel):

    def __init__(self, num_clases = 61):
        super().__init__(num_clases = num_clases)

        # Parte convolucional de VGG16 (version con batch norm) ya entrenada en ImageNet
        vgg = vgg16_bn(weights = VGG16_BN_Weights.IMAGENET1K_V1).features

        # ENCODER: sustituimos los bloques por los de VGG16
        # (vgg[6], vgg[13] y vgg[23] son max pooling, que nuestra U-Net ya hace con self.pool)
        self.encoder1 = vgg[0:6]          # 2 convoluciones de 64 filtros
        self.encoder2 = vgg[7:13]         # 2 convoluciones de 128 filtros
        self.encoder3 = vgg[14:23]        # 3 convoluciones de 256 filtros
        self.bloque_central = vgg[24:33]  # 3 convoluciones de 512 filtros

        # DECODER: mismos canales que en nuestra U-Net, pero con batch norm
        # (entrada = canales de la convolucion traspuesta + los de la skip connection)
        self.decoder3 = bloque_bn(256 + 256, 256)
        self.decoder2 = bloque_bn(128 + 128, 128)
        self.decoder1 = bloque_bn(64 + 64, 64)

        # VGG16 se entreno con las imagenes normalizadas con la media y desviacion de ImageNet,
        # asi que normalizamos igual. register_buffer: va a la GPU con el modelo pero no se entrena
        self.register_buffer("media", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, X):
        X = (X - self.media) / self.std
        return super().forward(X)  # el resto es exactamente nuestra U-Net
