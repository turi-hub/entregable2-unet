import torch
import torch.nn as nn

from src.models.layers.layers import ConvBlock

class UNetModel(nn.Module):

    def __init__(self, conv_filters = [64, 128, 256, 512], num_clases = 11):
        super().__init__()

        self.conv_filters = conv_filters
        self.num_clases = num_clases

        self.pool = nn.MaxPool2d(kernel_size = 2, stride = 2)

       # ENCODER: 4 bloques de convolucion. El ultimo sin pooling (BLOQUE CENTRAL)
        self.encoder1 = ConvBlock(3, conv_filters[0]) 
        self.encoder2 = ConvBlock(conv_filters[0], conv_filters[1])
        self.encoder3 = ConvBlock(conv_filters[1], conv_filters[2])
       
       # Bloque central no se le aplica pooling, solo convoluciones.
       # Obtenemos mapas de características 18x24 de 512 canales
        self.bloque_central = ConvBlock(conv_filters[2], conv_filters[3])

       # DECODER: 3 bloques (vamos duplicando hasta recuperar resolución) y el de salida, con 11 canales (1 por clase)
        self.up3 = nn.ConvTranspose2d(conv_filters[3], conv_filters[2], kernel_size = 2, stride = 2)
        # Por 2 por las skip connections, se concatena esta 
        # resolucion con las caracteristicas aprendidas en el encoder (256 + 256 canales)
        self.decoder3 = ConvBlock(conv_filters[2] * 2, conv_filters[2]) 
        
        self.up2 = nn.ConvTranspose2d(conv_filters[2], conv_filters[1], kernel_size = 2, stride = 2)
        self.decoder2 = ConvBlock(conv_filters[1] * 2, conv_filters[1])

        self.up1 = nn.ConvTranspose2d(conv_filters[1], conv_filters[0], kernel_size = 2, stride = 2)
        self.decoder1 = ConvBlock(conv_filters[0] * 2, conv_filters[0])

      # Kernel de tamaño 1 para mantener la resolucion, cambiando el numero de canales
        self.out = nn.Conv2d(conv_filters[0], num_clases, kernel_size = 1)

        
    
    def forward(self, X):
        # ENCODER
        # se guardan las caracteristicas antes del pooling para concatenar en el decoder
        skip1 = self.encoder1(X)
        X = self.pool(skip1)

        skip2 = self.encoder2(X)
        X = self.pool(skip2)

        skip3 = self.encoder3(X)
        X = self.pool(skip3)

        # BLOQUE CENTRAL 
        X = self.bloque_central(X)

        # DECODER
        X = self.up3(X)
        X = torch.cat([X, skip3], dim = 1)#dim = 1, pues concatenamos los canales 
        X = self.decoder3(X)

        X = self.up2(X)
        X = torch.cat([X, skip2], dim = 1)
        X = self.decoder2(X)

        X = self.up1(X)
        X = torch.cat([X, skip1], dim = 1)
        X = self.decoder1(X)

        # SALIDA
        X = self.out(X)

        return X

       