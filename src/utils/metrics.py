"""
CONCLUSION DE PORQUE NO NOS SIRVE CON EL PIXEL ACC: 

En nuestro dataset el fondo ocupa un porcentaje muy elevado (90%)
Un modelo que pinte toda la imagen como fondo saca un 0.9 de acc detectando 0 productos
Sacamos la matriz de confusion que es num_classes x num_clasess
donde la fila es la clase real y la columna la predicha
Por tanto cm[r, p] = nº piceles que eran clase r y se predijo p
La diagonal son los aciertos
Sacamos entonces los TP(aciertos) = cm[c,c], los FP(falsas alarmas) y las FN(no detectados)  
Asi calculamos IoU = TP / (TP + FP + FN) -> solapa lo predicho y lo real
Dice = 2TP / (2TP + FP + FN)
mIoU = media del IoU para todas las clases, todas pesan igual. 


Explicacion del indice Intersection over Union
A = pixeles que son de una zanahoria, B = pixeles que el modelo dice que son de una zanahoria
Interseccion A,B -> TP (pixeles que son y que el modelo dice que son zanahoria)
Union A,B -> TP + FP + FN (pixeles que son zanahoria o el modelo cree que lo son)
Solapamiento entre el area real y la predicha

El dice es F1 score aplicado a pixeles
"""
import torch

""" Este no es muy optimo para 60 clases, hace demasiadas pasadads
def actualizar_cm(cm, preds, labels, num_classes):
    for r in range(num_classes):
        for p in range(num_classes):
            cm[r,p] += ((labels == r) & (preds == p)).sum()
    return cm
"""

def actualizar_cm(cm, preds, labels, num_classes):
    # cada pareja (real, predicha) pasa a un indice unico real*num_classes + predicha
    # y bincount cuenta cuantas veces sale cada una
    idx = labels.flatten() * num_classes + preds.flatten()
    cm += torch.bincount(idx, minlength=num_classes ** 2).reshape(num_classes, num_classes)
    return cm



def metricas_desde_cm(cm):
    cm = cm.double()
    tp = cm.diag()
    fp = cm.sum(dim = 0) - tp
    fn = cm.sum(dim = 1) - tp

    iou = tp / (tp + fn + fp)
    dice = 2*tp / (2*tp + fp + fn)
    pixel_acc = tp.sum() / cm.sum()

    ausentes = cm.sum(dim = 1) == 0
    iou[ausentes] = float("nan")
    dice[ausentes] = float("nan")  

    return {"pixel_acc" : pixel_acc.item(),
            "miou": torch.nanmean(iou).item(),
            "miou_sin_fondo": torch.nanmean(iou[1:]).item(),
            "mdice": torch.nanmean(dice).item(),
            "iou_por_clase": iou.tolist(),}  