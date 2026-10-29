import torch
from src.utils.metrics import actualizar_cm, metricas_desde_cm

def test(model, test_loader, criterion, num_classes, device='cuda'):
    model.to(device)
    model.eval()
    running_loss = 0.0
    cm = torch.zeros(num_classes, num_classes, dtype = torch.long, device = device)
    with torch.no_grad():
        # number_pixels = 0
        # number_correct = 0
        # running_loss_test = 0.0
        
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.long().to(device)
            outputs = model(inputs)
            running_loss += criterion(outputs, labels).item()
            predictions = outputs.argmax(dim=1)
            cm = actualizar_cm(cm, predictions, labels, num_classes)

    m = metricas_desde_cm(cm)
    m["loss"] = running_loss / len(test_loader)
    # De donde vienen los errores (filas = clase real, columnas = predicha)
    c = cm.double()
    m["fondo_como_producto"] = (c[0, 1:].sum() / c[0].sum()).item()
    m["producto_como_fondo"] = (c[1:, 0].sum() / c[1:].sum()).item()
    m["producto_como_otro"] = ((c[1:, 1:].sum() - c[1:, 1:].diag().sum()) / c[1:].sum()).item()


    print(f"Test - Pixel acc: {m['pixel_acc']:.3f} | mIoU: {m['miou']:.3f} | "
          f"mIoU sin fondo: {m['miou_sin_fondo']:.3f} | mDice: {m['mdice']:.3f} | "
          f"CrossEntropy: {m['loss']:.3f}")
    for c, valor in enumerate(m["iou_por_clase"]):
        print(f"  clase {c:2d}: IoU = {valor:.3f}")

        # De donde vienen los errores (filas = clase real, columnas = predicha)
    c = cm.double()
    m["fondo_como_producto"] = (c[0, 1:].sum() / c[0].sum()).item()
    m["producto_como_fondo"] = (c[1:, 0].sum() / c[1:].sum()).item()
    m["producto_como_otro"] = ((c[1:, 1:].sum() - c[1:, 1:].diag().sum()) / c[1:].sum()).item()

    # return acc_test, mean_loss_test
    return m, cm