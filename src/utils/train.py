import torch
import time
from src.utils.metrics import actualizar_cm, metricas_desde_cm

def train(model, train_loader, criterion, optimizer,num_classes, val_loader=None, num_epochs=20, device='cuda', writer = None, ruta_best = None):

    # Listas para generar logs durante el entrenamiento:
    history = {"train_loss": [], "train_acc": [], "train_miou":  []} 
    
    if val_loader is not None:
        history["val_loss"] = []
        history["val_acc"] = []
        history["val_miou"] = [] 

    model.to(device)

    # Bucle de entrenamiento:
    mejor_miou = -1
    for epoch in range(num_epochs):
        model.train() #activamos el modo entrenamiento al inicio de cada epoca 

        running_loss = 0.0  # Acumulamos el valor de coste obtenido tras cada epoch
        # ya no usamos el acc normal porque lo que estamos prediciendo es casi siempre fondo al estar tan desbalanceado 
        # count_evaluated = 0 #cuantos pixeles se miran en total  
        # count_correct = 0 #pixeles acertados 
        cm = torch.zeros(num_classes, num_classes, dtype = torch.long, device = device)
           #hemos creado una confusion matrix 
        
        for batch_idx, data in enumerate(train_loader, 0):    
            # se transforma las mascaras de uint8 a long para usar CrossEntropyLoss
            inputs, labels = data[0].to(device), data[1].long().to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            running_loss += loss.item()  # Acumulamos el error obtenido para utilizarlo

            #Obtenemos la clase predicha para cada pixel (mayor probabilidad del pixel en los 11 canales)
            predictions = outputs.argmax(dim = 1) 

            cm = actualizar_cm(cm, predictions, labels, num_classes)

            # Contamos el número de pixeles evaluados y acertados:
            # Devuelve el numero total de elementos de un tensor. Es decir, si tenemos una imagen 3x144x192
            # Devolverá 3x144x192 =  82944, que será el número total de píxeles.
            # count_evaluated += labels.numel()
            # count_correct += (predictions == labels).sum().item()

        m = metricas_desde_cm(cm)
        epoch_loss = running_loss / len(train_loader)

        # epoch_acc = count_correct / count_evaluated

        history["train_loss"].append(epoch_loss)
        history["train_acc"].append(m["pixel_acc"]) 
        history["train_miou"].append(m["miou"]) 
        # Almacenamos la accuracy al final de la epoch (en train)
        # train_acc.append(epoch_acc)
        
        print(f"Training: epoch {epoch + 1}, loss: {epoch_loss:.3f}, "
              f"pixel acc: {m['pixel_acc']:.3f}, mIoU: {m['miou']:.3f}")

        if writer is not None:
            writer.add_scalar("Loss/train", epoch_loss, epoch)
            writer.add_scalar("PixelAcc/train", m["pixel_acc"], epoch)
            writer.add_scalar("mIoU/train", m["miou"], epoch) 
        
        
        ### Fase de validación ### 
        if val_loader is not None:
            model.eval() #se "desactiva" el modo entrenamiento 
            running_loss_val = 0.0
            # count_evaluated_val = 0
            # count_correct_val = 0
            cm_val = torch.zeros(num_classes, num_classes, dtype = torch.long, device =device)
            
            with torch.no_grad():
                for inputs_val, labels_val in val_loader:
                    
                    inputs_val, labels_val = inputs_val.to(device), labels_val.long().to(device)
                    outputs_val = model(inputs_val)

                    loss_val = criterion(outputs_val, labels_val)
                    running_loss_val += loss_val.item()

                    predictions_val = outputs_val.argmax(dim = 1)

                    # count_evaluated_val += labels_val.numel()
                    # count_correct_val += (predictions_val == labels_val).sum().item()
                    cm_val = actualizar_cm(cm_val, predictions_val, labels_val, num_classes)
                
                # Presentamos el resumen de la validación de la epoch:
                m_val = metricas_desde_cm(cm_val)
                epoch_loss_val = running_loss_val / len(val_loader)
                history["val_loss"].append(epoch_loss_val)
                history["val_acc"].append(m_val["pixel_acc"]) 
                history["val_miou"].append(m_val["miou"])
                
                # val_loss.append(epoch_loss_val)
                # val_acc.append(epoch_acc_val)

                print(f"Validation: epoch {epoch + 1}, loss: {epoch_loss_val:.3f}, "
                  f"pixel acc: {m_val['pixel_acc']:.3f}, mIoU: {m_val['miou']:.3f}")

                if ruta_best is not None and m_val["miou_sin_fondo"] > mejor_miou:
                    mejor_miou = m_val["miou_sin_fondo"]
                    torch.save(model.state_dict(), ruta_best)
                    print(f"  -> best.pth guardado (época {epoch + 1}, mIoU sin fondo {mejor_miou:.3f})")

                if writer is not None:
                    writer.add_scalar("Loss/val", epoch_loss_val, epoch)
                    writer.add_scalar("PixelAcc/val", m_val["pixel_acc"], epoch)
                    writer.add_scalar("mIoU/val", m_val["miou"], epoch)
        
        if writer is not None:
            for name, param in model.named_parameters():
                writer.add_histogram("Parametros/" + name, param.detach().cpu(), epoch) # En cpu para loggear
                if param.grad is not None:
                    writer.add_histogram("Gradientes/" + name, param.grad.detach().cpu(), epoch) #Pues puede fallar si algún parámetro no tiene gradiente 
    
          
    return model, history