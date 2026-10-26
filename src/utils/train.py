import torch

def train(model, train_loader, criterion, optimizer, val_loader=None, num_epochs=20, device='cuda', writer = None):

    # Listas para generar logs durante el entrenamiento:
    train_acc = []  
    train_loss = []
    if val_loader is not None:
        val_acc = []
        val_loss = []

    model.to(device)

    # Bucle de entrenamiento:
    for epoch in range(num_epochs):
        model.train() #activamos el modo entrenamiento al inicio de cada epoca 

        running_loss = 0.0  # Acumulamos el valor de coste obtenido tras cada epoch
        count_evaluated = 0
        count_correct = 0
        
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

            # Contamos el número de pixeles evaluados y acertados:
            # Devuelve el numero total de elementos de un tensor. Es decir, si tenemos una imagen 3x144x192
            # Devolverá 3x144x192 =  82944, que será el número total de píxeles.
            count_evaluated += labels.numel()

            count_correct += (predictions == labels).sum().item()

        
        epoch_loss = running_loss / len(train_loader)

        epoch_acc = count_correct / count_evaluated

        train_loss.append(epoch_loss)
        # Almacenamos la accuracy al final de la epoch (en train)
        train_acc.append(epoch_acc)
        
        print(
            f"Training: epoch {epoch + 1}, "
            f"loss: {epoch_loss:.3f}, "
            f"pixel accuracy: {epoch_acc:.3f}"
        )


        # Información TENSORBOARD train
        if writer is not None:
            writer.add_scalar("Loss_train", epoch_loss , epoch) #Añadimos el último valor añadido a la lista, correspondiente a la epoch 
            writer.add_scalar("Tasa_acierto_train", epoch_acc , epoch)
        
        
        ### Fase de validación ### 
        if val_loader is not None:
            model.eval() #se "desactiva" el modo entrenamiento 
            running_loss_val = 0.0
            count_evaluated_val = 0
            count_correct_val = 0
            
            with torch.no_grad():
                for inputs_val, labels_val in val_loader:
                    
                    inputs_val, labels_val = inputs_val.to(device), labels_val.long().to(device)
                    outputs_val = model(inputs_val)

                    loss_val = criterion(outputs_val, labels_val)
                    running_loss_val += loss_val.item()

                    predictions_val = outputs_val.argmax(dim = 1)

                    count_evaluated_val += labels_val.numel()
                    count_correct_val += (predictions_val == labels_val).sum().item()
                # Presentamos el resumen de la validación de la epoch:
                epoch_loss_val = running_loss_val / len(val_loader)
                epoch_acc_val = count_correct_val / count_evaluated_val
                
                val_loss.append(epoch_loss_val)
                val_acc.append(epoch_acc_val)

                print(
                f"Validation: epoch {epoch + 1}, "
                f"loss: {epoch_loss_val:.3f}, "
                f"pixel accuracy: {epoch_acc_val:.3f}"
            )
                

            # Información TENSORBOARD Val
            if writer is not None:
                writer.add_scalar("Coste_epoch_val/", epoch_loss_val , epoch)
                writer.add_scalar("Tasa_acierto_val/", epoch_acc_val  , epoch)
        
        if writer is not None:
            for name, param in model.named_parameters():
                writer.add_histogram("Parametros/" + name, param.detach().cpu(), epoch) # En cpu para loggear
                if param.grad is not None:
                    writer.add_histogram("Gradientes/" + name, param.grad.detach().cpu(), epoch) #Pues puede fallar si algún parámetro no tiene gradiente 
    
    # Devolvemos, tanto el modelo entrenado, como el diccionario con las estadísticas del entrenamiento
    history ={
        "train_loss" : train_loss,
        "train_acc" : train_acc
    }  

    if val_loader is not None:
        history["val_loss"] = val_loss
        history["val_acc"] = val_acc
          
    return model, history