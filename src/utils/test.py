import torch

def test(model, test_loader, criterion, device='cuda'):
    with torch.no_grad():
        number_pixels = 0
        number_correct = 0
        running_loss_test = 0.0
        
        for test_batch_idx, data_test in enumerate(test_loader, 0):
            inputs_test, labels_test = data_test[0].to(device), data_test[1].long().to(device)
            outputs_test = model(inputs_test)
            loss_test = criterion(outputs_test, labels_test)
            running_loss_test += loss_test.item()
            
            
            outputs_class_test = outputs_test.argmax(dim = 1)

            number_correct += (outputs_class_test == labels_test).sum().item()
            number_pixels += labels_test.numel()

    acc_test = number_correct / number_pixels
    mean_loss_test = running_loss_test/len(test_loader)

    print(f'Test - Pixel Accuracy: {acc_test:.3f}')
    print(f'Test - CrossEntropy: {mean_loss_test:.3f}')

    return acc_test, mean_loss_test