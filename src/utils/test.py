

def test(model, test_loader, criterion, device='cuda'):
    with torch.no_grad():
        number_samples = 0
        number_correct = 0
        running_loss_test = 0.0
        for test_batch_idx, data_test in enumerate(test_loader, 0):
            inputs_test, labels_test = data_test[0].to(device), data_test[1].long().to(device)
            outputs_test = model(inputs_test)
            loss = criterion(outputs_test, labels_test)
            running_loss_test += loss.cpu().numpy()
            # Accuracy:
            _, outputs_class = torch.max(outputs_test, dim=1)
            number_correct += torch.sum(outputs_class == labels_test).cpu().numpy()
            number_samples += len(labels_test)
        acc_test = number_correct / number_samples
        print('Test - Accuracy: %.3f' % acc_test)
        print('Test - CrossEntropy: %.3f' % (running_loss_test / (test_batch_idx+1)))