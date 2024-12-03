import cv2
import numpy as np
import os
import torch
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from tqdm import tqdm
import models
from Dataset import MnistDataset

os.environ["CUDA_VISIBLE_DEVICES"] = "4"

def train():
    import time
    model = "CNN"
    epochs = 30
    batch_size = 32
    lr = 0.0001
    
    day = time.strftime("%d", time.localtime())
    time_now = time.strftime("%H_%M", time.localtime())
    
    # load model if exists
    model_path = 'weights/'
    model_name = 'model_best.pth'
    if os.path.exists(os.path.join(model_path, model_name)):
        model = torch.load(model_path)
    else:
        model = getattr(models, model)
    model=model.CNN(1, 10)
    
    print('Current device:', torch.cuda.get_device_name(torch.cuda.current_device()))
    
    print('Model:', model)

    # load data
    dataset = MnistDataset('mnist_train')
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    print('Data size:', len(dataset))
    
    # define loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_loss = float('inf')
    
    losses = []
    accuracies = []
    total, correct = 0, 0
    # train
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        with tqdm(total=len(dataloader)) as _tqdm:
            _tqdm.set_description(f'Epoch: {epoch}/{epochs}')
            for i, (images, labels) in enumerate(dataloader):
                images = images.unsqueeze(1).float()
                labels = labels.long()
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                _tqdm.set_postfix(loss=f'{loss.item():.6f}')
                _tqdm.update(1)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                
        
        # calculate average loss and accuracy
        avg_loss = epoch_loss / len(dataloader)
        accuracy = 100 * correct / total if total > 0 else 0.0
        losses.append(avg_loss)
        accuracies.append(accuracy)
        print(f'Epoch {epoch}, Average Loss: {avg_loss:.6f}, Accuracy: {accuracy:.2f}%')

        
        # save the model if the loss is lower than the best loss
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model, os.path.join(model_path, f"model_best_{day}_{time_now}.pth"))
        
        # save the latest model
        torch.save(model, os.path.join(model_path, f"model_latest_{day}_{time_now}.pth"))
    
    
    # evaluate model convergence
    convergence_epoch = None
    threshold = 0.01
    for i in range(1, len(losses)):
        if abs(losses[i] - losses[i - 1]) < threshold:
            convergence_epoch = i + 1
            break
    
    if convergence_epoch:
        print(f"Model starts to converge at epoch {convergence_epoch}.")
    else:
        print("Model did not converge within the training epochs.")

    # draw loss and accuracy curves
    plt.figure(figsize=(12, 6))

    # Loss curve
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs + 1), losses, marker='o', label='Loss')
    plt.title("Training Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    if convergence_epoch:
        plt.axvline(x=convergence_epoch, linestyle='--', color='red', label="Convergence Epoch")
    plt.grid()
    plt.legend()

    # Accuracy curve
    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs + 1), accuracies, marker='o', label='Accuracy', color='orange')
    plt.title("Training Accuracy Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.grid()
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(model_path, f"training_loss_accuracy_curves_{day}_{time_now}.png"))
    plt.show()
    
if __name__ == '__main__':
    train()
