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


def train(model_root):
    import time
    model = "CNN"
    epochs = 30
    batch_size = 64
    lr = 0.001
    
    day = time.strftime("%d", time.localtime())
    time_now = time.strftime("%H_%M", time.localtime())
    
    # load model if exists
    model_name = 'model_best.pth'
    if os.path.exists(os.path.join(model_root, model_name)):
        model = torch.load(model_root)
    else:
        model = getattr(models, model)
    model=model.CNN(1, 10)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    print('Current device:', torch.cuda.get_device_name(torch.cuda.current_device()))
    

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
                images = images.unsqueeze(1).float().to(device)
                labels = labels.long().to(device)
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
            torch.save(model.state_dict(), os.path.join(model_root, f"model_best_{day}_{time_now}.pth"))
        
        # save the latest model
        torch.save(model.state_dict(), os.path.join(model_root, f"model_latest_{day}_{time_now}.pth"))
    
    
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
    plt.savefig(os.path.join("results", f"training_loss_accuracy_curves_{day}_{time_now}.png"))
    
    return day, time_now

def test(model_path, test_path, model="CNN"):
    model = getattr(models, model)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=model.CNN(1, 10).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    test_dataset = MnistDataset(test_path)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    correct = 0
    total = 0
    with torch.no_grad():
        with tqdm(total=len(test_loader)) as _tqdm:
            for images, labels in test_loader:
                images = images.unsqueeze(1).float().to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                _tqdm.update(1)
    print('Accuracy of the model on the test images: %d %%' % (100 * correct / total))

if __name__ == '__main__':
    test_path = "./mnist_test"
    model_root = "./weights"
    day, time = train(model_root=model_root)
    model_path = os.path.join(model_root, f"model_latest_{day}_{time}.pth")
    
    print('start running the test...')
    
    test(model_path=model_path, test_path=test_path, model="CNN")
