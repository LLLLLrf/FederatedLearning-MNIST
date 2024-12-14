import numpy as np
import struct
import os
import torch
from torch.utils.data import DataLoader, Dataset
from Dataset import MnistDataset
from tqdm import tqdm
import models
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# test path: mnist_test/0~9/xxx.png
test_path = './mnist_test'
model_root = './weights'
model_name = 'fed_global_model_14_15_58.pth'
model_path = os.path.join(model_root, model_name)

print('load model from:', model_path)

def test(model_path, test_path, model_name="LeNet5"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = getattr(models, model_name)

    if model_name == "LeNet5":
        model = model.LeNet5(1, 10).to(device)
    elif model_name == "CNN":
        model = model.CNN(1, 10).to(device)
    else:
        raise ValueError("Unknown model name")
    
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
                print('total:', total, 'correct:', correct, 'accuracy:', correct / total)
                
                _tqdm.update(1)
    print('Accuracy of the model on the test images: %d %%' % (100 * correct / total))

if __name__ == '__main__':
    test(model_path, test_path, 'CNN')