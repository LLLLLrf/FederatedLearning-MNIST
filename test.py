import numpy as np
import struct
import os
import torch
from torch.utils.data import DataLoader, Dataset
from Dataset import MnistDataset

# test path: mnist_test/0~9/xxx.png
test_path = './mnist_test'
model_root = './weights'
model_name = 'model_best_25_19_57.pth'
model_path = os.path.join(model_root, model_name)

import time
def test(model_path, test_path):
    model = torch.load(model_path)
    model.eval()
    test_dataset = MnistDataset(test_path)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.unsqueeze(1).float()
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy of the model on the test images: %d %%' % (100 * correct / total))

if __name__ == '__main__':
    test(model_path, test_path)