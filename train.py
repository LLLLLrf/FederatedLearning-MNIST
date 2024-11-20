import cv2
import numpy as np
import os
import torch
from torch.utils.data import DataLoader, Dataset
import models

class MnistDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.images = []
        self.labels = []
        for i in range(10):
            print('init data:', i)
            files = os.listdir(os.path.join(root, str(i)))
            for file in files:
                self.images.append(os.path.join(root, str(i), file))
                self.labels.append(i)
        self.images = np.array(self.images)
        self.labels = np.array(self.labels)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = cv2.imread(self.images[idx], cv2.IMREAD_GRAYSCALE)
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

def train():
    model="CNN"
    epochs = 10
    
    # load model if exists
    model_path = 'model.pth'
    if os.path.exists(model_path):
        model = torch.load(model_path)
    else:
        model = getattr(models, model)
    print('model:', model)
    model=model.CNN(1, 10)

    # load data
    dataset = MnistDataset('mnist_train')
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    print('data:', len(dataset))
    
    # train
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(epochs):
        for i, (images, labels) in enumerate(dataloader):
            images = images.unsqueeze(1).float()
            labels = labels.long()
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            print('epoch:', epoch, 'step:', i, 'loss:', loss.item())
    torch.save(model, model_path)
    print('model saved:', model_path)
    

if __name__ == '__main__':
    train()