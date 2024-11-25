import cv2
import numpy as np
import os
import torch
from torch.utils.data import DataLoader, Dataset
import models
from Dataset import MnistDataset
from tqdm import tqdm

def train():
    import time
    model="CNN"
    epochs = 30
    batch_size = 16
    
    day = time.strftime("%d", time.localtime())
    time = time.strftime("%H_%M", time.localtime())

    # load model if exists
    model_path = 'weights/'
    model_name = 'model_best.pth'
    if os.path.exists(os.path.join(model_path, model_name)):
        model = torch.load(model_path)
    else:
        model = getattr(models, model)
    model=model.CNN(1, 10)
    print('model:', model)

    # load data
    dataset = MnistDataset('mnist_train')
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    print('data:', len(dataset))
    
    # train
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    best=float('inf')
    
    # tqdm
    for epoch in range(1, epochs+1):
        with tqdm(total=len(dataloader)) as _tqdm:
            _tqdm.set_description('epoch: {}/{}'.format(epoch, epochs))
            for i, (images, labels) in enumerate(dataloader):
                images = images.unsqueeze(1).float()
                labels = labels.long()
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                _tqdm.set_postfix(loss='{:.6f}'.format(loss.item()))
                _tqdm.update(1)
                
            if best > loss.item():
                best = loss.item()
                torch.save(model, os.path.join(model_path, "model_best_{}_{}.pth".format(day, time)))
                # print('\nmodel saved:', os.path.join(model_path, "model_best_{}_{}.pth".format(day, time)), 'loss:', loss.item())
            # save latest model
            torch.save(model, os.path.join(model_path, "model_latest_{}_{}.pth".format(day, time)))
                
            # print('epoch:', epoch, 'loss:', loss.item())
        
    torch.save(model, os.path.join(model_path, "model_latest_{}_{}.pth".format(day, time)))
    print('model saved:', os.path.join(model_path, "model_latest_{}_{}.pth".format(day, time)), 'loss:', loss.item())
    

if __name__ == '__main__':
    train()