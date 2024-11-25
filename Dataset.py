import os
import cv2
import numpy as np
from torch.utils.data import DataLoader, Dataset

class MnistDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.images = []
        self.labels = []
        for i in range(10):
            # print('init data:', i)
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
        # print('label:',label)
        image = image / 255.0
        image = cv2.resize(image, (28, 28))
        if self.transform:
            image = self.transform(image)
        return image, label
