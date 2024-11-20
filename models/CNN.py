import torch

class CNN(torch.nn.Module):
    def __init__(self, in_channels=1, classes=10):
        super(CNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels, 32, kernel_size=5)
        self.max_pool1 = torch.nn.MaxPool2d(2, 2)

        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=5)
        self.max_pool2 = torch.nn.MaxPool2d(2, 2)
        self.linear = torch.nn.Linear(64 * 4 * 4, classes)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.max_pool1(x)

        x = self.conv2(x)
        x = self.max_pool2(x)

        x = x.view(x.size(0), -1)
        x = self.linear(x)
        x = self.relu(x)
        return x


if __name__ == '__main__':
    model = CNN(3, 10)
    print(model)
    x = torch.randn((1, 3, 28, 28))
    y = model(x)
    print(y.shape)
    print(y)
