import torch
from torch.utils.data import Subset

def split_data(dataset, num_clients):
    num_samples = len(dataset)
    samples_per_client = num_samples // num_clients
    clients = {}

    for i in range(num_clients):
        start = i * samples_per_client
        end = start + samples_per_client if i < num_clients - 1 else num_samples
        clients[i] = Subset(dataset, list(range(start, end)))

    return clients

num_clients = 5
clients = split_data(MnistDataset(root='path_to_dataset'), num_clients)
