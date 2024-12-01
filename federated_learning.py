import torch
from torch.utils.data import Subset, DataLoader
from Dataset import MnistDataset
import importlib
import models

def split_data(dataset, num_clients):
    num_samples = len(dataset)
    samples_per_client = num_samples // num_clients
    clients = {}

    for i in range(num_clients):
        start = i * samples_per_client
        end = start + samples_per_client if i < num_clients - 1 else num_samples
        clients[i] = Subset(dataset, list(range(start, end)))

    return clients

def train_client(model, client_data, epochs=5, learning_rate=0.01):
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(epochs):
        for data, target in DataLoader(client_data, batch_size=32, shuffle=True):
            data = data.unsqueeze(0).float()
            optimizer.zero_grad()
            output = model(data.view(-1, 1, 28, 28))
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
    return model.state_dict()

def aggregate_models(models, weights):
    averaged_model = {}
    for key in models[0].keys():
        averaged_model[key] = torch.zeros_like(models[0][key])
        for model, weight in zip(models, weights):
            averaged_model[key] += weight * model[key]
        averaged_model[key] /= sum(weights)
    return averaged_model

def fed_avg(Model, models, clients, epochs=5, learning_rate=0.01, num_rounds=10):
    global_model = Model.CNN(1, 10)
    global_state_dict = global_model.state_dict()

    for round in range(num_rounds):
        print(f"Round {round + 1}/{num_rounds}")
        model_weights = []
        client_weights = []

        for client_id, client_data in clients.items():
            model = Model.CNN(1, 10)
            model.load_state_dict(global_state_dict)
            model_weights.append(train_client(model, client_data, epochs, learning_rate))
            client_weights.append(len(client_data))

        new_global_model = aggregate_models(model_weights, client_weights)
        global_model.load_state_dict(new_global_model)
        global_state_dict = new_global_model

    return global_model

def evaluate_model(model, test_data):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_data:
            data = data.unsqueeze(0).float()
            output = model(data.view(-1, 1, 28, 28))
            _, predicted = torch.max(output, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy of the model on the test images: {accuracy}%')
    return accuracy

if __name__ == "__main__":
    dataset = MnistDataset(root='mnist_train')
    model = "CNN"
    Model = getattr(models, model)
    num_clients = 5
    clients = split_data(dataset, num_clients)
    global_model = fed_avg(Model, [Model.CNN(1, 10) for _ in clients], clients)
    test_data = DataLoader(MnistDataset(root='mnist_test'), batch_size=32, shuffle=False)
    evaluate_model(global_model, test_data)
    