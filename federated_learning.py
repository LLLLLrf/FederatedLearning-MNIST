import torch
from torch.utils.data import Subset, DataLoader
from Dataset import MnistDataset
import importlib
import models
import time
import random
import os
from collections import defaultdict
import matplotlib.pyplot as plt

if torch.cuda.is_available():
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    print('Current device:', torch.cuda.get_device_name(torch.cuda.current_device()))

def split_data(dataset, num_clients):
    num_samples = len(dataset)
    samples_per_client = num_samples // num_clients
    clients = {}

    for i in range(num_clients):
        start = i * samples_per_client
        end = start + samples_per_client if i < num_clients - 1 else num_samples
        clients[i] = Subset(dataset, list(range(start, end)))

    return clients

def split_data_non_iid(dataset, num_clients, num_classes=10, min_classes_per_client=1, max_classes_per_client=3):
    label_to_indices = defaultdict(list)
    for idx, (_, label) in enumerate(dataset):
        label_to_indices[label].append(idx)

    # Shuffle the data for each label
    for label in label_to_indices:
        random.shuffle(label_to_indices[label])

    clients = {i: [] for i in range(num_clients)}
    available_labels = list(range(num_classes))

    # Initial allocation to ensure each client gets data
    for client_id in range(num_clients):
        num_classes_for_client = random.randint(min_classes_per_client, max_classes_per_client)
        assigned_labels = random.sample(available_labels, num_classes_for_client)

        for label in assigned_labels:
            if label_to_indices[label]:  # Check if there is data left for the label
                num_samples_for_label = max(1, len(label_to_indices[label]) // num_clients)
                clients[client_id].extend(label_to_indices[label][:num_samples_for_label])
                label_to_indices[label] = label_to_indices[label][num_samples_for_label:]

    # Redistribute remaining data to clients with fewer samples
    for label in available_labels:
        while label_to_indices[label]:
            for client_id in range(num_clients):
                if not label_to_indices[label]:
                    break
                clients[client_id].append(label_to_indices[label].pop())

    # Convert to Subset objects
    for client_id in clients:
        clients[client_id] = Subset(dataset, clients[client_id])

    return clients


def train_client(client_id, model, client_data, epochs=10, learning_rate=0.001, batch_size=32):
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0
    num_batches = 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(epochs):
        for data, target in DataLoader(client_data, batch_size=batch_size, shuffle=True):
            data, target = data.to(device), target.to(device)
            data = data.float().view(-1, 1, 28, 28)  # Ensure correct input shape
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1
            # Accuracy calculation
            _, predicted = torch.max(output, 1)
            correct += (predicted == target).sum().item()
            total += target.size(0)

    avg_loss = total_loss / len(DataLoader(client_data, batch_size=batch_size))

    accuracy = 100 * correct / total if total > 0 else 0.0
    print(f"Client {client_id} - Loss: {avg_loss:.6f} - Accuracy: {accuracy:.2f}%")
    
    return model.state_dict(), avg_loss, accuracy

def aggregate_models(models, weights):
    averaged_model = {}
    for key in models[0].keys():
        averaged_model[key] = torch.zeros_like(models[0][key])
        for model, weight in zip(models, weights):
            averaged_model[key] += weight * model[key]
        averaged_model[key] /= sum(weights)
    return averaged_model

def fed_avg(Model, model_name, models, model_path, clients, cfg, epochs=1, learning_rate=0.001, num_rounds=10, batch_size=32):
    if model_name=="LeNet5":
        global_model = Model.LeNet5(1, 10)
    elif model_name=="CNN":
        global_model = Model.CNN(1, 10)
    else:
        print("Model not found")
        return
    global_state_dict = global_model.state_dict()
    avg_losses = []
    avg_accuracies = []
    
    day = cfg['day']
    time_now = cfg['time_now']
    mode = cfg['mode']
    num_clients = cfg['num_clients']

    for round in range(num_rounds):
        print(f"Round {round + 1}/{num_rounds}")
        model_weights = []
        client_weights = []
        round_losses = []
        round_accuracies = []

        for client_id, client_data in clients.items():
            if len(client_data) == 0:
                print(f"Warning: Client {client_id} has no data.")
                continue
            if model_name=="LeNet5":
                model = Model.LeNet5(1, 10)
            elif model_name=="CNN":
                model = Model.CNN(1, 10)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)
            model.load_state_dict(global_state_dict)
            # send initial weights to client
            # train on the client and collect metrics
            state_dict, client_loss, client_accuracy = train_client(client_id, model, client_data, epochs, learning_rate, batch_size)
            model_weights.append(state_dict)
            client_weights.append(len(client_data))
            round_losses.append(client_loss)
            round_accuracies.append(client_accuracy)

        # Aggregate metrics
        avg_losses.append(sum(round_losses) / len(round_losses))
        avg_accuracies.append(sum(round_accuracies) / len(round_accuracies))

        # Aggregate models
        new_global_model = aggregate_models(model_weights, client_weights)
        global_model.load_state_dict(new_global_model)
        global_state_dict = new_global_model

    # evaluate model convergence
    convergence_epoch = None
    threshold = 0.01
    for i in range(1, len(avg_losses)):
        if abs(avg_losses[i] - avg_losses[i - 1]) < threshold:
            convergence_epoch = i + 1
            break
    
    if convergence_epoch:
        print(f"Model starts to converge at epoch {convergence_epoch}.")
    else:
        print("Model did not converge within the training epochs.")
        
    rounds = range(1, len(avg_losses) + 1)

    # Plot training metrics
    plt.figure(figsize=(12, 6))

    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(rounds, avg_losses, marker='o', label="Average Loss")
    plt.title("Average Training Loss Over Rounds")
    if convergence_epoch:
        plt.axvline(x=convergence_epoch, linestyle='--', color='red', label="Convergence Epoch")
    plt.xlabel("Rounds")
    plt.ylabel("Loss")
    plt.grid()
    plt.legend()

    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(rounds, avg_accuracies, marker='o', label="Average Accuracy", color='orange')
    plt.title("Average Training Accuracy Over Rounds")
    plt.xlabel("Rounds")
    plt.ylabel("Accuracy (%)")
    plt.grid()
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join("results", f"fed_{mode}_{num_clients}clients_{day}_{time_now}.png"))
    plt.show()

    return global_model

def evaluate_model(model, test_data):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        for data, target in test_data:
            data, target = data.to(device), target.to(device)
            data = data.float().view(-1, 1, 28, 28)
            output = model(data)
            _, predicted = torch.max(output, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy of the model on the test images: {accuracy}%')
    return accuracy

def draw_dataset_distribution(clients):
    plt.figure(figsize=(10, 6))
    for client_id, client_data in clients.items():
        if len(client_data) == 0:
            continue
        loader = DataLoader(client_data, batch_size=len(client_data))
        for data, labels in loader:
            plt.hist(labels.numpy(), bins=range(11), alpha=0.5, label=f'Client {client_id}')

    plt.title('Dataset Distribution Among Clients')
    plt.xlabel('Label')
    plt.ylabel('Number of Samples')
    plt.xticks(range(10))
    plt.legend()
    plt.savefig('dataset_distribution.png')

if __name__ == "__main__":
    day = time.strftime("%d", time.localtime())
    time_now = time.strftime("%H_%M", time.localtime())
    
    model_path = "weights"
    mode = "IID"
    print("running {} mode...".format(mode))
    
    dataset = MnistDataset(root='mnist_train')
    model = "CNN"
    Model = getattr(models, model)
    num_clients = 15
    epochs = 1
    num_rounds = 30
    learning_rate = 0.001
    batch_size = 64
        
    cfg = {}
    cfg["mode"]=mode
    cfg["num_clients"]=num_clients
    cfg["model"]=model
    cfg["epochs"]=epochs
    cfg["num_rounds"]=num_rounds
    cfg["learning_rate"]=learning_rate
    cfg["batch_size"]=batch_size
    cfg["day"]=day
    cfg["time_now"]=time_now

    
    if mode == "IID":
        clients = split_data(dataset, num_clients)
    elif mode == "non-IID":
        clients = split_data_non_iid(dataset, num_clients)
    else:
        raise ValueError("Invalid mode. Please choose IID or non-IID.")
    
    draw_dataset_distribution(clients)
    if model == "LeNet5":
        global_model = fed_avg(Model, model, [Model.LeNet5(1, 10) for _ in clients], model_path, clients, cfg, epochs, learning_rate, num_rounds, batch_size)
    elif model == "CNN":
        global_model = fed_avg(Model, model, [Model.CNN(1, 10) for _ in clients], model_path, clients, cfg, epochs, learning_rate, num_rounds, batch_size)

    # save the global model
    torch.save(global_model.state_dict(), os.path.join(model_path,'fed_global_model_{}_{}.pth'.format(day, time_now)))
    print("model saved to {}".format(os.path.join(model_path,'fed_global_model_{}_{}.pth'.format(day, time_now))))

    print("Running test...")
    test_data = DataLoader(MnistDataset(root='mnist_test'), batch_size=1, shuffle=False)
    evaluate_model(global_model, test_data)
    