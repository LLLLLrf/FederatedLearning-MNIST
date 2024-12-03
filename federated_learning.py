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


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
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
    # divide the dataset into non-IID groups based on the label
    label_to_indices = defaultdict(list)
    for idx, (_, label) in enumerate(dataset):
        label_to_indices[label].append(idx)
    
    # shuffle the order of samples in each label group
    for label in label_to_indices:
        random.shuffle(label_to_indices[label])
    
    # assign samples to clients
    clients = {i: [] for i in range(num_clients)}
    available_labels = list(range(num_classes))
    
    for client_id in range(num_clients):
        # randomly decide the number of classes for the client
        num_classes_for_client = random.randint(min_classes_per_client, max_classes_per_client)
        assigned_labels = random.sample(available_labels, num_classes_for_client)
        
        for label in assigned_labels:
            num_samples_for_label = len(label_to_indices[label]) // (num_clients // num_classes_for_client)
            clients[client_id].extend(label_to_indices[label][:num_samples_for_label])
            label_to_indices[label] = label_to_indices[label][num_samples_for_label:]
        
    # convert to Subset objects
    for client_id in clients:
        clients[client_id] = Subset(dataset, clients[client_id])
    
    return clients

def train_client(client_id, model, client_data, epochs=10, learning_rate=0.001, batch_size=32):
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0

    for epoch in range(epochs):
        for data, target in DataLoader(client_data, batch_size=batch_size, shuffle=True):
            data = data.unsqueeze(0).float()
            optimizer.zero_grad()
            output = model(data.view(-1, 1, 28, 28))
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            # Accuracy calculation
            _, predicted = torch.max(output, 1)
            correct += (predicted == target).sum().item()
            total += target.size(0)

    avg_loss = total_loss / (len(client_data) // batch_size)  # Average loss per batch
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

def fed_avg(Model, models, clients, epochs=10, learning_rate=0.001, num_rounds=10, batch_size=32):
    global_model = Model.CNN(1, 10)
    global_state_dict = global_model.state_dict()

    for round in range(num_rounds):
        print(f"Round {round + 1}/{num_rounds}")
        model_weights = []
        client_weights = []
        round_losses = []
        round_accuracies = []
        avg_losses = []
        avg_accuracies = []
        
        for client_id, client_data in clients.items():
            if len(client_data) == 0:
                print(f"Warning: Client {client_id} has no data.")
                continue
            
            model = Model.CNN(1, 10)
            model.load_state_dict(global_state_dict)
            # send initail weights to client
            # train on the client and collect metrics
            state_dict, client_loss, client_accuracy = train_client(client_id, model, client_data, epochs, learning_rate)
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
    plt.savefig("training_metrics_over_rounds.png")
    plt.show()

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
    day = time.strftime("%d", time.localtime())
    time = time.strftime("%H_%M", time.localtime())
    
    model_path = "weights"
    mode = "non-IID"
    
    dataset = MnistDataset(root='mnist_train')
    model = "CNN"
    Model = getattr(models, model)
    num_clients = 5
    epochs = 1
    num_rounds = 30
    learning_rate = 0.0001
    batch_size = 32
    
    if mode == "IID":
        clients = split_data(dataset, num_clients)
    elif mode == "non-IID":
        clients = split_data_non_iid(dataset, num_clients)
    else:
        raise ValueError("Invalid mode. Please choose IID or non-IID.")
    
    global_model = fed_avg(Model, [Model.CNN(1, 10) for _ in clients], clients, epochs, learning_rate, num_rounds, batch_size)

    # save the global model
    torch.save(global_model.state_dict(), os.path.join(model_path,'fed_global_model_{}_{}.pt'.format(day, time)))
    print("model saved to {}".format(os.path.join(model_path,'fed_global_model_{}_{}.pt'.format(day, time))))

    test_data = DataLoader(MnistDataset(root='mnist_test'), batch_size=32, shuffle=False)
    evaluate_model(global_model, test_data)
    