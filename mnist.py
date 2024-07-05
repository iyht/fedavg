
from collections import defaultdict
import os
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset, Subset
from torchvision.transforms import Compose, ToTensor, Normalize
from torchvision import datasets, transforms

from flwr_datasets import FederatedDataset
import matplotlib.pyplot as plt
import json
import argparse
import random
import numpy as np
from functools import reduce

random.seed(0)
torch.manual_seed(0)
np.random.seed(0)

def load_train_conf(config_path):
    with open(config_path, 'r') as file:
        train_conf = json.load(file)
    return train_conf



# pylint: disable=unsubscriptable-object
class Net(nn.Module):
    """
    The code is adapted from https://github.com/pytorch/examples/blob/main/mnist/main.py
    """
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output



def partition_data_iid(train_dataset, num_clients):
    """
    Return the num_clients number of list of indices. The indices are randomly selected
    """
    data_per_client = len(train_dataset) // num_clients
    indices = np.random.permutation(len(train_dataset))
    client_data = [indices[i * data_per_client : (i + 1) * data_per_client] for i in range(num_clients)]
    return client_data

def partition_data_non_iid(train_dataset, num_clients, num_shards):
    """
    To create a simple non-iid, sort the data with its target label, and then split in to num_shards shards.
    The data in the same shard are non-iid.
    Then according to the num_clients to randomly distributed the shard to the different client.
    """
    data = train_dataset.data
    targets = train_dataset.targets
    data_sorted_by_label = defaultdict(list) #{label: [index]}
    for idx, label in enumerate(targets):
        data_sorted_by_label[label.item()].append(idx)

    shard_size = len(data) // num_shards
    shards = [[] for _ in range(0, num_shards)]
    cur_fill_shard = 0
    for label, indices in data_sorted_by_label.items():
        for i in range(0, len(indices)):
            if len(shards[cur_fill_shard]) >= shard_size:
                cur_fill_shard += 1
            shards[cur_fill_shard].append(indices[i])

    np.random.shuffle(shards)
    assert(len(shards) == num_shards)
    assert(reduce(lambda a, b: a + b, map(lambda x: len(x), shards), 0) == len(train_dataset))

    client_data = [[] for _ in range(num_clients)]
    for i, shard in enumerate(shards):
        client_data[i % num_clients].extend(shard)
    assert(reduce(lambda a, b: a + b, map(lambda x: len(x), client_data), 0) == len(train_dataset))
    return client_data

def calculate_label_distribution(targets, client_data):
    label_distribution = defaultdict(lambda: np.zeros(10, dtype=int))  # 10 classes for MNIST
    for client_id, indices in enumerate(client_data):
        for idx in indices:
            label = targets[idx].item()
            label_distribution[client_id][label] += 1
    return label_distribution

def plot_label_distribution(label_distribution, title, outfile):
    clients = list(label_distribution.keys())
    labels = list(range(10))  # MNIST has 10 classes

    distribution_matrix = np.array([label_distribution[client] for client in clients])
    distribution_matrix = distribution_matrix / distribution_matrix.sum(axis=1, keepdims=True)  # Normalize

    fig, ax = plt.subplots(figsize=(10, 6))
    bar_width = 0.85
    bottom = np.zeros(len(clients))

    for label in labels:
        ax.barh(clients, distribution_matrix[:, label], bar_width, left=bottom, label=f'Class {label}')
        bottom += distribution_matrix[:, label]

    ax.set_xlabel('Class distribution')
    ax.set_ylabel('Client')
    ax.set_title(title)
    ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.05))
    plt.savefig(outfile)
    plt.close()

def load_data(partition_id: int, train_conf: Dict[str, Any]):

    t = Compose([ToTensor()])
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=t)
    test_dataset = datasets.MNIST('./data', train=False, transform=t)

    if train_conf["non_iid"]:
        client_data = partition_data_non_iid(train_dataset, train_conf["num_clients"], train_conf["num_shards"])
    else:
        client_data = partition_data_iid(train_dataset, train_conf["num_clients"])

    prefix = os.getenv("PREFIX_PATH")
    if prefix == None: prefix = "./"
    output_file = os.path.join(prefix, "data distribution: "+train_conf["description"]+".png")
    plot_label_distribution(calculate_label_distribution(train_dataset.targets, client_data), train_conf["description"], output_file)
    train_indices = client_data[partition_id]
    train_dataset_subset = Subset(train_dataset, train_indices)
    # check the sharding is correct
    assert(torch.equal(train_dataset_subset[0][0], train_dataset[train_indices[0]][0]))
    assert(train_dataset_subset[0][1] == train_dataset[train_indices[0]][1])
    train_loader = DataLoader(train_dataset_subset, batch_size=train_conf["batchsize"], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=train_conf["batchsize"])

    return train_loader, test_loader


def train(
    net: Net,
    train_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    train_conf: Dict[str, Any]
) -> None:
    """Train the network."""

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    # optimizer = torch.optim.Adam(net.parameters(), lr = 0.001);

    net.to(train_conf["device"])
    net.train()
    result = {"accuracy": []}
    for epoch in range(train_conf["epochs"]):  
        for batch_idx, data in enumerate(train_loader, 0):
            images = data[0].to(train_conf["device"])
            labels = data[1].to(train_conf["device"])

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()


            # print statistics every 10 batches
            if batch_idx % train_conf["log_interval"] == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(images), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))
        epoch_loss, epoch_acc = test(net, test_loader, train_conf)
        result["accuracy"].append([epoch, epoch_acc])
    return result


def test(
    net: Net,
    test_loader: torch.utils.data.DataLoader,
    train_conf: Dict[str, Any]
) -> Tuple[float, float]:
    """Validate the network on the entire test set."""

    criterion = nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    net.to(train_conf["device"])
    net.eval()
    with torch.no_grad():
        for data in test_loader:
            images = data[0].to(train_conf["device"])
            labels = data[1].to(train_conf["device"])
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)  # pylint: disable=no-member
            correct += (predicted == labels).sum().item()
    accuracy = correct / len(test_loader.dataset)
    print("Test Accuracy: {}, Test Loss: {}".format(accuracy, loss))
    return loss, accuracy


def main():
    parser = argparse.ArgumentParser(description='Training Configuration')
    parser.add_argument('--train-conf', type=str, required=True, help='Path to the configuration file')
    args = parser.parse_args()
    conf = load_train_conf(args.train_conf)

    print("Centralized PyTorch training")
    print("Load data")
    print("DEVICE: ", conf["device"])

    train_loader, test_loader = load_data(0, conf)
    net = Net().to(conf["device"])
    net.eval()
    print("Start training")
    result = train(net, train_loader, test_loader, conf)
    prefix = os.getenv("PREFIX_PATH")
    if prefix == None: prefix = "./"
    result_path = os.path.join(prefix, "centralized_result.json")
    json.dump(result, open(result_path, 'w' ) )


if __name__ == "__main__":
    main()
