"""fl0: A Flower / pytorch_msg_api app."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor
from datasets import load_dataset
from flwr.app import ArrayRecord, MetricRecord
from torchvision import models
from fl0.net import SimpleMedicalNet
resnet = models.resnet18(pretrained=True)



#fedrated dataset
fds = None  # Cache FederatedDataset

pytorch_transforms = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


def apply_transforms(batch):
    """Apply transforms to the partition from FederatedDataset."""
    batch["image"] = [pytorch_transforms(image) for image in batch["image"]]
    return batch


def load_data(partition_id: int, num_partitions: int):
    """Load partition CIFAR10 data."""
    # Only initialize `FederatedDataset` once
    global fds
    if fds is None:
        partitioner = IidPartitioner(num_partitions=num_partitions)
        fds = FederatedDataset(
            # dataset="uoft-cs/cifar10",
            # dataset="chenqitao/BreakHis",
            dataset = "chenqitao/BreakHis",
            partitioners={"train": partitioner},
        )
    partition = fds.load_partition(partition_id)
    # Divide data on each node: 80% train, 20% test
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)
    # Construct dataloaders
    partition_train_test = partition_train_test.with_transform(apply_transforms)
    trainloader = DataLoader(partition_train_test["train"], batch_size=32, shuffle=True)
    testloader = DataLoader(partition_train_test["test"], batch_size=32)
    return trainloader, testloader


def train(net, trainloader, epochs, lr, device):
    """Train the model on the training set."""
    net.to(device)  # move model to GPU if available
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    net.train()
    running_loss = 0.0
    for _ in range(epochs):
        for batch in trainloader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
    avg_trainloss = running_loss / len(trainloader)
    return avg_trainloss


def test(net, testloader, device):

    """Validate the model on the test set."""
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    with torch.no_grad():
        for batch in testloader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    loss = loss / len(testloader)
    return loss, accuracy

def central_evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
    """Evaluate model on the server side."""

    # Load the model and initialize it with the received weights
    #加载训练模型，并初始化从客户端接受的权重
    model = SimpleMedicalNet()
    # model = resnet
    model.load_state_dict(arrays.to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load the entire CIFAR10 test dataset
    # It's a huggingface dataset, so we can load it directly and apply transforms
    cifar10_test = load_dataset("chenqitao/BreakHis", split="test")

    pytorch_transforms = Compose(
        [ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    # Define transforms and construct DataLoader for the test set
    def apply_transforms(batch):
        batch["image"] = [pytorch_transforms(image) for image in batch["image"]]
        return batch

    testset = cifar10_test.with_transform(apply_transforms)
    testloader = DataLoader(testset, batch_size=64)

    # Evaluate the model on the test set
    loss, accuracy = test(model, testloader, device)

    # Return the evaluation metrics
    return MetricRecord({"accuracy": accuracy, "loss": loss})

