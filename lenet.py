import torch
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
from src.LeNet import *
import sys
import os
from tqdm import tqdm


def dataset_prep():
    transform = transforms.Compose([  # done to appropriate convert the data into tensor and normalize
        transforms.ToTensor(),
        transforms.Normalize(
            (0.5, 0.5, 0.5),
            (0.5, 0.5, 0.5)
        )
    ])

    trainset = torchvision.datasets.CIFAR10(root='./cifar-10-python/', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./cifar-10-python/', train=False, download=True, transform=transform)
    return trainset, testset


def train(model, optimizer):
    model.train()
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    train_losses = []  # all losses for each epoch
    train_accuracies = []  # all accuracies for each epoch

    for epoch in range(no_epochs):
        with tqdm(trainloader, unit="batch") as tepoch:
            tepoch.set_description(f"Epoch {epoch}")

            training_loss = 0.0  # total loss for all images in this epoch
            correct_predictions = 0
            total_no_labels = 0
            for (images, labels) in trainloader:  # trainloader will give images whose number is equal to batch_size
                images, labels = images.to(device), labels.to(device)

                optimizer.zero_grad()  # initialise all grads to zero

                outputs = model(images)  # forward pass
                loss = criterion(outputs, labels)
                loss.backward()  # backprop
                optimizer.step()  # update

                training_loss += loss.item()
                total_no_labels += labels.size(0)
                predicted = torch.argmax(outputs, 1)  # do argmax to get index of the max in softmax
                correct_predictions += (predicted == labels).sum().item()  # correct markings

            avg_loss = training_loss / len(trainloader)  # len(trainloader) is iteration no
            train_accuracy = (
                                         correct_predictions / total_no_labels) * 100  # ration of correct labels and total no of labels

            train_losses.append(avg_loss)
            train_accuracies.append(train_accuracy)
            tepoch.set_postfix(loss=avg_loss, accuracy=train_accuracy)
    return train_losses, train_accuracies


def test(model):
    model.eval()  # model in testing mode
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_losses = []  # test losses
    test_accuracies = []  # test accuracies

    for epoch in range(no_epochs):
        with tqdm(testloader, unit="batch") as tepoch:
            tepoch.set_description(f"Epoch {epoch}")
            test_loss = 0.0
            correct_val = 0
            with torch.no_grad():
                for images, labels in testloader:
                    images, labels = images.to(device), labels.to(device)

                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    test_loss += loss.item()

                    predicted = torch.argmax(outputs, 1)
                    correct_val += (predicted == labels).sum().item()

            avg_test_loss = test_loss / len(testloader)
            test_accuracy = (correct_val / 10000) * 100

            test_losses.append(avg_test_loss)
            test_accuracies.append(test_accuracy)
            tepoch.set_postfix(loss=avg_test_loss, accuracy=test_accuracy)
    return test_losses, test_accuracies


def lenet(device, lr, batch_norm=False):
    model = UnnormalisedLeNet() if batch_norm else NormalisedLeNet()
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)  # selecting adam optimizer
    train_loss, train_acc = train(model, optimizer)
    test_loss, test_acc = test(model)
    return train_loss, train_acc, test_loss, test_acc


if __name__ == '__main__':
    lr = int(sys.argv[1])
    no_epochs = int(sys.argv[2])
    batch_size = int(sys.argv[3])
    batch_norm = (sys.argv[4] == 'yes')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    no_train_batches, no_test_batches = np.ceil(50000 / batch_size), np.ceil(10000 / batch_size)
    num_workers = max(2, os.cpu_count() if device.type == 'cpu' else torch.cuda.device_count())
    criterion = nn.CrossEntropyLoss()
    trainset, testset = dataset_prep()

    if batch_norm in ['yes', 'no']:
        train_loss,train_acc, test_loss, test_acc = lenet(device, lr, batch_norm)
