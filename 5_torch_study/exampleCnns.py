import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data.dataloader import DataLoader

import torch.nn.functional as F
import torch.optim as optim


class CNNs(nn.Module):
    def __init__(self, numClass):
        super(CNNs, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)

        self.fc1 = nn.Linear(12544, 128)
        self.fc2 = nn.Linear(128, numClass)
        self.flatten = nn.Flatten()

    def forward(self, X):
        X = self.conv1(X)  # (28+2-3)/1+1=28
        X = F.relu(X)
        X = self.dropout1(X)

        X = self.conv2(X)  # (28+2-3)/1+1=28
        X = F.relu(X)
        X = self.pool(X)  # (28-2)/2 + 1 = 14

        X = self.flatten(X)

        X = self.fc1(X)
        X = F.relu(X)
        X = self.dropout2(X)
        X = self.fc2(X)
        output = F.relu(X)

        return output


def train(model, optim, criterion, dataloader):

    size = len(dataloader.dataset)
    batchSize = len(dataloader)

    totalLoss = 0
    for batch, (X, y) in enumerate(dataloader):
        X = X.to(device)
        y = y.to(device)

        pred = model(X)

        #backpropagation
        optim.zero_grad()
        loss = criterion(pred, y)
        loss.backward()
        optim.step()
        totalLoss += loss.item()

        if (batch % 100 == 0):
            print(f"Loss: {loss.item():.3f}, {batch * batch_size}/{size}")

    meanLossPerBatch = totalLoss / batchSize
    return meanLossPerBatch


def test(model, criterion, dataloader):
    size = len(dataloader.dataset)
    batchSize = len(dataloader)

    totalLoss = 0
    correct = 0
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        with torch.no_grad():
            pred = model(X)
            loss = criterion(pred, y)
            totalLoss += loss.item()
            correct += (pred.argmax(1) == y).sum().type(torch.float)

    correct = correct / size
    meanLossPerBatch = totalLoss / batchSize
    return meanLossPerBatch, correct


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Operate under {device}")

    #1. Dastset
    train_data = datasets.FashionMNIST(
        root="./data", train=True, transform=transforms.ToTensor(), download=True)
    test_data = datasets.FashionMNIST(
        root="./data", train=False, transform=transforms.ToTensor(), download=True)
    numClass = len(train_data.classes)

    #1-1. Dataloader
    batch_size = 64
    train_dataloader = DataLoader(
        dataset=train_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(
        dataset=test_data, batch_size=batch_size, shuffle=True)

    # 2. Modeling(CNNs)
    model = CNNs(numClass).to(device)

    # # 3. Train
    epochs = 3
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=model.parameters(), lr=1e-3)

    for epoch in range(epochs):
        print(f"Epoch: {epoch+1}\n--------------------")
        meanLoss = train(model, optimizer, criterion, train_dataloader)
        print(f"meanLoss: {meanLoss}\n")

    # 4. Evaluation
    meanLoss, correct = test(model, criterion, test_dataloader)
    print(
        f"meanLoss: {meanLoss}, Precision: {correct*100}%\n--------------------")

    # 4-1. Visualization
    from matplotlib import pyplot as plt
    import random

    classes = train_data.classes
    idx = random.randint(0, len(test_data))
    X, y = test_data[idx][0], test_data[idx][1]
    X = X.to(device)

    with torch.no_grad():
        pred = model(X.unsqueeze(0))
    plt.imshow(X.to("cpu").squeeze().numpy(), cmap='gray')
    plt.title(f"Pred: {classes[pred.argmax(1)]} Label: {classes[y]}")
    plt.tight_layout()
    plt.show()

    # 5. ETC
    torch.save(model.state_dict(), "model.pth")
    modelLoaded = CNNs(numClass)
    modelLoaded.load_state_dict(torch.load("model.pth"))

    print(modelLoaded)
