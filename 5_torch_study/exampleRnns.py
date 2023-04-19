# Ref(using RNN module): https://www.kaggle.com/code/namanmanchanda/rnn-in-pytorch

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd


class Preprocessing:
    @staticmethod
    def input_data(seq, windowSize):
        out = list()
        L = len(seq)

        for i in range(L-windowSize):
            window = seq[i:i+windowSize]
            label = seq[i+windowSize:i+windowSize+1]
            out.append((window, label))
        return out


class RNNs(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, output_size=1):
        super(RNNs, self).__init__()
        self.hidden_size = hidden_size
        self.initHidden()
        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, seq):
        rnn_out, self.hidden = self.rnn(seq.view(len(seq), 1, -1), self.hidden)
        pred = self.fc(rnn_out.view(len(seq), -1))
        return pred[-1]

    def initHidden(self):
        self.hidden = torch.zeros(1, 1, self.hidden_size).to(device)


def train(model, optim, criterion, dataloader):
    size = len(dataloader)
    batchSize = 1

    totalLoss = 0
    for batch, (X, y) in enumerate(dataloader):
        X = X.to(device)
        y = y.to(device)

        model.initHidden()
        pred = model(X)

        #backpropagation
        optim.zero_grad()
        loss = criterion(pred, y)
        loss.backward()
        optim.step()

        totalLoss += loss.item()

    return totalLoss


if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device is {device}")

    # 1. Data Preparing
    # 1-1. Data preparing (TimeSeriest using Sin wave)
    dataSteps = 1000
    X_all = torch.linspace(start=0, end=999, steps=dataSteps)
    Y_all = torch.sin(X_all*2*np.pi/80)/dataSteps*(X_all-dataSteps/2)

    # 1-2. Orgainization: Train/Test
    test_size = 100
    X_train = X_all[:-test_size]
    Y_train = Y_all[:-test_size]
    X_test = X_all[-test_size:]
    Y_test = Y_all[-test_size:]

    # 1-3. Make data as sequence with moving window
    window_size = 40
    train_seq = Preprocessing.input_data(
        Y_train, window_size)  # size = 900 - 40 = 860

    # 2. Modeling
    model = RNNs(hidden_size=50).to(device)
    print(model)

    # 3. Training
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.02)

    epochs = 31
    for e in range(epochs):
        print(f"Epoch {e+1}")
        totalLoss = train(model, optimizer, criterion, train_seq)
        print(f"totalLoss: {totalLoss}")

        preds = Y_train
        for f in range(test_size):
            seq = torch.FloatTensor(preds[-window_size:]).to(device)

            with torch.no_grad():
                model.initHidden()
                pred = model(seq).item()
                preds = torch.cat((preds, torch.FloatTensor([pred])))

        totalLoss = criterion(preds[-test_size:], Y_test)
        print(f"Performance on test range: {totalLoss}\n----------------")

    # 4. Evaluation
        if (e % 10 == 0):
            fig, (ax1, ax2) = plt.subplots(1, 2)
            ax1.set_xlim(-10, 1010)
            ax1.grid()
            ax1.plot(X_train, Y_train.numpy(), color='#8000ff',  label='train')
            ax1.plot(X_test, Y_test.numpy(), color="#ff8000",
                     linestyle='--',  marker='.', label='test')
            ax1.set_title("GT")
            ax1.legend()
            ax2.set_xlim(-10, 1010)
            ax2.grid()
            ax2.plot(X_train, Y_train, color='#8000ff', label="input")
            ax2.plot(X_test, preds[-test_size:], color='#ff8000',
                     linestyle='--',  marker='.', label='predict')
            ax2.legend()
            ax2.set_title(f"Epoch{e+1}: MSE {totalLoss:.3f}")
            plt.tight_layout()
            plt.show()

    # 5. ETC - Save
    torch.save(model.state_dict(), "model.pth")
    modelLoaded = RNNs(hidden_size=50)
    modelLoaded.load_state_dict(torch.load("model.pth"))
