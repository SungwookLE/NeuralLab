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
    def input_data(seq, windowSize, batchSize=1, shuffle=True):
        out = list()
        L = len(seq)

        
        for i in range(L-windowSize):
            window = seq[i:i+windowSize]
            label = seq[i+windowSize:i+windowSize+1]
            out.append((window, label))

        out = np.array([out], dtype=object)
        out = out.reshape(-1, batchSize, 2)

        if shuffle:
            np.random.shuffle(out)

        return out


class RNNs(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, output_size=1):
        super(RNNs, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.rnn = nn.RNN(input_size=self.input_size , hidden_size=hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, seq):
        rnn_out, self.hidden = self.rnn(seq.view(len(seq[0]), -1, self.input_size), self.hidden)
        pred = self.fc(rnn_out.view(len(seq[0]), -1, self.hidden_size))
        return pred[-1]

    def initHidden(self, batch_size):
        self.hidden = torch.zeros(1, batch_size, self.hidden_size).to(device)


def train(model, optim, criterion, dataloader):
    batchSize = dataloader.shape[1]
    size = len(dataloader) *batchSize


    totalLoss = 0
    for batch, (data) in enumerate(dataloader):
        X = torch.tensor(data.T[0].tolist()).float().to(device)
        y = torch.tensor(data.T[1].tolist()).float().to(device)
        
        model.initHidden(batch_size=batchSize)
        pred = model(X)

        #backpropagation
        optim.zero_grad()
        loss = criterion(pred, y)
        loss.backward()
        optim.step()

        totalLoss += loss.item()

        if (batch % 80 == 0):
            print(f"Loss: {loss.item():.9f}, {batch * batchSize}/{size}")

    return totalLoss / batchSize


if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device is {device}")

    # 1. Data Preparing
    # 1-1. Data preparing (TimeSeriest using Sin wave)
    dataSteps = 1000
    X_all = np.linspace(start=0, stop=999, num=dataSteps)
    Y_all = np.sin(X_all*2*np.pi/80)/dataSteps*(X_all-dataSteps/2)

    # 1-2. Orgainization: Train/Test
    test_size = 100
    X_train = X_all[:-test_size]
    Y_train = Y_all[:-test_size]
    X_test = X_all[-test_size:]
    Y_test = Y_all[-test_size:]

    # 1-3. Make data as sequence with moving window
    window_size = 40
    batchSize = 5
    train_dataloader = Preprocessing.input_data(
        Y_train, window_size, batchSize=batchSize, shuffle=True)
    print(f"data shape(EA, batch_size, 2) is {train_dataloader.shape}")

    # 2. Modeling
    model = RNNs(input_size=1, hidden_size=50).to(device)
    print(model)

    # 3. Training
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.15)

    epochs = 51
    for e in range(epochs):
        print(f"Epoch {e+1}")
        model.train()
        totalLoss = train(model, optimizer, criterion, train_dataloader)
        print(f"totalLoss: {totalLoss}")

        preds = torch.Tensor(Y_train)
        for f in range(test_size):
            seq = preds[-window_size:].unsqueeze(0).to(device)
            
            model.eval()
            with torch.no_grad():
                model.initHidden(batch_size=1)
                pred = model(seq).item()
                preds = torch.cat( (preds, torch.tensor([pred])))

        totalLoss = criterion(preds[-test_size:], torch.Tensor(Y_test))
        print(f"Performance on test range: {totalLoss}\n----------------")

    # 4. Evaluation
        if (e % 10 == 0):
            fig, (ax1, ax2) = plt.subplots(1, 2)
            ax1.set_xlim(-10, 1010)
            ax1.grid()
            ax1.plot(X_train, Y_train, color='#8000ff',  label='train')
            ax1.plot(X_test, Y_test, color="#ff8000",
                     linestyle='--', label='test')
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