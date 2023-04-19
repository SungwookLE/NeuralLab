# Ref(hand implementation): https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html

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
    def __init__(self, input_size, hidden_size, output_size):
        super(RNNs, self).__init__()
        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size+hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size+hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        ouput = self.softmax(output)
        return ouput, hidden
    
    def initHidden(self):
        return torch.zeros(1, self.hidden_size)
    