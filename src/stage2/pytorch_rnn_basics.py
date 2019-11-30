#!/usr/bin/env python3
"""
Helper functions for RNN, LSTM
"""

import torch


class rnn(torch.nn.Module):

    def __init__(self):
        super(rnn, self).__init__()

        self.ih = torch.nn.Linear(64, 128)
        self.hh = torch.nn.Linear(128, 128)

    def rnnCell(self, input, hidden):
        """
        Create an Elman RNN cell. The network should takes
              input (inputDim = 64) and the current hidden state
              (hiddenDim = 128), and return the new hidden state.
        """
        tanh_input = self.ih(input) + self.hh(hidden)
        return torch.tanh(tanh_input)

    def forward(self, input):
        hidden = torch.zeros(128)
        """
        Create a model that takes as input
              a sequence of size [seqLength, batchSize, inputDim]
              and passes each input through the rnn sequentially,
              updating the (initally zero) hidden state.
              Returns the final hidden state after the
              last input in the sequence has been processed.
        """
        for e in input:
            hidden = self.rnnCell(e, hidden)
        return hidden


class rnnSimplified(torch.nn.Module):

    def __init__(self):
        super(rnnSimplified, self).__init__()
        """
        PyTorch module such that
              the network defined by this class is equivalent to the
              one defined in class "rnn".
        """
        self.net = torch.nn.RNN(64, 128, batch_first=False)

    def forward(self, input):
        _, hidden = self.net(input)

        return hidden


def lstm(input, hiddenSize):
    """
    Variable input is of size [batchSize, seqLength, inputDim]
    """
    lstm = torch.nn.LSTM(input_size=input.size(2), hidden_size=hiddenSize, batch_first=True)
    return lstm(input)


def conv(input, weight):
    """
    Returns the convolution of input and weight tensors,
          where input contains sequential data.
          The convolution is along the sequence axis.
          input is of size [batchSize, inputDim, seqLength]
    """
    output = torch.nn.functional.conv1d(input=input, weight=weight)
    return output
