import torch
import torch.nn as tnn
import torch.nn.functional as F
import torch.optim as topti
from torchtext import data
from torchtext.vocab import GloVe
from data.imdb_dataloader import IMDB


# Class for creating the neural network.
class Network(tnn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.gru = tnn.GRU(input_size=50, hidden_size=100, num_layers=3, dropout=0.5, bias=True, batch_first=True,
                           bidirectional=True)
        self.fc1 = tnn.Linear(in_features=100 * 2, out_features=64)
        self.fc2 = tnn.Linear(in_features=64, out_features=1)

    def forward(self, input, length):

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        h0 = torch.zeros(3 * 2, input.size(0), 100).to(device)

        output, hn = self.gru(input, h0)

        output = F.relu(self.fc1(output[:, -1, :]))
        output = self.fc2(output)

        return output.squeeze()


class PreProcessing():
    stop_words_list = ["a", "an", "the"]

    def pre(x):
        """Called after tokenization"""
        punctuations_list = ["!", "#", "$", "'", "%", "&", "(", ")", "*", "+", ",", "-", ".", "/", ":", ";", "<", "=",
                             ">", "?", "@", "[", "]", "^", "_", "`", "{", "|", "}", "~"]

        cleaned_tokens = [''.join(letter for letter in tokens if letter not in punctuations_list) for tokens in x]
        return [not_empty for not_empty in cleaned_tokens if not_empty]

    def post(batch, vocab):
        """Called after numericalization but prior to vectorization"""
        return batch

    text_field = data.Field(lower=True, stop_words=stop_words_list, include_lengths=True, batch_first=True,
                            preprocessing=pre, postprocessing=post)


def lossFunc():
    return tnn.BCEWithLogitsLoss()


def main():
    # Use a GPU if available, as it should be faster.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device: " + str(device))

    # Load the training dataset, and create a data loader to generate a batch.
    textField = PreProcessing.text_field
    labelField = data.Field(sequential=False)

    train, dev = IMDB.splits(textField, labelField, train="train", validation="dev")

    textField.build_vocab(train, dev, vectors=GloVe(name="6B", dim=50))
    labelField.build_vocab(train, dev)

    trainLoader, testLoader = data.BucketIterator.splits((train, dev), shuffle=True, batch_size=64,
                                                         sort_key=lambda x: len(x.text), sort_within_batch=True)

    net = Network().to(device)
    criterion = lossFunc()
    optimiser = topti.Adam(net.parameters(), lr=0.001)

    for epoch in range(10):
        running_loss = 0

        for i, batch in enumerate(trainLoader):
            # Get a batch and potentially send it to GPU memory.
            inputs, length, labels = textField.vocab.vectors[batch.text[0]].to(device), batch.text[1].to(
                device), batch.label.type(torch.FloatTensor).to(device)

            labels -= 1

            optimiser.zero_grad()

            # Forward pass through the network.
            output = net(inputs, length)

            loss = criterion(output, labels)

            # Calculate gradients.
            loss.backward()

            # Minimise the loss according to the gradient.
            optimiser.step()

            running_loss += loss.item()

            if i % 32 == 31:
                print("Epoch: %2d, Batch: %4d, Loss: %.3f" % (epoch + 1, i + 1, running_loss / 32))
                running_loss = 0

    num_correct = 0

    # Save mode
    torch.save(net.state_dict(), "./model/model.pth")
    print("Saved model")

    with torch.no_grad():
        for batch in testLoader:
            # Get a batch and potentially send it to GPU memory.
            inputs, length, labels = textField.vocab.vectors[batch.text[0]].to(device), batch.text[1].to(
                device), batch.label.type(torch.FloatTensor).to(device)

            labels -= 1

            # Get predictions
            outputs = torch.sigmoid(net(inputs, length))
            predicted = torch.round(outputs)

            num_correct += torch.sum(labels == predicted).item()

    accuracy = 100 * num_correct / len(dev)

    print(f"Classification accuracy: {accuracy}")


if __name__ == '__main__':
    main()
