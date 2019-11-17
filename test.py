import torch
import string

all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)


def letterToIndex(letter):
    return all_letters.find(letter)


def letterToTensor(letter):
    tensor = torch.zeros(1, n_letters)
    tensor[0][letterToIndex(letter)] = 1
    return tensor

print(letterToTensor(letterToTensor('A')))