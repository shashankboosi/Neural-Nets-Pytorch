#!/usr/bin/env python3
"""
torch_functions.py
Simple torch operations for practice to understand its use cases
"""
import torch


# Simple addition function that accepts two tensors and returns the result.
def simple_addition(x, y):
    return torch.add(x, y)


# Resize tensors
# Function that reshapes the given tensor as the given shape and returns the result.
def simple_reshape(x, shape):
    return x.view(shape)


# A function that flattens the given tensor and returns the result.
def simple_flat(x):
    return x.view(torch.numel(x))


# Transpose and Permutation
"""
    A function that swaps the first dimension and
    the second dimension of the given matrix x and returns the result.
"""


def simple_transpose(x):
    return torch.transpose(x, 0, 1)


"""
    A function that permute the dimensions of the given tensor
    x according to the given order and returns the result.
"""


def simple_permute(x, order):
    return x.permute(order)


# Matrix multiplication (with broadcasting).
"""
    A function that computes the dot product of
    two rank 1 tensors and returns the result.
"""


def simple_dot_product(x, y):
    return torch.dot(x, y)


"""
    A function that performs a matrix multiplication
    of two given rank 2 tensors and returns the result.
"""


def simple_matrix_mul(x, y):
    return torch.mm(x, y)


"""
    A function that computes the matrix product of two tensors and returns the result.
    The function is broadcastable.
"""


def broadcastable_matrix_mul(x, y):
    return torch.matmul(x, y)


# Concatenate and stack.
"""
    A function that concatenates the given sequence of tensors
    in the first dimension and returns the result
"""


def simple_concatenate(tensors):
    return torch.cat(tensors, 0)


"""
    A function that concatenates the given sequence of tensors
    along a new dimension(dim) and returns the result.
"""


def simple_stack(tensors, dim):
    return torch.stack(tensors, dim)
