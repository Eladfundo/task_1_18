
# NOTE: You can only use Tensor API of PyTorch

import torch
import torch.nn as nn
import torch.nn.functional as F

# Extra TODO: Document with proper docstring
def sigmoid(z):
    """Calculates sigmoid values for tensors

    """
    result =torch.sigmoid(z) 
    return result

# Extra TODO: Document with proper docstring
def delta_sigmoid(z):
    """Calculates derivative of sigmoid function

    """
    z.requires_grad_(True)
    sigmoid_z=torch.sigmoid(z)
    sigmoid_z.backward(z)
    grad_sigmoid = z.grad
    return grad_sigmoid 
    

# Extra TODO: Document with proper docstring
def softmax(x):
    """Calculates stable softmax (minor difference from normal softmax) values for tensors.
    A stable softmax function handles ths issue of over-flow and under-flow.
    This is done by subtracting all the value in the tensor with the largest value in the tensor.

    """
    z=x-max(x)
    stable_softmax = F.softmax(z,dim=0)
    return stable_softmax

#moudule Testing
"""
a=torch.rand((10, 1),requires_grad=True)
print(a)
print(softmax(a))
"""
if __name__ == "__main__":
    pass