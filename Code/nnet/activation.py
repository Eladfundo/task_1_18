
# NOTE: You can only use Tensor API of PyTorch

import torch
import torch.nn as nn

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
    grad_sigmoid = 
    return grad_sigmoid 
    

# Extra TODO: Document with proper docstring
def softmax(x):
    """Calculates stable softmax (minor difference from normal softmax) values for tensors.
    A stable softmax function handles ths issue of over-flow and under-flow.
    This is done by subtracting all the value in the tensor with the largest value in the tensor.

    """
    z=x-max(x)
    temp_softmax= nn.Softmax()#Declaring softmax object
    stable_softmax =temp_softmax(z)
    return stable_softmax

#moudule Testing
a=torch.rand((5, 1),requires_grad=True)
print(a)
print(delta_sigmoid(a))
if __name__ == "__main__":
    pass