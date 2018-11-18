
# NOTE: You can only use Tensor API of PyTorch

import torch
import torch.nn as nn


# Extra TODO: Document with proper docstring
def cross_entropy_loss(outputs, labels):
    """Calculates cross entropy loss given outputs and actual labels

    """  
    loss = nn.CrossEntropyLoss()  
    creloss = loss(outputs,labels)
    return creloss.item()   # should return float not tensor

# Extra TODO: Document with proper docstring
def delta_cross_entropy_softmax(outputs, labels):
    """Calculates derivative of cross entropy loss (C) w.r.t. weighted sum of inputs (Z). 
    
    """
    der=torch.tensor(1.0)
    der.requires_grad_(True)
    outputs.requires_grad_(True)
    loss = nn.CrossEntropyLoss()
    creloss = loss(outputs,labels)
    creloss.backward(der)
    avg_grads = der.grad
    return avg_grads

if __name__ == "__main__":
    pass
#local Testing BLock
"""
output = torch.randn(10, 120).float()
print("o/p=",output.size())
target = torch.FloatTensor(10).uniform_(0, 120).long()
print("target=",target)
print(delta_cross_entropy_softmax(output,target))
"""