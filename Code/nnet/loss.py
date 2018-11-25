
# NOTE: You can only use Tensor API of PyTorch

import torch
import torch.nn as nn
import torch.nn.functional as F
#import activation 


# Extra TODO: Document with proper docstring
def cross_entropy_loss(outputs, labels):
    """Calculates cross entropy loss given outputs and actual labels

    """  
    labels=labels.long()
    loss = F.cross_entropy(outputs,labels)
    creloss = loss
    return creloss.item()   # should return float not tensor

# Extra TODO: Document with proper docstring
def delta_cross_entropy_softmax(outputs, labels):
    """Calculates derivative of cross entropy loss (C) w.r.t. weighted sum of inputs (Z). 
    
    """
    """
    labels=labels.long()
    loss = F.cross_entropy(outputs,labels)
    loss.backward()
    """
    labels=labels.long()
    inputs_size_arr=list(outputs.size())
    batch_size=inputs_size_arr[0]
    m=batch_size
    grad=F.softmax(outputs,dim=0)
    grad[range(m),labels]-=1
    avg_grads=grad/m
    return avg_grads

def delta_sigmoid(z):
    
    sig=torch.sigmoid(z)
    rhs=1-sig
    ans=sig*rhs
    
    return ans

if __name__ == "__main__":
    pass
#local Testing BLock
"""
output = torch.randn(4, 10).float()
print("o/p=",output)
target = torch.Tensor([5, 4, 7, 5])
print("target=",target.size())
#print(delta_cross_entropy_softmax(output,target))
print(delta_cross_entropy_softmax(output,target).size())
#print(delta_sigmoid(output).size())
"""