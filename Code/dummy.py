#Standard packages

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data.dataloader as dataloader
import torch.optim as optim

#torchvision packages
from torch.utils.data import TensorDataset
from torch.autograd import Variable
from torchvision import transforms
from torchvision.datasets import MNIST

#nnet packages to be used
#from nnet import weight_bias_generator as wbg
from nnet import model

#The Transformation parameter to tensor
transforms=transforms.Compose([transforms.ToTensor(),])# ToTensor does min-max normalization. 

#Getting the dataset transformed into 1d tensor
train = MNIST('./data', train=True, download=True, transform= transforms, )

#Getting the test dataset transformed into 1d tensor with label
test = MNIST('./data', train=False, download=True, transform= transforms, )

#Loading the data using DataLoader for better iteration 
train_data_loader = dataloader.DataLoader(train,batch_size=4,shuffle=True)

test_data_loader = dataloader.DataLoader(test,batch_size=4,shuffle=True)

#Printing the image for dataloader 
def image_printer(dataloader,batch_size):
    """
    Takes a dataloader object and display the image and
    Args:
        dataloader:(torch.utils.data.DataLoader)
    Returns:
        Nothing 
    """
    for i in train_data_loader:
        img=i[0]
        label=i[1:]
        img_np_array=img.numpy()

        plt.title("label is {label}".format(label=label))
        img_np_array=img_np_array.reshape((28,28))
        plt.imshow(img_np_array,cmap='gray')
        plt.show()
        break

def image_printer_elemental(img_tensor,label_tensor):
    """
    Takes a dataloader object and display the image and
    Args:
        img_tensor:(torch.tensor)
        label_tensor:(torch.tensor)
    Returns:
        Nothing 
    """
    label=label_tensor
    img_np_array=img_tensor.numpy()
    plt.title("label is {label}".format(label=label))
    img_np_array=img_np_array.reshape((28,28))
    plt.imshow(img_np_array,cmap='gray')
    plt.show()

#Getting the shape of test_data_loader
#image_printer(test_data_loader,1)
#print("Image fuction working")

def bias_initialiser(N_current_layer,device='cpu'):
        """
        Initializes the bias as a tensor.
        The value of the bias will be  b=0
        
        Args:
        N_current_layer:Number of elmemets in current Layer.

        Returns:
        bias: Tensor of filled with 0.
        """
        bias=torch.zeros((N_current_layer,1),requires_grad=True)
        bias=bias.to(device)
        return bias


""" Bathc len code does not work with enumerate
count=1
for i in train_data_loader:
    print(i[1].size())
    batch_len=list(i[1].size())
    print(batch_len[0])
    image_printer_elemental(i[0],i[1],batch_len[0])
    if count == 5:
        break
    count=count+1

"""

net=model.FullyConnected(28*28,256,256,10)


EPOCHS=1
""" TO view images and test the loops
for i in range(EPOCHS):
    for batch_idx,(data,label) in enumerate(train_data_loader):
        for img_tensor,label_tensor in zip(data,label):
            image_printer_elemental(img_tensor,label_tensor)
        break     
"""

for i in range(EPOCHS):
    for batch_idx,(inputs,label) in enumerate(train_data_loader): 
        #print("Inputs shape",inputs.size())
        inputs_size_arr=list(inputs.size())
        batch_size=inputs_size_arr[0]
        #print("Batch size",batch_size)
        op=net.forward(inputs)
        #print("Predict Output")
        #print(net.predict(inputs))
        #print(op)
        print("Accuracy",net.accuracy(op,label))
        
        """
        for inputs_data_tensor in inputs:
            print("inputs_data_tensor",inputs_data_tensor.size())
            input_matrix_tensor=torch.reshape(inputs_data_tensor,(1,784))
            print("Input matrix to foward size",input_matrix_tensor.size())
        break
        """
    break

