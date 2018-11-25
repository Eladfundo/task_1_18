
# Homecoming (eYRC-2018): Task 1A
# Build a Fully Connected 2-Layer Neural Network to Classify Digits

# NOTE: You can only use Tensor API of PyTorch

from nnet import model

# TODO: import torch and torchvision libraries
# We will use torchvision's transforms and datasets
import torch 
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data.dataloader as dataloader
import torch.optim as optim
import matplotlib.pyplot as plt

#torchvision packages
from torch.utils.data import TensorDataset
from torch.autograd import Variable
from torchvision import transforms
from torchvision.datasets import MNIST



# TODO: Defining torchvision transforms for preprocessing
transforms=transforms.Compose([transforms.ToTensor(),])# ToTensor does min-max normalization. 

# TODO: Using torchvision datasets to load MNIST
#Getting the dataset transformed into 1d tensor
train = MNIST('./data', train=True, download=True, transform= transforms, )

#Getting the test dataset transformed into 1d tensor with label
test = MNIST('./data', train=False, download=True, transform= transforms, )


# TODO: Use torch.utils.data.DataLoader to create loaders for train and test
# NOTE: Use training batch size = 4 in train data loader.
train_data_loader = dataloader.DataLoader(train,batch_size=4,shuffle=True)

test_data_loader = dataloader.DataLoader(test,batch_size=4,shuffle=True)


# NOTE: Don't change these settings
device = "cuda:0" if torch.cuda.is_available() else "cpu"
device="cpu"

# NOTE: Don't change these settings
# Layer size
N_in = 28 * 28 # Input size
N_h1 = 256 # Hidden Layer 1 size
N_h2 = 256 # Hidden Layer 2 size
N_out = 10 # Output size
# Learning rate
lr = 0.01


# init model
net = model.FullyConnected(N_in, N_h1, N_h2, N_out, device=device)

# TODO: Define number of epochs
N_epoch = 5 # Or keep it as is


# TODO: Training and Validation Loop
# >>> for n epochs
## >>> for all mini batches
### >>> net.train(...)
## at the end of each training epoch
## >>> net.eval(...)
outputs_arr,accuracy_arr,creloss_arr=[],[],[]
for epoch_index in range(N_epoch):
        for batch_idx,(inputs,label) in enumerate(train_data_loader): 
                creloss, accuracy, outputs=net.train(inputs,label)
                accuracy_arr.append(accuracy)
                creloss_arr.append(creloss)
                #print(net.crel(outputs,label))
                """
                if batch_idx % 100 == 1 and batch_idx * len(inputs)==len(train_data_loader.dataset) :
                        print('\r Train Epoch: {}/{} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch_index+1,
                        N_epoch,
                        batch_idx * len(inputs), 
                        len(train_data_loader.dataset),
                        100. * batch_idx / len(train), 
                        creloss), 
                        end='')"""
        num=sum(accuracy_arr)
        accuracyf=num/len(accuracy_arr)
        print('\r Train Epoch: {}/{} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\t Test Accuracy: {:.4f}%'.format(
        epoch_index+1,
        N_epoch,
        len(train_data_loader.dataset), 
        len(train_data_loader.dataset),
        100. * batch_idx / len(train_data_loader), 
        creloss,
        accuracyf*100,
        end=''))  
print("End of train")
plt.plot(creloss_arr)   
plt.show()
        
# TODO: End of Training
# make predictions on randomly selected test examples
# >>> net.predict(...)

