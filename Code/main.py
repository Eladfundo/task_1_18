
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

# NOTE: Don't change these settings
# Layer size
N_in = 28 * 28 # Input size
N_h1 = 256 # Hidden Layer 1 size
N_h2 = 256 # Hidden Layer 2 size
N_out = 10 # Output size
# Learning rate
lr = 0.001


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

# TODO: End of Training
# make predictions on randomly selected test examples
# >>> net.predict(...)