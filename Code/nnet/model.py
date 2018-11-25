
# NOTE: You can only use Tensor API of PyTorch

import math
import torch
import torch.nn as nn

#Other modules



class FullyConnected:
    """Constructs the Neural Network architecture.

    Args:
        N_in (int): input size
        N_h1 (int): hidden layer 1 size
        N_h2 (int): hidden layer 2 size
        N_out (int): output size
        device (str, optional): selects device to execute code. Defaults to 'cpu'
    
    Examples:
        >>> network = model.FullyConnected(2000, 512, 256, 5, device='cpu')
        >>> creloss, accuracy, outputs = network.train(inputs, labels)
    """

    def __init__(self, N_in, N_h1, N_h2, N_out, device='cpu'):
        """Initializes weights and biases, and construct neural network architecture.
        
        One [recommended](http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf) approach is to initialize weights randomly but uniformly in the interval from [-1/n^0.5, 1/n^0.5] where 'n' is number of neurons from incoming layer. For example, number of neurons in incoming layer is 784, then weights should be initialized randomly in uniform interval between [-1/784^0.5, 1/784^0.5].
        
        You should maintain a list of weights and biases which will be initalized here. They should be torch tensors.

        Optionally, you can maintain a list of activations and weighted sum of neurons in a dictionary named Cache to avoid recalculation of those. If tensors are too large it could be an issue.
        """
        self.N_in = N_in
        self.N_h1 = N_h1
        self.N_h2 = N_h2
        self.N_out = N_out

        self.device = torch.device(device)
        #Here w1 represent the weight of the N
        w1 = wbg.weight_initialiser(N_in,N_h1,device=device)
        w2 = wbg.weight_initialiser(N_h1,N_h2,device=device)
        w3 = wbg.weight_initialiser(N_h2,N_out,device=device)
        self.weights = {'w1': w1, 'w2': w2, 'w3': w3}

        b1 = wbg.bias_initialiser(N_h1,device=device)
        b2 = wbg.bias_initialiser(N_h2,device=device)
        b3 = wbg.bias_initialiser(N_out,device=device)
        self.biases = {'b1': b1, 'b2': b2, 'b3': b3}

        self.cache = {'z1': "Not_def", 'z2': "Not_def" ,'z3': "Not_def"}

    # TODO: Change datatypes to proper PyTorch datatypes
    def train(self, inputs, labels, lr=0.001, debug=False):
        """Trains the neural network on given inputs and labels.

        This function will train the neural network on given inputs and minimize the loss by backpropagating and adjusting weights with some optimizer.

        Args:
            inputs (torch.tensor): inputs to train neural network. Size (batch_size, N_in) 
            labels (torch.tensor): correct labels. Size (batch_size)
            lr (float, optional): learning rate for training. Defaults to 0.001
            debug (bool, optional): prints loss and accuracy on each update. Defaults to False

        Returns:
            creloss (float): average cross entropy loss
            accuracy (float): ratio of correctly classified to total samples
            outputs (torch.tensor): predictions from neural network. Size (batch_size, N_out)
        """
        outputs =self.forward(inputs) # forward pass
        creloss = loss.cross_entropy_loss(outputs,labels)# calculate loss
        
        accuracy =self.accuracy(outputs,labels) # calculate accuracy
        
        if debug:
            print('loss: ', creloss)
            print('accuracy: ', accuracy)
        #"train dwn ,dbn"
        dw1, db1, dw2, db2, dw3, db3 = self.backward(inputs,labels,outputs)
        # Use below optimizer.mbgd(self.weights,self.biases,dw1, db1, dw2, db2, dw3, db3,lr)
        self.weights, self.biases = optimizer.mbgd(self.weights,self.biases,dw1, db1, dw2, db2, dw3, db3,lr)
        return creloss, accuracy, outputs

    def predict(self, inputs):
        """Predicts output probability and index of most activating neuron

        This function is used to predict output given inputs. You can then use index in classes to show which class got activated. For example, if in case of MNIST fifth neuron has highest firing probability, then class[5] is the label of input.

        Args:
            inputs (torch.tensor): inputs to train neural network. Size (batch_size, N_in) 

        Returns:
            score (torch.tensor): max score for each class. Size (batch_size)
            idx (torch.tensor): index of most activating neuron. Size (batch_size)  
        """
        outputs = self.forward(inputs)# forward pass
        score, idx =torch.max(outputs,1) # find max score and its index
        print("outputs",outputs)
        print("score",score)
        print("idx",idx)

        return score, idx

    def eval(self, inputs, labels, debug=False):
        """Evaluate performance of neural network on inputs with labels.

        This function is used to evaluate loss and accuracy of neural network on new examples. Unlike predict(), this function will not only predict but also calculate and return loss and accuracy w.r.t given inputs and labels.

        Args:
            inputs (torch.tensor): inputs to train neural network. Size (batch_size, N_in) 
            labels (torch.tensor): correct labels. Size (batch_size)
            debug (bool, optional): print loss and accuracy on every iteration. Defaults to False

        Returns:
            loss (float): average cross entropy loss
            accuracy (float): ratio of correctly to uncorrectly classified samples
            outputs (torch.tensor): predictions from neural network. Size (batch_size, N_out)
        """
        outputs =self.forward(inputs) # forward pass
        creloss = loss.cross_entropy_loss(outputs,labels)# calculate loss
        accuracy = self.accuracy(inputs,labels)# calculate accuracy

        if debug:
            print('loss: ', creloss)
            print('accuracy: ', accuracy)
            
        return creloss, accuracy, outputs

    def accuracy(self, outputs, labels):
        """Accuracy of neural network for given outputs and labels.
        
        Calculates ratio of number of correct outputs to total number of examples.

        Args:
            outputs (torch.tensor): outputs predicted by neural network. Size (batch_size, N_out)
            labels (torch.tensor): correct labels. Size (batch_size)
        
        Returns:
            accuracy (float): accuracy score 
        """
        batch_size=wbg.batch_size_calc(outputs)
        correct_score=0
        score, idx =torch.max(outputs,1)
        equality_tensor=torch.eq(idx,labels)
        non_zero_tensor=torch.nonzero(equality_tensor)
        #print(idx,labels)
        #Eprint(equality_tensor)
        #print("Non-zero",non_zero_tensor)
        try:
            eqt=len(list(non_zero_tensor))
            #eqt=torch.max(non_zero_tensor).item()
        except RuntimeError:
            return 0
        accuracy =eqt /batch_size
        return accuracy

    def forward(self, inputs):
        """Forward pass of neural network

        Calculates score for each class.

        Args:
            inputs (torch.tensor): inputs to train neural network. Size (batch_size, N_in) 

        Returns:
            outputs (torch.tensor): predictions from neural network. Size (batch_size, N_out)
        """
        batch_size=wbg.batch_size_calc(inputs)
        #print("Empty outputs size",outputs.size())
        count=1

        for input_tensor in inputs:
            #print("input layer")
            input_matrix_tensor=torch.reshape(input_tensor,(784,1))
            #print("hl1")
            z1 = self.weighted_sum(input_matrix_tensor,self.weights['w1'],self.biases['b1'])
            a1 = activation.sigmoid(z1)
           # print("hl2")
            z2= self.weighted_sum(a1,self.weights['w2'],self.biases['b2'])
            a2 = activation.sigmoid(z2)
            #print("output_layer")
            z3 = self.weighted_sum(a2,self.weights['w3'],self.biases['b3'])
            outputs_element = activation.softmax(z3)
            if count ==1:
                #print("1st pass in batch")
                outputs=outputs_element
                z1_torch=z1
                z2_torch=z2
                z3_torch=z3
            else:
                outputs=torch.cat((outputs,outputs_element),1)
                z1_torch=torch.cat((z1_torch,z1),1)
                z2_torch=torch.cat((z2_torch,z2),1)
                z3_torch=torch.cat((z3_torch,z3),1)
            count=count+1
        outputs=torch.reshape(outputs,(batch_size,self.N_out))
        self.cache['z1']=torch.transpose(z1_torch, 0, 1)
        self.cache['z2']=torch.transpose(z2_torch, 0, 1)
        self.cache['z3']=torch.transpose(z3_torch, 0, 1)
        #print("Foward pass z2 size",self.cache['z3'].size())
        return outputs

    def weighted_sum(self, X, w, b):
        """Weighted sum at neuron
        
        Args:
            X (torch.tensor): matrix of Size (K, L)
            w (torch.tensor): weight matrix of Size (J, L)
            b (torch.tensor): vector of Size (J)

        Returns:
            result (torch.tensor): w*X + b of Size (K, J)
        """
        mul=torch.mm(w,X)#porduct component
        #print("b size",b.size())
        #print("X size",X.size())
        #print("W size",w.size())
        #print("mul size",mul.size())
        result=b+mul    
        return result

    def backward(self, inputs, labels, outputs):
        """Backward pass of neural network
        
        Changes weights and biases of each layer to reduce loss
        
        Args:
            inputs (torch.tensor): inputs to train neural network. Size (batch_size, N_in) 
            labels (torch.tensor): correct labels. Size (batch_size)
            outputs (torch.tensor): outputs predicted by neural network. Size (batch_size, N_out)
        
        Returns:
            dw1 (torch.tensor): Gradient of loss w.r.t. w1. Size like w1
            db1 (torch.tensor): Gradient of loss w.r.t. b1. Size like b1
            dw2 (torch.tensor): Gradient of loss w.r.t. w2. Size like w2
            db2 (torch.tensor): Gradient of loss w.r.t. b2. Size like b2
            dw3 (torch.tensor): Gradient of loss w.r.t. w3. Size like w3
            db3 (torch.tensor): Gradient of loss w.r.t. b3. Size like b3
        """
        # Calculating derivative of loss w.r.t weighted sum
        dout = loss.delta_cross_entropy_softmax(outputs,labels)#Size(batch_size,dim of N_out)
        
        d2 = torch.mm(dout,self.weights['w3'])*loss.delta_sigmoid(self.cache['z2']) #d2 (torch.tensor): error at hidden layer 2. Size like a2 (or z2)
        d1 = torch.mm(d2,self.weights['w2'])*loss.delta_sigmoid(self.cache['z1'])
        """
        d2 = torch.mm(dout,self.weights['w3'])*self.cache['z2'] #d2 (torch.tensor): error at hidden layer 2. Size like a2 (or z2)
        d1 = torch.mm(d2,self.weights['w2'])*(self.cache['z1'])
        """
        #print("dout",dout.size())
        #print("d2",d2.size())
        #print("d1",d1.size())
        dw1, db1, dw2, db2, dw3, db3 = self.calculate_grad(inputs, d1, d2, dout)# calculate all gradients
        return dw1, db1, dw2, db2, dw3, db3

    def calculate_grad(self, inputs, d1, d2, dout):
        """Calculates gradients for backpropagation
        
        This function is used to calculate gradients like loss w.r.t. weights and biases.

        Args:
            inputs (torch.tensor): inputs to train neural network. Size (batch_size, N_in) 
            dout (torch.tensor): error at output. Size like aout or a3 (or z3)
            d2 (torch.tensor): error at hidden layer 2. Size like a2 (or z2)
            d1 (torch.tensor): error at hidden layer 1. Size like a1 (or z1)

        Returns:
            dw1 (torch.tensor): Gradient of loss w.r.t. w1. Size like w1
            db1 (torch.tensor): Gradient of loss w.r.t. b1. Size like b1
            dw2 (torch.tensor): Gradient of loss w.r.t. w2. Size like w2
            db2 (torch.tensor): Gradient of loss w.r.t. b2. Size like b2
            dw3 (torch.tensor): Gradient of loss w.r.t. w3. Size like w3
            db3 (torch.tensor): Gradient of loss w.r.t. b3. Size like b3
        """
        #"notdw1","notdb1","notdw2","notdb2","notdw3","notdb3"
        
        m=wbg.batch_size_calc(inputs)
        trnasformed_input=torch.reshape(inputs,(m,784))
        """
        dw3 = (torch.mm(torch.transpose(dout, 0, 1),torch.sigmoid(self.cache['z2'])))/m
        dw2 = (torch.mm(torch.transpose(d2, 0, 1),torch.sigmoid(self.cache['z1'])))/m
        dw1 = (torch.mm(torch.transpose(d1, 0, 1),trnasformed_input))/m
        """

        dw3 = (torch.mm(torch.transpose(dout, 0, 1),self.cache['z2']))/m
        dw2 = (torch.mm(torch.transpose(d2, 0, 1),self.cache['z1']))/m
        dw1 = (torch.mm(torch.transpose(d1, 0, 1),trnasformed_input))/m

        db1t=  (torch.sum(d1,dim=0))/m
        db2t= (torch.sum(d2,dim=0))/m
        db3t= (torch.sum(dout,dim=0))/m
        
        db1=torch.reshape(db1t,(256,1))
        db2=torch.reshape(db2t,(256,1))
        db3=torch.reshape(db3t,(10,1))

        #print("dw1",dw1.size())
        #print("dw2",dw2.size())
        #print("dw3",dw3.size())
        #print("db1",db1.size())
        #print("db2",db2.size())
        #print("db3",db3.size())
        return dw1, db1, dw2, db2, dw3, db3
    #Delete this later
    def crel(self,outputs,labels):
        return loss.cross_entropy_loss(outputs,labels)
    def del_crel(self,outputs,labels,dterm):
        return loss.delta_cross_entropy_softmax(outputs,labels,dterm)



if __name__ == "__main__":
    import activation, loss, optimizer,wbg
else:
    from nnet import activation, loss, optimizer,wbg

