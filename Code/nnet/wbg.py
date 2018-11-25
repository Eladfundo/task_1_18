import torch


def weight_initialiser(N_prev_layer,N_current_layer,device='cpu'):
        """
        Initializes the weight in the constructor class as a tensor.
        The value of the weight will be  w=1/sqr_root(N_prev_layer)
        Where U(a,b)=
        
        Args:
        N_prev_layer: Number of element in the previous layer.
        N_current_layer:Number of elmemets in current Layer.

        Returns:
        weight: Tensor of value of weight.
        """
        weight_val = 1.0/(N_prev_layer**0.5)
        #print(weight_val)
        tensor = torch.rand((N_current_layer,N_prev_layer))
        #weight=tensor.new_full((N_current_layer, N_prev_layer), weight_val)
        weight=tensor
        weight=weight.to(device)
        return weight

def bias_initialiser(N_current_layer,device='cpu'):
        """
        Initializes the bias as a tensor.
        The value of the bias will be  b=0
        
        Args:
        N_current_layer:Number of elmemets in current Layer.

        Returns:
        bias: Tensor of filled with 0.
        """
        bias=torch.rand((N_current_layer,1))
        bias=bias.to(device)
        return bias



def batch_size_calc(inputs):
        """
        parm:Takes in the input tensor 

        returns:batch size(Integer)
        """
        inputs_size_arr=list(inputs.size())
        batch_size=inputs_size_arr[0]
        return batch_size
