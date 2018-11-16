import torch

def weight_initialiser(N_prev_layer,N_current_layer):
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
        weight_val = 1/(N_prev_layer**0.5)
        weight=torch.empty(N_current_layer,1)
        weight._fill(weight_val)
        return weight

def bias_initialiser(N_current_layer):
        """
        Initializes the bias as a tensor.
        The value of the bias will be  b=0
        
        Args:
        N_current_layer:Number of elmemets in current Layer.

        Returns:
        bias: Tensor of filled with 0.
        """
        bias=torch.zeros(N_current_layer,1)
        
        return bias