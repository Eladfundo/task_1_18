import torch
print("Import Done")

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
        print(weight_val)
        tensor = torch.ones((N_current_layer,1), dtype=torch.float64,requires_grad=True)
        weight=tensor.new_full((N_current_layer, 1), 3.141592)
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
        bias=torch.zeros((N_current_layer,1),requires_grad=True)
        bias=bias.to(device)
        return bias



