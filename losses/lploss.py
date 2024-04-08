import torch

class LpLoss(object):
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()
        
        # Check if the dimensionality is valid
        assert d > 0, "The dimensionality must be greater than 0"
        # Check if the p-value is valid
        assert p > 0, "The p-value must be greater than 0"
        
        # Set the dimensionality
        self.d = d
        # Set the p-value
        self.p = p
        
        # Set the size average
        self.size_average = size_average
        
        # Set the reduction
        self.reduction = reduction
        
    def abs(self, x, y):
        # Get the batch size
        num_examples = x.shape[0]
        
        # Assume uniform mesh
        h = 1.0 / (x.shape[1] - 1.0)
        
        # Calculate the Lp norm
        all_norms = (h**(self.d/self.p))*torch.norm(x.view(num_examples,-1) - y.view(num_examples,-1), self.p, 1)
        
        # Check if we need to reduce the loss
        if self.reduction:
            # Check if we need to average the loss
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)
            
        return all_norms
    
    def relative(self, x, y):
        # Get the batch size
        num_examples = x.shape[0]
        
        # Calculate the difference in norms
        diff_norms = torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), self.p, 1)
        # Calculate the norm of y
        y_norms = torch.norm(y.reshape(num_examples,-1), self.p, 1)
        
        # Check if we need to reduce the loss
        if self.reduction:
            # Check if we need to average the loss
            if self.size_average:
                return torch.mean(diff_norms/y_norms)
            else:
                return torch.sum(diff_norms/y_norms)
            
        return diff_norms/y_norms
    
    def __call__(self, x, y):
        return self.relative(x, y)
        
        
        