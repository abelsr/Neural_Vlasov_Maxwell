import torch
import torch.nn as nn

class MLP(nn.Module):
    """
    Multi-Layer Perceptron (MLP) neural network model.

    Args:
        in_features (int): Number of input features.
        hidden_size (int): Number of units in the hidden layer.
        out_features (int): Number of output features. If not provided, defaults to `in_features`.
        activation (torch.nn.Module, optional): Activation function to use. Defaults to `torch.nn.ReLU`.
        dropout (float, optional): Dropout probability. Defaults to 0.0.

    Attributes:
        out_features (int): Number of output features.
        hidden_size (int): Number of units in the hidden layer.
        activation (torch.nn.Module): Activation function.
        fc1 (torch.nn.Linear): First fully connected layer.
        fc2 (torch.nn.AdaptiveAvgPool1d): Second fully connected layer.
        dropout (torch.nn.Dropout): Dropout layer.

    """
    def __init__(self, in_features: int, hidden_size: int, out_features: int = None, activation: nn.Module = nn.ReLU, dropout: float = 0.0):
        super().__init__()
        self.out_features = out_features or in_features
        self.hidden_size = hidden_size or in_features
        self.activation = activation()
        
        # Layers
        self.fc1 = nn.Linear(in_features, hidden_size)
        self.fc2 = nn.AdaptiveAvgPool1d(self.out_features)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the MLP module.
        
        Args:
            x (torch.Tensor): Input tensor.
            
        Returns:
            torch.Tensor: Output tensor.
        """
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x