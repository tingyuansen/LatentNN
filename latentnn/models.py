"""
Neural network model definitions for LatentNN.

This module provides:
- LatentMLP: Neural network with learnable latent input values
- build_mlp: Standard MLP for baseline comparison
"""

import torch
import torch.nn as nn


class LatentMLP(nn.Module):
    """
    Multi-layer perceptron with learnable latent input values.
    
    This is the core of the LatentNN approach. Instead of treating the observed
    inputs x_obs as exact, we treat the true values x_true as latent variables
    (x_latent) to be estimated alongside the network parameters.
    
    The joint optimization minimizes:
        L = (1/σ_y²) Σ(y - f(x_latent))² + (1/σ_x²) Σ(x_obs - x_latent)²
    
    Parameters
    ----------
    n_samples : int
        Number of training samples. One latent value per sample.
    input_dim : int, optional
        Dimension of input features. Default is 1.
    hidden_size : int, optional
        Number of units per hidden layer. Default is 64.
    num_layers : int, optional
        Number of hidden layers. Default is 2.
    
    Attributes
    ----------
    network : nn.Sequential
        The neural network mapping x_latent → y_pred
    x_latent : nn.Parameter
        Learnable latent input values, shape (n_samples, input_dim)
    
    Example
    -------
    >>> model = LatentMLP(n_samples=1000, input_dim=1)
    >>> # Initialize latent values at observed values
    >>> model.x_latent.data = torch.FloatTensor(x_obs.reshape(-1, 1))
    >>> # Forward pass returns predictions from latent values
    >>> y_pred = model()
    """
    
    def __init__(self, n_samples, input_dim=1, hidden_size=64, num_layers=2):
        super().__init__()
        
        # Build network layers
        layers = []
        in_features = input_dim
        for _ in range(num_layers):
            layers.extend([
                nn.Linear(in_features, hidden_size),
                nn.ReLU()
            ])
            in_features = hidden_size
        layers.append(nn.Linear(hidden_size, 1))
        
        self.network = nn.Sequential(*layers)
        
        # Learnable latent values - one per sample
        # Initialized to zeros, should be set to x_obs before training
        self.x_latent = nn.Parameter(torch.zeros(n_samples, input_dim))
    
    def forward(self):
        """
        Forward pass through the network using latent values.
        
        Returns
        -------
        torch.Tensor
            Predictions, shape (n_samples, 1)
        """
        return self.network(self.x_latent)
    
    def predict(self, x):
        """
        Make predictions for new input values.
        
        Parameters
        ----------
        x : torch.Tensor
            Input values, shape (n, input_dim)
        
        Returns
        -------
        torch.Tensor
            Predictions, shape (n, 1)
        """
        return self.network(x)


def build_mlp(input_dim=1, hidden_size=64, num_layers=2):
    """
    Build a standard multi-layer perceptron.
    
    This serves as the baseline model that exhibits attenuation bias
    when trained on noisy inputs.
    
    Parameters
    ----------
    input_dim : int, optional
        Dimension of input features. Default is 1.
    hidden_size : int, optional
        Number of units per hidden layer. Default is 64.
    num_layers : int, optional
        Number of hidden layers. Default is 2.
    
    Returns
    -------
    nn.Sequential
        The MLP model
    
    Example
    -------
    >>> model = build_mlp(input_dim=10, hidden_size=64, num_layers=2)
    >>> x = torch.randn(100, 10)
    >>> y_pred = model(x)
    """
    layers = []
    in_features = input_dim
    for _ in range(num_layers):
        layers.extend([
            nn.Linear(in_features, hidden_size),
            nn.ReLU()
        ])
        in_features = hidden_size
    layers.append(nn.Linear(hidden_size, 1))
    
    return nn.Sequential(*layers)

