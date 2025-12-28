"""
LatentNN: Correcting Attenuation Bias in Neural Networks

A method for correcting attenuation bias in neural network regression by treating
true input values as latent variables to be estimated alongside model parameters.

Reference:
    Ting, Y.-S. (2025). "Why Machine Learning Models Systematically Underestimate 
    Extreme Values II: How to Fix It with LatentNN". arXiv:XXXX.XXXXX

Example:
    >>> from latentnn import LatentMLP, train_latent_mlp, train_mlp
    >>> # Train LatentNN
    >>> model, losses = train_latent_mlp(X_obs, y_obs, sigma_x, sigma_y)
    >>> # Make predictions
    >>> y_pred = predict_latent(model, X_new)
"""

from .models import LatentMLP, build_mlp
from .training import train_mlp, train_latent_mlp, predict_mlp, predict_latent
from .utils import (
    generate_data, generate_correlated_data, 
    calc_lambda, get_device, theoretical_attenuation,
    train_val_test_split
)

__version__ = "1.0.0"
__author__ = "Yuan-Sen Ting"
__email__ = "ting.74@osu.edu"

__all__ = [
    # Models
    "LatentMLP",
    "build_mlp",
    # Training
    "train_mlp",
    "train_latent_mlp",
    "predict_mlp",
    "predict_latent",
    # Utilities
    "generate_data",
    "generate_correlated_data",
    "calc_lambda",
    "get_device",
    "theoretical_attenuation",
    "train_val_test_split",
]

