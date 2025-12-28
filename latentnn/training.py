"""
Training functions for LatentNN and standard MLP.

This module provides training routines that implement:
- Standard MLP training (exhibits attenuation bias)
- LatentNN training (corrects attenuation bias)
"""

import torch
import torch.nn as nn
import numpy as np
from .models import LatentMLP, build_mlp
from .utils import calc_lambda, get_device


def train_mlp(x_obs, y_obs, input_dim=None, hidden_size=64, num_layers=2,
              epochs=20000, lr=0.03, weight_decay=0.0, device=None):
    """
    Train a standard MLP on noisy observations.
    
    This baseline approach treats x_obs as exact, which leads to 
    attenuation bias when inputs have measurement errors.
    
    Parameters
    ----------
    x_obs : np.ndarray
        Observed input values, shape (n_samples,) or (n_samples, input_dim)
    y_obs : np.ndarray
        Observed output values, shape (n_samples,)
    input_dim : int, optional
        Input dimension. Inferred from x_obs if None.
    hidden_size : int, optional
        Hidden layer size. Default is 64.
    num_layers : int, optional
        Number of hidden layers. Default is 2.
    epochs : int, optional
        Number of training epochs. Default is 20000.
    lr : float, optional
        Learning rate. Default is 0.03.
    weight_decay : float, optional
        L2 regularization strength. Default is 0.
    device : torch.device, optional
        Device to train on. Auto-detected if None.
    
    Returns
    -------
    nn.Sequential
        Trained MLP model
    
    Example
    -------
    >>> model = train_mlp(x_obs, y_obs, epochs=10000)
    >>> y_pred = predict_mlp(model, x_test)
    """
    if device is None:
        device = get_device()
    
    # Reshape inputs if needed and infer input_dim
    if x_obs.ndim == 1:
        x_obs = x_obs.reshape(-1, 1)
        input_dim = 1
    else:
        input_dim = x_obs.shape[1] if input_dim is None else input_dim
    
    model = build_mlp(input_dim, hidden_size, num_layers).to(device)
    
    x_t = torch.FloatTensor(x_obs).to(device)
    y_t = torch.FloatTensor(y_obs.reshape(-1, 1)).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = torch.mean((model(x_t) - y_t)**2)
        loss.backward()
        optimizer.step()
    
    return model


def train_latent_mlp(x_obs, y_obs, sigma_x, sigma_y, 
                     input_dim=None, hidden_size=64, num_layers=2,
                     epochs=20000, lr=0.03, weight_decay=0.0,
                     x_val=None, y_true_val=None,
                     return_losses=False, eval_interval=100, device=None):
    """
    Train LatentNN with joint optimization of network and latent values.
    
    This method corrects attenuation bias by treating the true input values
    as latent variables to be estimated alongside the network parameters.
    
    The loss function is:
        L = (1/σ_y²) Σ(y - f(x_latent))² + (1/σ_x²) Σ(x_obs - x_latent)²
    
    For multivariate inputs (p > 1), the regularization term is:
        (1/σ_x²) Σ_i Σ_j (x_obs_ij - x_latent_ij)²
    
    Parameters
    ----------
    x_obs : np.ndarray
        Observed input values, shape (n_samples,) or (n_samples, input_dim)
    y_obs : np.ndarray
        Observed output values, shape (n_samples,)
    sigma_x : float
        Measurement uncertainty in x (standard deviation)
    sigma_y : float
        Measurement uncertainty in y (standard deviation)
    input_dim : int, optional
        Input dimension. Inferred from x_obs if None.
    hidden_size : int, optional
        Hidden layer size. Default is 64.
    num_layers : int, optional
        Number of hidden layers. Default is 2.
    epochs : int, optional
        Number of training epochs. Default is 20000.
    lr : float, optional
        Learning rate. Default is 0.03.
    weight_decay : float, optional
        L2 regularization on network parameters only. Default is 0.0.
        Note: NOT applied to latent values (they are regularized by the
        likelihood term in the loss).
    x_val : np.ndarray, optional
        Validation inputs for model selection (best λ closest to 1).
    y_true_val : np.ndarray, optional
        True validation outputs for computing validation λ.
    return_losses : bool, optional
        If True, return training loss history. Default is False.
    eval_interval : int, optional
        Interval for evaluating validation metrics. Default is 100.
    device : torch.device, optional
        Device to train on. Auto-detected if None.
    
    Returns
    -------
    model : LatentMLP
        Trained LatentNN model
    losses : dict (only if return_losses=True)
        Dictionary containing loss history:
        - 'total': Total loss
        - 'pred': Prediction loss term
        - 'reg': Latent regularization term (x_latent likelihood)
        - 'lambda_train': Training λ over epochs
        - 'lambda_val': Validation λ over epochs (if validation data provided)
    
    Example
    -------
    >>> model, losses = train_latent_mlp(
    ...     x_obs, y_obs, sigma_x=0.5, sigma_y=0.1,
    ...     return_losses=True
    ... )
    >>> y_pred = predict_latent(model, x_test)
    """
    if device is None:
        device = get_device()
    
    # Reshape inputs if needed and infer input_dim
    if x_obs.ndim == 1:
        x_obs = x_obs.reshape(-1, 1)
        input_dim = 1
    else:
        input_dim = x_obs.shape[1] if input_dim is None else input_dim
    
    n_samples = len(x_obs)
    model = LatentMLP(n_samples, input_dim, hidden_size, num_layers).to(device)
    
    x_t = torch.FloatTensor(x_obs).to(device)
    y_t = torch.FloatTensor(y_obs.reshape(-1, 1)).to(device)
    
    # Initialize latent values at observed values
    model.x_latent.data = x_t.clone()
    
    # Separate optimizers: weight decay only on network parameters
    optimizer = torch.optim.Adam([
        {'params': model.network.parameters(), 'weight_decay': weight_decay},
        {'params': [model.x_latent], 'weight_decay': 0.0}  # No weight decay on latent
    ], lr=lr)
    
    # Prepare validation if provided
    track_best = x_val is not None and y_true_val is not None
    if track_best:
        if x_val.ndim == 1:
            x_val = x_val.reshape(-1, 1)
        x_val_t = torch.FloatTensor(x_val).to(device)
        best_state = None
        best_dist = float('inf')
    
    # Loss tracking
    losses = {
        'total': [], 'pred': [], 'reg': [],
        'lambda_train': [], 'lambda_val': []
    } if return_losses or track_best else None
    
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        y_pred = model()
        
        # Prediction loss (normalized by σ_y²)
        pred_loss = torch.mean((y_pred - y_t)**2) / (sigma_y**2)
        
        # Latent regularization (normalized by σ_x²)
        # For multivariate: sum over dimensions, mean over samples
        if input_dim > 1:
            reg_loss = torch.mean(torch.sum((model.x_latent - x_t)**2, dim=1)) / (sigma_x**2)
        else:
            reg_loss = torch.mean((model.x_latent - x_t)**2) / (sigma_x**2)
        
        loss = pred_loss + reg_loss
        loss.backward()
        optimizer.step()
        
        # Tracking
        if (return_losses or track_best) and epoch % eval_interval == 0:
            with torch.no_grad():
                if losses is not None:
                    losses['total'].append(loss.item())
                    losses['pred'].append(pred_loss.item())
                    losses['reg'].append(reg_loss.item())
                    
                    lam_train, _ = calc_lambda(y_obs, model().cpu().numpy().flatten())
                    losses['lambda_train'].append(lam_train)
                
                if track_best:
                    y_val_pred = model.network(x_val_t).cpu().numpy().flatten()
                    lam_val, _ = calc_lambda(y_true_val, y_val_pred)
                    
                    if losses is not None:
                        losses['lambda_val'].append(lam_val)
                    
                    # Save best model (closest λ to 1)
                    if abs(lam_val - 1.0) < best_dist:
                        best_dist = abs(lam_val - 1.0)
                        best_state = {
                            'net': {k: v.clone() for k, v in model.network.state_dict().items()},
                            'x_latent': model.x_latent.data.clone()
                        }
    
    # Restore best model if validation was used
    if track_best and best_state is not None:
        model.network.load_state_dict(best_state['net'])
        model.x_latent.data = best_state['x_latent']
    
    if return_losses:
        return model, losses
    return model


def predict_mlp(model, x, device=None):
    """
    Make predictions with a standard MLP.
    
    Parameters
    ----------
    model : nn.Sequential
        Trained MLP model
    x : np.ndarray
        Input values
    device : torch.device, optional
        Device to use. Auto-detected if None.
    
    Returns
    -------
    np.ndarray
        Predictions
    """
    if device is None:
        device = get_device()
    
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    
    model.eval()
    with torch.no_grad():
        return model(torch.FloatTensor(x).to(device)).cpu().numpy().flatten()


def predict_latent(model, x, device=None):
    """
    Make predictions with a LatentNN model.
    
    Uses the learned network to predict from new inputs (not the training
    latent values, but the network applied to new x values).
    
    Parameters
    ----------
    model : LatentMLP
        Trained LatentNN model
    x : np.ndarray
        Input values
    device : torch.device, optional
        Device to use. Auto-detected if None.
    
    Returns
    -------
    np.ndarray
        Predictions
    """
    if device is None:
        device = get_device()
    
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    
    model.eval()
    with torch.no_grad():
        return model.network(torch.FloatTensor(x).to(device)).cpu().numpy().flatten()

