"""
Utility functions for LatentNN.

This module provides:
- Data generation utilities for testing
- Metric calculation (attenuation factor λ)
- Device detection for GPU/CPU
"""

import numpy as np
import torch
from sklearn.linear_model import LinearRegression


def get_device(prefer_gpu=False):
    """
    Get the best available device.
    
    For small networks (hidden_size <= 64), CPU is often faster.
    Set prefer_gpu=True to force GPU usage for larger models.
    
    Parameters
    ----------
    prefer_gpu : bool, optional
        If True, prefer GPU when available. Default is False.
    
    Returns
    -------
    torch.device
        Available device (cuda, mps, or cpu)
    """
    if prefer_gpu and torch.cuda.is_available():
        return torch.device('cuda')
    elif prefer_gpu and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


def calc_lambda(y_true, y_pred):
    """
    Calculate the attenuation factor λ.
    
    λ is defined as the slope of the regression of y_pred against y_true.
    - λ = 1 means no attenuation (unbiased predictions)
    - λ < 1 means attenuation (predictions compressed toward mean)
    - λ > 1 would indicate overcorrection
    
    Parameters
    ----------
    y_true : np.ndarray
        True output values
    y_pred : np.ndarray
        Predicted output values
    
    Returns
    -------
    lambda_val : float
        Attenuation factor
    intercept : float
        Intercept of the regression
    
    Example
    -------
    >>> lam, intercept = calc_lambda(y_true, y_pred)
    >>> print(f"Attenuation factor: λ = {lam:.3f}")
    """
    lr = LinearRegression(fit_intercept=True)
    lr.fit(y_true.reshape(-1, 1), y_pred.reshape(-1, 1))
    return lr.coef_[0][0], lr.intercept_[0]


def generate_data(n_samples, snr_x, snr_y=10, x_range=(-5.0, 5.0), 
                  true_slope=2.0, seed=42):
    """
    Generate synthetic 1D regression data with measurement errors.
    
    The true relationship is y = true_slope * x.
    
    Parameters
    ----------
    n_samples : int
        Number of samples to generate
    snr_x : float
        Signal-to-noise ratio in x: SNR_x = σ_range / σ_x
        where σ_range is the standard deviation of x_true
    snr_y : float, optional
        Signal-to-noise ratio in y: SNR_y = σ_y_range / σ_y
        Default is 10.
    x_range : tuple, optional
        Range for uniform sampling of x_true. Default is (-5, 5).
    true_slope : float, optional
        True slope of the linear relationship. Default is 2.0.
    seed : int, optional
        Random seed for reproducibility. Default is 42.
    
    Returns
    -------
    dict
        Dictionary containing:
        - 'x_true': True x values
        - 'x_obs': Observed (noisy) x values
        - 'y_true': True y values
        - 'y_obs': Observed (noisy) y values
        - 'sigma_x': Standard deviation of x noise
        - 'sigma_y': Standard deviation of y noise
    
    Example
    -------
    >>> data = generate_data(1000, snr_x=10, snr_y=10)
    >>> print(f"σ_x = {data['sigma_x']:.3f}, σ_y = {data['sigma_y']:.3f}")
    """
    np.random.seed(seed)
    
    # Generate true values
    x_true = np.random.uniform(x_range[0], x_range[1], n_samples)
    y_true = true_slope * x_true
    
    # Compute noise levels from SNR
    sigma_x = np.std(x_true) / snr_x
    sigma_y = np.std(y_true) / snr_y
    
    # Add noise
    x_obs = x_true + np.random.normal(0, sigma_x, n_samples)
    y_obs = y_true + np.random.normal(0, sigma_y, n_samples)
    
    return {
        'x_true': x_true,
        'x_obs': x_obs,
        'y_true': y_true,
        'y_obs': y_obs,
        'sigma_x': sigma_x,
        'sigma_y': sigma_y
    }


def generate_correlated_data(n_samples, p, snr_x, snr_y=10, seed=42):
    """
    Generate synthetic multivariate data with correlated features.
    
    All p features are derived from a single latent variable z:
        X_true = z * a^T
    where a is a vector of scaling factors.
    
    The true output is y = X_true @ β where β = a.
    
    This setup creates strongly correlated features, mimicking
    astronomical data like stellar spectra where different pixels
    respond coherently to changes in physical parameters.
    
    Parameters
    ----------
    n_samples : int
        Number of samples to generate
    p : int
        Number of input features
    snr_x : float
        Signal-to-noise ratio in x (per feature)
    snr_y : float, optional
        Signal-to-noise ratio in y. Default is 10.
    seed : int, optional
        Random seed for reproducibility. Default is 42.
    
    Returns
    -------
    dict
        Dictionary containing:
        - 'X_true': True input values, shape (n_samples, p)
        - 'X_obs': Observed (noisy) input values, shape (n_samples, p)
        - 'y_true': True output values, shape (n_samples,)
        - 'y_obs': Observed (noisy) output values, shape (n_samples,)
        - 'sigma_x': Standard deviation of x noise
        - 'sigma_y': Standard deviation of y noise
        - 'beta_true': True coefficients
    
    Example
    -------
    >>> data = generate_correlated_data(1000, p=10, snr_x=5)
    >>> print(f"X shape: {data['X_obs'].shape}")
    X shape: (1000, 10)
    """
    np.random.seed(seed)
    
    # Latent variable
    z = np.random.uniform(-0.5, 0.5, n_samples)
    
    # Scaling factors and true coefficients
    a = np.linspace(0.2, 0.4, p)
    beta_true = a
    
    # Generate correlated features
    X_true = np.outer(z, a)
    
    # Compute noise level
    sigma_x = np.std(z) / snr_x
    X_obs = X_true + np.random.normal(0, sigma_x, (n_samples, p))
    
    # Output
    y_true = X_true @ beta_true
    sigma_y = np.std(y_true) / snr_y
    y_obs = y_true + np.random.normal(0, sigma_y, n_samples)
    
    return {
        'X_true': X_true,
        'X_obs': X_obs,
        'y_true': y_true,
        'y_obs': y_obs,
        'sigma_x': sigma_x,
        'sigma_y': sigma_y,
        'beta_true': beta_true
    }


def theoretical_attenuation(snr_x, p=1, a=None):
    """
    Compute the theoretical attenuation factor for linear regression.
    
    For 1D: λ = 1 / (1 + 1/SNR_x²)
    For correlated multivariate: λ = Σa_j² / (1/SNR_x² + Σa_j²)
    
    Parameters
    ----------
    snr_x : float or np.ndarray
        Signal-to-noise ratio in x
    p : int, optional
        Number of features. Default is 1.
    a : np.ndarray, optional
        Coefficient vector for multivariate case.
        If None, uses default a = linspace(0.2, 0.4, p).
    
    Returns
    -------
    float or np.ndarray
        Theoretical attenuation factor λ
    
    Example
    -------
    >>> # 1D case
    >>> lam = theoretical_attenuation(snr_x=10)  # ~0.99
    >>> lam = theoretical_attenuation(snr_x=1)   # 0.5
    >>> 
    >>> # Multivariate case with p=10
    >>> lam = theoretical_attenuation(snr_x=5, p=10)
    """
    if p == 1:
        return 1.0 / (1.0 + 1.0 / snr_x**2)
    else:
        if a is None:
            a = np.linspace(0.2, 0.4, p)
        sum_a2 = np.sum(a**2)
        kappa = 1.0 / snr_x**2
        return sum_a2 / (kappa + sum_a2)


def train_val_test_split(data, n_train=1000, n_val=200, n_test=200):
    """
    Split data dictionary into train/validation/test sets.
    
    IMPORTANT: Proper data splitting avoids data leakage!
    - Train: Used for training the model
    - Validation: Used for hyperparameter tuning (weight decay selection)
    - Test: Used ONLY for final evaluation
    
    Parameters
    ----------
    data : dict
        Data dictionary with arrays of length n_total >= n_train + n_val + n_test
    n_train : int, optional
        Number of training samples. Default is 1000.
    n_val : int, optional
        Number of validation samples. Default is 200.
    n_test : int, optional
        Number of test samples. Default is 200.
    
    Returns
    -------
    tuple
        (train_dict, val_dict, test_dict) each containing the split data
    """
    idx_val = n_train
    idx_test = n_train + n_val
    
    train = {}
    val = {}
    test = {}
    
    for key, value in data.items():
        if isinstance(value, np.ndarray) and len(value) >= idx_test + n_test:
            train[key] = value[:idx_val]
            val[key] = value[idx_val:idx_test]
            test[key] = value[idx_test:idx_test + n_test]
        else:
            # Scalar values (like sigma_x, sigma_y)
            train[key] = value
            val[key] = value
            test[key] = value
    
    return train, val, test

