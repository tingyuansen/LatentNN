# LatentNN

**Correcting Attenuation Bias in Neural Networks for Astronomical Applications**

## The Problem: Why Astronomical ML is Different

Modern astronomy increasingly relies on neural networks for inference—deriving stellar parameters from spectra, photometric redshifts from imaging, stellar ages from light curves. These applications share a common feature that sets them apart from typical machine learning domains: **significant measurement uncertainties in the input data**.

In real life applications, inputs (e.g., images) have negligible noises. But in astronomy, we routinely work with spectra where measurement errors are 1-10% of the signal. When we train neural networks treating these noisy observations as exact, something systematic goes wrong.

### Attenuation Bias

When inputs have measurement errors, regression models systematically **underestimate the dynamic range** of predictions:
- High values are predicted too low
- Low values are predicted too high  
- The predicted range is compressed toward the mean

This is **attenuation bias** (also known as regression dilution), a phenomenon well-understood in linear regression but largely ignored in neural network applications. The bias:
- Cannot be fixed by collecting more training data
- Cannot be fixed by using more complex models
- Depends only on the ratio of measurement error to signal range

For a linear relationship, the attenuation factor is:

$$\lambda = \frac{1}{1 + (\sigma_x / \sigma_{\mathrm{range}})^2}$$

At SNR = 10, this gives λ ≈ 0.99 (1% bias). At SNR = 3, λ ≈ 0.90 (10% bias). At SNR = 1, λ = 0.50 (50% bias). Neural networks follow this same curve—model complexity provides no protection.

### Why This Matters for Astronomy

Consider stellar spectroscopy. When transferring labels from high-resolution surveys (APOGEE) to low-resolution surveys (LAMOST, DESI, Gaia XP), we operate exactly in the regime where attenuation bias is severe. A metal-poor star at [M/H] = -2 might be predicted as [M/H] = -1.5, potentially misclassifying a halo star as thick-disk. The bias directly affects:
- Metallicity distribution functions and their metal-poor tails
- Age estimates at both young and old extremes
- Distance measurements and Galactic structure
- Any quantity where the extremes are scientifically important

Elements with few spectral features (K, V) suffer more than Fe with its many lines, because the effective dimensionality is lower and attenuation is stronger.

## The Solution: Latent Variable Approach

The classical solution for linear regression is **Deming regression**: treat the true input values as latent variables to be estimated alongside model parameters. LatentNN extends this idea to neural networks.

Instead of minimizing the standard loss, LatentNN minimizes:

$$\mathcal{L} = \frac{1}{\sigma_y^2} \sum_i \left( y_{\mathrm{obs},i} - f_\theta(x_{\mathrm{latent},i}) \right)^2 + \frac{1}{\sigma_x^2} \sum_i \left( x_{\mathrm{obs},i} - x_{\mathrm{latent},i} \right)^2$$

where $x_{\mathrm{latent}}$ are learnable parameters initialized at the observed values. The first term fits the model; the second term keeps the latent values close to observations. The balance is controlled by the known uncertainties $\sigma_x$ and $\sigma_y$.

For theoretical details, derivations, and extensive validation, see:

> Ting, Y.-S. (2025). "Why Machine Learning Models Systematically Underestimate Extreme Values II: How to Fix It with LatentNN". arXiv:XXXX.XXXXX

## Installation

```bash
git clone https://github.com/tingyuansen/LatentNN.git
cd LatentNN
pip install -e .
```

## Usage

### Training a Standard MLP (Baseline)

```python
from latentnn import train_mlp, predict_mlp

# Train standard MLP - will exhibit attenuation bias
model = train_mlp(
    x_train, y_train,
    hidden_size=64,           # Hidden units per layer
    num_layers=2,             # Number of hidden layers
    epochs=10000,
    lr=0.03,                  # Learning rate
    weight_decay=0.01
)

# Predict
y_pred = predict_mlp(model, x_test)
```

The standard MLP treats observed inputs as exact, leading to attenuation bias when inputs are noisy.

### Training LatentNN (Bias Correction)

```python
from latentnn import train_latent_mlp, predict_latent

# Train LatentNN with known measurement uncertainties
model = train_latent_mlp(
    x_train, y_train,
    sigma_x=0.5,              # Input measurement uncertainty
    sigma_y=0.1,              # Output measurement uncertainty
    hidden_size=64,           # Hidden units per layer
    num_layers=2,             # Number of hidden layers
    epochs=20000,
    lr=0.03,                  # Learning rate
    weight_decay=0.0001,
    x_val=x_val,              # Validation data for model selection
    y_true_val=y_true_val
)

# Predict - uses learned network, not latent values
y_pred = predict_latent(model, x_test)
```

LatentNN jointly optimizes network parameters and latent input values, correcting attenuation bias.

### Hyperparameter Tuning: weight_decay

The `weight_decay` parameter is critical for LatentNN. It regularizes the network parameters (but not the latent values, which are regularized by the likelihood term). The optimal value depends on the SNR regime:

```python
from latentnn import train_latent_mlp, predict_latent, calc_lambda

# Search over weight_decay values
weight_decays = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]
best_wd, best_lam = None, 0

for wd in weight_decays:
    model = train_latent_mlp(
        x_train, y_train, sigma_x, sigma_y,
        epochs=20000, weight_decay=wd,
        x_val=x_val, y_true_val=y_true_val
    )
    lam, _ = calc_lambda(y_true_val, predict_latent(model, x_val))
    
    # Select weight_decay giving λ closest to 1
    if best_wd is None or abs(lam - 1.0) < abs(best_lam - 1.0):
        best_wd, best_lam = wd, lam

# Train final model with best weight_decay
final_model = train_latent_mlp(
    x_train, y_train, sigma_x, sigma_y,
    epochs=20000, weight_decay=best_wd,
    x_val=x_val, y_true_val=y_true_val
)
```

**Important:** Use a three-way split (train/validation/test). The validation set is used for hyperparameter selection; the test set is held out for final evaluation.

### Measuring Attenuation

```python
from latentnn import calc_lambda

# λ = 1 means no attenuation (unbiased)
# λ < 1 means predictions are compressed toward the mean
lam, intercept = calc_lambda(y_true_test, y_pred)
print(f"Attenuation factor: λ = {lam:.3f}")
```

## API Reference

### Training Functions

| Function | Description |
|----------|-------------|
| `train_mlp(x_obs, y_obs, ...)` | Train standard MLP (baseline with attenuation bias) |
| `train_latent_mlp(x_obs, y_obs, sigma_x, sigma_y, ...)` | Train LatentNN (corrects attenuation bias) |

**Network architecture:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `hidden_size` | 64 | Number of hidden units per layer |
| `num_layers` | 2 | Number of hidden layers |

**Training parameters:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `epochs` | 20000 | Number of training iterations |
| `lr` | 0.03 | Learning rate for Adam optimizer |
| `weight_decay` | 0.0 | L2 regularization on network weights |

**LatentNN-specific:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `sigma_x` | (required) | Input measurement uncertainty |
| `sigma_y` | (required) | Output measurement uncertainty |
| `x_val`, `y_true_val` | None | Validation data for model selection (best λ) |
| `return_losses` | False | Return loss history for diagnostics |

### Prediction Functions

| Function | Description |
|----------|-------------|
| `predict_mlp(model, x)` | Predict with standard MLP |
| `predict_latent(model, x)` | Predict with LatentNN (uses learned network) |

### Utilities

| Function | Description |
|----------|-------------|
| `calc_lambda(y_true, y_pred)` | Measure attenuation factor λ (1 = unbiased, <1 = compressed) |

## Tutorials

Pedagogical notebooks demonstrating the method:
- `tutorial_single_variate.ipynb` — Core concepts with 1D regression
- `tutorial_multivariate.ipynb` — Correlated features (e.g., spectral pixels)

## Citation

```bibtex
@article{Ting2025_LatentNN,
    author = {Ting, Yuan-Sen},
    title = "{Why Machine Learning Models Systematically Underestimate Extreme Values II: How to Fix It with LatentNN}",
    journal = {arXiv e-prints},
    year = 2025,
    eprint = {XXXX.XXXXX},
    archivePrefix = {arXiv},
    primaryClass = {astro-ph.IM}
}
```

## License

This code is freely available for academic use. If you use it, please cite the paper above.
