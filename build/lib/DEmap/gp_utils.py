import torch
import gpytorch
import numpy as np
import os
from math import pi

def optimise_model (model, X_train, y_train_std, likelihood, mll):
    # Optimizer (Adam with all model parameters)
    """
    Optimizes a Gaussian Process model using Adam optimizer.

    Args:
        model (ExactGPModel): The Gaussian Process model to be optimized.
        X_train (torch.Tensor): The training inputs.
        y_train_std (torch.Tensor): The standardized training outputs.
        likelihood (MultivariateNormal): The likelihood of the training outputs.
        mll (gpytorch.mlls.ExactMarginalLogLikelihood): The marginal log likelihood of the training outputs.

    Returns:
        None
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

    # Train mode for model + likelihood
    model.train(); likelihood.train()
    prev_loss = None
    for i in range(250):
        optimizer.zero_grad()
        output = model(X_train)
        loss = -mll(output, y_train_std)
        loss_value = loss.item()
        loss.backward()
        optimizer.step()
        # Early stopping based on tolerance
        if prev_loss is not None and abs(prev_loss - loss_value) < 0.00001:
           #print(f"Converged at iteration {i} with change {abs(prev_loss - loss_value):.6e}")
           break


        prev_loss = loss_value

def converge_check(imse, patience=100, slope_tol=1e-3):
    """
    More robust convergence check based on IMSE stabilization.

    Parameters:
        imse (list or np.ndarray): Sequence of IMSE values.
        patience (int): Number of recent iterations to consider.
        slope_tol (float): Maximum allowed average slope magnitude of recent IMSEs.

    Returns:
        bool: True if converged, False otherwise.
    """
    if len(imse) < patience:
        return False

    recent = np.array(imse[-patience:])



    # Compute average slope (simple linear regression)
    x = np.arange(len(recent))
    coeffs = np.polyfit(x, recent, 1)
    slope = coeffs[0]

    # Converged if slope near zero
    if abs(slope) < slope_tol:
        return True
    return False


def query_model(model, probe_pts, y_std, y_mean,likelihood=None):

    """
    Queries a Gaussian Process model at a set of probe points.

    Args:
        model (ExactGPModel): The Gaussian Process model to be queried.
        probe_pts (torch.Tensor): The points at which to query the model.
        y_std (float): The standard deviation of the training outputs.
        y_mean (float): The mean of the training outputs.
        likelihood (MultivariateNormal): The likelihood of the training outputs.

    Returns:
        mean_pred (torch.Tensor): The predicted mean values at the probe points.
        var_pred (torch.Tensor): The predicted variance values at the probe points.
    """
    if likelihood is not None:
        likelihood.eval()
    model.eval()
    with torch.no_grad():
        if likelihood is not None:
            test_noise = torch.full((probe_pts.shape[0],), 0.5 / (y_std**2), dtype=torch.double)
            preds = likelihood(model(probe_pts), noise=test_noise)
        else:
            preds = model(probe_pts)
        mean_pred = preds.mean * y_std + y_mean     # de-standardize back to eV
        var_pred = preds.variance * (y_std**2)

    return mean_pred, var_pred


def save_checkpoint(X_train, y_train, noise_train, it, imse, stage="iter"):
    """
    Saves a checkpoint of the current model state.

    Args:
        X_train (torch.Tensor): The training inputs.
        y_train (torch.Tensor): The training outputs.
        noise_train (torch.Tensor): The training noise.
        it (int): The current iteration number.
        imse (float): The current IMSE value.
        stage (str): The current stage of the active learning loop.

    Returns:
        None
    """
    torch.save({
        'X_train': X_train,
        'y_train': y_train,
        'noise_train': noise_train,
        'iteration': it,
        'stage': stage,
        'imse': imse
    }, 'checkpoint.pt')

def load_checkpoint():
    """
    Loads a checkpoint of the current model state.

    Returns:
        dict: A dictionary containing the state of the model if a checkpoint exists, None otherwise.
    """
    if os.path.exists('checkpoint.pt'):
        return torch.load('checkpoint.pt', weights_only=False)
    return None

def estimate_imse(model, likelihood, probe_pts, y_std, theta_range, phi_range):
    """
    Estimates the integrated mean squared error (IMSE) of the model on a given set of probe points.

    Parameters:
        model (ExactGPModel): The Gaussian Process model to estimate the IMSE for.
        likelihood (MultivariateNormal): The likelihood of the training outputs.
        probe_pts (torch.Tensor): The probe points to estimate the IMSE at.
        y_std (torch.Tensor): The standard deviation of the training outputs.
        theta_range (tuple[float, float], optional): The range of theta values to estimate the IMSE for. Defaults to None.
        phi_range (tuple[float, float], optional): The range of phi values to estimate the IMSE for. Defaults to None.

    Returns:
        float: The estimated IMSE value.
    """
    model.eval(); likelihood.eval()
    if theta_range is not None:
        theta_min, theta_max = theta_range
    else:
        theta_min, theta_max = 0, np.pi
    if phi_range is not None:
        phi_min, phi_max = phi_range
    else:
        phi_min, phi_max = 0, 2 * np.pi
    # Fraction of total surface area of sphere (to scale IMSE expression)
    area_fraction = (phi_max - phi_min) * (np.cos(theta_min) - np.cos(theta_max)) / (4 * np.pi)

    # Don't compute gradients for resources, use fater vairance prediction at minimal accuracy loss
    with torch.no_grad(), gpytorch.settings.cholesky_jitter(1e-8):
        # Extract model predictions at probe points
        preds = model(probe_pts)
        # Preds is GPyTorch MultivariateNormal object with .mean and .variance
        # Multiply by train_y_std^2 to get variance in original units
        var = torch.clamp(preds.variance, min=0.0) * (y_std**2)
    # Integrating over sphere requires 4 pi multiplier so variance spread over the whole sphere gives higher value.
    return area_fraction * ((4*pi )/ probe_pts.shape[0]) * var.sum().item() # Sum of vairance at all points, multiplied by weight factor (float)