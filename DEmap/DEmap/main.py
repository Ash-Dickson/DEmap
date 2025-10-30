"""
TDE Surface Active Learning with GPyTorch on the Sphere
-----------------------------------------------------

- Models TDE as a function on the unit sphere using a Gaussian Process.
- Uses GPyTorch for scalable GP inference.
- Space-filling initialization with Fibonacci lattice.
- Active learning loop that reduces global surface uncertainty via IMSE.
"""


import numpy as np
from DEmap import __version__
from typing import Tuple, Optional
import torch
import gpytorch
from plot_tools import *
from TDE import evaluate_tde
from sphere_utils import fibonacci_sphere, random_sphere_points
from config import Config
from kernels import ExactGPModel
from gp_utils import save_checkpoint, load_checkpoint, query_model, estimate_imse, optimise_model, converge_check
from lmp_input_generator import generate_lammps_input




def demap(cfg, theta_range: Optional[Tuple[float, float]] = None,
    phi_range: Optional[Tuple[float, float]] = None, restart: bool = False,
    plot: bool=False):
    """
    Active learning of the TDE surface using Gaussian Processes.

    Parameters:
    cfg (Config): Config object containing hyperparameters for the run
    theta_range (Optional[Tuple[float, float]]): Range of theta values for the sphere
    phi_range (Optional[Tuple[float, float]]): Range of phi values for the sphere
    restart (bool): If True, load the latest checkpoint and continue from there
    plot (bool): If True, plot the mean and variance of the TDE on all probe points

    Returns:
    None
    """


    startup_text = f'''
---------------------------------------------
  ___  ___                
 |   \| __|_ __  __ _ _ __ 
 | |) | _|| '  \/ _` | '_ \ 
 |___/|___|_|_|_\__,_| .__/ 
                     |_|

VERSION: {__version__} 
AUTHOR: Ashley Dickson
CORRESPONDANCE: a.dickson2@lancaster.ac.uk
---------------------------------------------
    ''' 
    print(startup_text)
    # Validate theta_range
    if theta_range is not None:
        if not (0 <= theta_range[0] < theta_range[1] <= np.pi):
            raise ValueError(
                f"Invalid theta_range {theta_range}. "
                f"It must satisfy 0 <= theta_min < theta_max <= pi."
            )

    # Validate phi_range
    if phi_range is not None:
        if not (0 <= phi_range[0] < phi_range[1] <= 2 * np.pi):
            raise ValueError(
                f"Invalid phi_range {phi_range}. "
                f"It must satisfy 0 <= phi_min < phi_max <= 2pi."
            )
    # Initialise seeds to maintain stability on restart   
    np.random.seed(cfg.random_seed)
    torch.manual_seed(cfg.random_seed)
    # Load checkpoint file on restart 
    checkpoint = load_checkpoint() if restart else None
    # Initialise list of IMSE for each iteration
    imse_list = []
    if checkpoint and checkpoint['stage'] == "init":
        # Logic for restarting on initiasl point sample
        X_train = checkpoint['X_train']
        y_train = checkpoint['y_train']
        noise_train = checkpoint['noise_train']
        start_idx = X_train.shape[0]
        init_pts = fibonacci_sphere(n=cfg.init_n, theta_range=theta_range, phi_range=phi_range)
        print(f"Resuming initial sampling at point {start_idx}/{len(init_pts)}")
    elif not checkpoint:
        # Run from scratch w/o restart 
        print('Starting initial point sampling...')
        init_pts = torch.tensor(fibonacci_sphere(n=cfg.init_n, theta_range=theta_range, phi_range=phi_range), dtype=torch.float32)
        print('Total number of initial points:', len(init_pts))
        X_train = torch.empty((0, len(init_pts[0])), dtype=torch.float32)
        y_train = torch.empty((0,), dtype=torch.float32)
        noise_train = torch.empty((0,), dtype=torch.float32)
        start_idx = 0
    else:
        # Loaded full checkpoint (past init)
        print('Resuming from checkpoint.pt...')
        X_train = checkpoint['X_train']
        y_train = checkpoint['y_train']
        noise_train = checkpoint['noise_train']
        start_idx = None

    # Initial sampling 
    if start_idx is not None:
        for i in range(start_idx, len(init_pts)):
            u = init_pts[i]
            val, noise = evaluate_tde(cfg=cfg, u=u)
            print(f"TDE of initial point {i+1}/{len(init_pts)}: {val:.2f} eV")

            X_train = torch.cat([X_train, u.unsqueeze(0).clone().detach()])
            y_train = torch.cat([y_train, torch.tensor([val])])
            noise_train = torch.cat([noise_train, torch.tensor([noise])])

            save_checkpoint(X_train, y_train, noise_train, i, imse_list, stage="init")

        print('Initial points sampled...')

    # Build GP model 
    y_mean = y_train.mean()
    y_std = y_train.std()
    # Standardise to unit variance
    y_train_std = (y_train - y_mean) / y_std

    # Likelihood (noise) model
    likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(
        noise=noise_train, learn_additional_noise=False
    )
    model = ExactGPModel(X_train, y_train_std, likelihood)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    # Optimise model hyperparams given initial data 
    optimise_model(model, X_train, y_train_std, likelihood, mll)

    # Generate probe points on sphere (randomly distributed)
    probe = torch.tensor(random_sphere_points(n=cfg.probe_n, theta_range=theta_range, phi_range=phi_range), dtype=torch.float32)
    
    # Iterative loop 
    start_it = checkpoint['iteration'] + 1 if checkpoint and checkpoint['stage'] == "iter" else 1
    
    for it in range(start_it, cfg.max_iters + 1):
        
        imse = estimate_imse(model, likelihood, probe, y_std, theta_range, phi_range)
        imse_list.append(imse)
        if converge_check(imse_list):
            print('IMSE converged, exiting...')
            break

        save_checkpoint(X_train, y_train, noise_train, it, imse, stage="iter")
        
        # Grab predicted variance at all probe points
        _, var_pred = query_model(model=model, likelihood=None, probe_pts=probe, y_std=y_std, y_mean=y_mean)
        idx = torch.argmax(var_pred).item()
        # Find point with maximum variance
        new_pts = probe[idx].unsqueeze(0)

        pred_mean, pred_var = query_model(model=model, likelihood=None, probe_pts=new_pts, y_std=y_std, y_mean=y_mean)
     
        
        # Efficient TDE evaluation by inclusion of guess start energy two standard deviations below predicted TDE, if lower than 1 eV
        # otherwise start from 1 eV
        guess_start_energy = max(int(round(pred_mean[0].item() - (pred_var[0].sqrt().item() * 2))), 1)
        # New TDE and new Noise
        new_y, new_n = [], []
        
        val, noise = evaluate_tde(cfg=cfg,u=new_pts[0], start_energy=guess_start_energy)
        if it == start_it:
            print(f"{'Iter':>5} | {'Evaluated TDE (eV)':>20} | {'Predicted TDE (eV)':>22} | {'IMSE':>8}")
            print("-" * 70)

        # Print nice summary
        print(
            f"{it:5d} | "
            f"{val:20.2f} | "
            f"{pred_mean[0].item():10.2f} ± {np.sqrt(pred_var[0].item())*2:6.2f} eV | "
            f"{imse:8.2f}"
        )
        new_y.append(val)
        new_n.append(noise)
        # Add new data to train arrays
        X_train = torch.cat([X_train, new_pts])
        y_train = torch.cat([y_train, torch.tensor(new_y)])
        noise_train = torch.cat([noise_train, torch.tensor(new_n)])
        likelihood.noise = noise_train.clone()
        # Renormalise data 
        y_mean = y_train.mean()
        y_std = y_train.std()
        y_train_std = (y_train - y_mean) / y_std

        # Retrain model
        model.set_train_data(X_train, y_train_std, strict=False)
        optimise_model(model, X_train, y_train_std, likelihood, mll)
        # Create arrays for evaluations at probe points
        # mean_pred, var_pred = query_model(model=model, likelihood=likelihood, probe_pts=probe, y_std=y_std, y_mean=y_mean)
        
        # np.savetxt('probe_points.txt', probe.numpy())
        # np.savetxt('tde_points.txt', mean_pred.numpy())
        # np.savetxt('var_points.txt', var_pred.numpy())


    # Evaluate trained GP on all probe points
    mean_pred, var_pred = query_model(model=model, likelihood=likelihood, probe_pts=probe, y_std=y_std, y_mean=y_mean)

    X_train_np = probe.numpy()
    train_y_np = mean_pred.numpy()
    # Save vectors
    np.savetxt('probe_points.txt', probe.numpy())
    # Save TDE in eV
    np.savetxt('tde_points.txt', mean_pred.numpy())
    # Save variance in eV**2
    np.savetxt('var_points.txt', var_pred.numpy())
    # If plot argument is used, plot on all probe points
    if plot:
        #Plot TDE
        Plot_Tools(X_train_np, train_y_np, theta_range=theta_range, phi_range=phi_range).plot(plot_type="mean", fname='TDEMAP.png')
        #Plot variance
        Plot_Tools(X_train_np, var_pred.numpy(), theta_range=theta_range, phi_range=phi_range).plot(plot_type="var", fname='VARMAP.png')
    

 

# if __name__ == "__main__":
#     generate_lammps_input(masses = [(1, 55.845)], pair_styles = ['pair_style hybrid/overlay eam/fs tabgap'], 
#                            pair_coeffs = ['pair_coeff * * eam/fs ../Fe-2021-04-06.eam.fs Fe', 
#                                           'pair_coeff * * tabgap ../Fe-2021-04-06.tabgap Fe no yes'],
#                              units = 'metal', atom_style = 'atomic', read_data = 'Fe.data', run_steps = 2000, PKA_id=4367)

#     cfg = Config(run_line='mpirun /storage/hpc/51/dickson5/codes/tablammps/lammps_w_hdf5/build/lmp -in tde.in',
#                                init_n=200, max_iters=1400)

#     demap(theta_range=(0, np.pi/4), phi_range=(0, np.pi/4), restart=False, cfg=cfg, plot=False)

