import os
import torch
import torch.nn.functional as F
from codebase.utils import normalize
from codebase import setup
from codebase.classifiers import vgg
import numpy as np

def load_target_model(exp_cfg):
    """
    This function loads the target model based on the configuration settings. 
    If a model is provided in exp_cfg.target_model, it is used directly. 
    Otherwise, it loads the model from the most recent checkpoint available in 
    the target_model folder inside the output directory. The model is then 
    moved to the specified device and set to evaluation mode.
    """
    device = exp_cfg.device
    if hasattr(exp_cfg, "target_model") and exp_cfg.target_model is not None:
        return exp_cfg.target_model.to(device).eval()
    
    # Search for checkpoint files in the specified directory
    ckpt_dir = exp_cfg.out_dir.joinpath("target_model")
    ckpt_files = list(ckpt_dir.glob("*.pth")) + list(ckpt_dir.glob("*.pt")) + list(ckpt_dir.glob("*.tar"))
    if len(ckpt_files) == 0:
        raise RuntimeError(f"No checkpoint file found in {ckpt_dir}.")
    
    # Select the most recent checkpoint file
    ckpt_file = max(ckpt_files, key=lambda x: x.stat().st_mtime)
    
    # Load the checkpoint in a safe manner
    with torch.serialization.safe_globals([np.core.multiarray.scalar]):
        dic_saved = torch.load(str(ckpt_file), map_location="cpu", weights_only=False)
    
    # Initialize the model and load its weights from the checkpoint
    model = vgg.vgg11_bn(num_classes=10)
    model.load_state_dict(dic_saved["model_state"])
    torch.cuda.empty_cache()
    
    # Transfer model to the device and set it to evaluation mode
    model = model.to(device).eval()
    return model

def generate_attack(x_arr: torch.Tensor, 
                    y_arr: torch.Tensor, 
                    adv_target_arr: torch.Tensor, 
                    adv_l_inf: float, 
                    exp_cfg, 
                    batch_size: int = 10,
                    num_restarts: int = 3) -> torch.Tensor:
    """
    exp: 
    This function generates adversarial examples targeting a specific label using a 
    grey-box attack technique. The method applies finite-difference approximation 
    to estimate gradients and incorporates momentum and hybrid updates. It also 
    supports multiple random restarts per mini-batch to improve the chances of success.
    
    Parameters added:
      x_arr         - Tensor of original input images [N, 3, H, W].
      y_arr         - Tensor of true labels [N], though unused here.
      adv_target_arr- Tensor of target labels [N] the attack aims to force the model to predict.
      adv_l_inf     - Maximum allowed perturbation (L∞ norm) for each input image.
      exp_cfg       - Configuration object that contains device settings and paths.
      batch_size    - The number of images to process per batch.
      num_restarts  - Number of random restarts to apply per batch to avoid local minima.
    
    Returns:
      A tensor of adversarial examples designed to misclassify the inputs as the target labels.
    """
    device = exp_cfg.device
    model = load_target_model(exp_cfg)  # Retrieve the target model for attack
    model.eval()  # Ensure model is in evaluation mode
    
    # Define the mean and standard deviation for CIFAR-10 normalization
    mean = torch.tensor(setup.CIFAR10_MEAN, dtype=torch.float32, device=device).view(1,3,1,1)
    std  = torch.tensor(setup.CIFAR10_STD, dtype=torch.float32, device=device).view(1,3,1,1)
    
    # Set attack hyperparameters
    num_steps = 250            # Total number of steps for each restart.
    n_samples = 50             # Number of samples to use for finite-difference gradient estimation.
    initial_step_size = adv_l_inf / 250  # Starting value for the step size.
    sigma = 0.001              # Noise scale for finite differences.
    beta = 0.95                # Decay factor for momentum updates.
    alpha = 0.6                # Weighting factor for combining momentum with its sign.
    
    adv_x_batches = []
    N = x_arr.size(0)  # Total number of samples
    
    # Process the images in mini-batches for efficient computation
    for i in range(0, N, batch_size):
        x_batch = x_arr[i:i+batch_size].to(device).float()
        adv_target_batch = adv_target_arr[i:i+batch_size].to(device).long()
        
        best_delta = None
        best_fool_rate = 0.0
        
        # Perform multiple restarts per batch to increase the likelihood of success.
        for r in range(num_restarts):
            delta = torch.zeros_like(x_batch, device=device)  # Initialize perturbation for this restart
            momentum = torch.zeros_like(x_batch, device=device)  # Initialize momentum term
            
            # Perform the iterative update for each step
            for t in range(num_steps):
                # Step size decays exponentially as the iterations progress
                step_size = initial_step_size * np.exp(-t / 150.0)
                
                with torch.no_grad():
                    # Apply perturbation to the batch and normalize the images
                    x_adv = torch.clamp(x_batch + delta, 0, 1)
                    x_normed = normalize(x_adv, mean, std)
                    
                    grad_estimate = torch.zeros_like(delta, device=device)
                    # Use finite-difference method to estimate the gradient
                    for _ in range(n_samples):
                        u = torch.empty_like(delta).uniform_(-1, 1).sign()  # Random perturbation directions
                        x_adv_plus = torch.clamp(x_batch + delta + sigma * u, 0, 1)
                        x_adv_minus = torch.clamp(x_batch + delta - sigma * u, 0, 1)
                        
                        # Compute the loss for perturbed images and estimate the gradient
                        loss_plus = F.cross_entropy(model(normalize(x_adv_plus, mean, std)),
                                                    adv_target_batch, reduction="none")
                        loss_minus = F.cross_entropy(model(normalize(x_adv_minus, mean, std)),
                                                     adv_target_batch, reduction="none")
                        
                        grad_diff = ((loss_plus - loss_minus) / (2 * sigma)).view(-1, 1, 1, 1) * u
                        grad_estimate += grad_diff
                    grad_estimate /= n_samples
                    
                    # Update momentum and combine with the sign of the gradient for the final update
                    momentum = beta * momentum + grad_estimate
                    update = alpha * momentum + (1 - alpha) * momentum.sign()
                    delta = delta - step_size * update
                    
                    # Ensure the perturbation stays within the L∞ norm constraint
                    delta = torch.clamp(delta, -adv_l_inf, adv_l_inf)
                    delta = (x_batch + delta).clamp(0, 1) - x_batch
            
            # After performing all steps, evaluate the effectiveness of this perturbation
            with torch.no_grad():
                # Apply the final perturbation and evaluate predictions
                x_adv_final = torch.clamp(x_batch + delta, 0, 1)
                normed_final = normalize(x_adv_final, mean, std)
                preds = model(normed_final).argmax(dim=1)
                
                # Calculate the fooling rate as the fraction of images misclassified as the target
                fool_rate = (preds == adv_target_batch).float().mean().item()
            
            # If this restart yields a higher fooling rate, store the current perturbation
            if fool_rate > best_fool_rate:
                best_fool_rate = fool_rate
                best_delta = delta.clone()
            
            # Optionally, stop if perfect fooling is achieved
            if best_fool_rate >= 1.0:
                break
        
        # Add the best adversarial examples from this batch to the final list
        adv_x_batch = torch.clamp(x_batch + best_delta, 0, 1)
        adv_x_batches.append(adv_x_batch)
        torch.cuda.empty_cache()  # Clear memory
    
    # Concatenate all adversarial examples and return them
    adv_x = torch.cat(adv_x_batches, dim=0)
    return adv_x.detach()
