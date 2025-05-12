import torch
import torch.nn.functional as F

def generate_UAPs(
        target_model: torch.nn.Module,
        x_arr: torch.Tensor,
        y_arr: torch.Tensor,
        UAP_target: int,
        UAP_l_inf: float,
        exp_cfg,
) -> torch.Tensor:
    """
    Generate targeted universal adversarial perturbations (UAPs) to fool a target model.
    """
    # Ensure model is in evaluation mode
    target_model.eval()
    
    # Get dimensions from the input images
    N, C, H, W = x_arr.shape
    
    # For normalization
    from codebase import utils, setup
    cifar10_mean_tensor = torch.Tensor(setup.CIFAR10_MEAN).reshape([1, 3, 1, 1]).to(exp_cfg.device)
    cifar10_std_tensor = torch.Tensor(setup.CIFAR10_STD).reshape([1, 3, 1, 1]).to(exp_cfg.device)
    
    # Best settings based on previous runs
    alpha = 0.01  # Learning rate
    num_iterations = 800
    
    # Initialize several different UAPs and pick the best one
    num_trials = 5
    best_uap = None
    best_fool_rate = 0.0
    
    for trial in range(num_trials):
        # Different starting points for different trials
        if trial == 0:
            # Start with zeros
            uap = torch.zeros((C, H, W), device=exp_cfg.device)
        elif trial == 1:
            # Start with full positive values
            uap = torch.ones((C, H, W), device=exp_cfg.device) * UAP_l_inf
        elif trial == 2:
            # Start with alternating patterns
            uap = torch.ones((C, H, W), device=exp_cfg.device) * UAP_l_inf
            uap[:, ::2, ::2] *= -1  # Checkerboard pattern
        else:
            # Start with random values
            uap = (torch.rand((C, H, W), device=exp_cfg.device) * 2 - 1) * UAP_l_inf
        
        # Current trial's best
        trial_best_uap = uap.clone()
        trial_best_rate = 0.0
        
        # Optimization with momentum
        momentum = torch.zeros_like(uap)
        momentum_factor = 0.9
        
        for i in range(num_iterations):
            uap = uap.detach().requires_grad_(True)
            
            # Apply UAP to all images
            adv_images = torch.clamp(x_arr + uap, 0, 1)
            
            # Normalize images
            norm_adv_images = utils.normalize(adv_images, cifar10_mean_tensor, cifar10_std_tensor)
            
            # Forward pass
            logits = target_model(norm_adv_images)
            
            # Get logits for all classes
            target_logits = logits[:, UAP_target]
            
            # Others are all non-target classes
            other_logits = logits.clone()
            other_logits[:, UAP_target] = -1000.0
            max_other_logits = other_logits.max(dim=1)[0]
            
            # We want target logits to be much higher than others
            margin = 20.0
            loss = torch.clamp(max_other_logits - target_logits + margin, min=0).mean()
            
            # Backward pass
            loss.backward()
            
            # Update with momentum
            with torch.no_grad():
                if uap.grad is not None:
                    # Update momentum
                    momentum = momentum_factor * momentum + uap.grad
                    
                    # Take step with momentum
                    uap = uap - alpha * momentum.sign()
                    
                    # Project back to L∞ constraint
                    uap = torch.clamp(uap, -UAP_l_inf, UAP_l_inf)
            
            # Check every 50 iterations
            if (i + 1) % 50 == 0:
                with torch.no_grad():
                    adv_images = torch.clamp(x_arr + uap, 0, 1)
                    norm_adv_images = utils.normalize(adv_images, cifar10_mean_tensor, cifar10_std_tensor)
                    preds = target_model(norm_adv_images).argmax(dim=1)
                    fool_rate = (preds == UAP_target).float().mean().item()
                    
                    # Track best for this trial
                    if fool_rate > trial_best_rate:
                        trial_best_rate = fool_rate
                        trial_best_uap = uap.clone()
                    
                    # Early stopping if perfect
                    if fool_rate >= 1.0:
                        break
        
        # After trial completes, check if this is the best overall
        if trial_best_rate > best_fool_rate:
            best_fool_rate = trial_best_rate
            best_uap = trial_best_uap.clone()
    
    # Return the best UAP found across all trials
    return best_uap