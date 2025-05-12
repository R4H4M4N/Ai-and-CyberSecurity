import torch
import torch.nn.functional as F
from torchvision import transforms
from codebase import setup, utils

def generate_attack(
    target_model: torch.nn.Module,
    x_arr: torch.Tensor,
    y_arr: torch.Tensor,
    adv_target_arr: torch.Tensor,
    adv_l_inf: float,
    exp_cfg,
) -> torch.Tensor:
    """
    Generates targeted adversarial examples using PGD with EOT against a stochastic defense
    (e.g., RandomResizedCrop) under an L-infinity constraint.
    """

    # Load device and normalization constants
    device = exp_cfg.device
    mean = torch.tensor(setup.CIFAR10_MEAN, dtype=torch.float32).reshape(1, 3, 1, 1).to(device)
    std = torch.tensor(setup.CIFAR10_STD, dtype=torch.float32).reshape(1, 3, 1, 1).to(device)

    # Clone inputs to avoid modifying original tensors
    x_arr = x_arr.clone().detach().to(device)
    adv_target_arr = adv_target_arr.to(device)

    # PGD parameters
    num_steps = 250           # Total number of PGD iterations
    eot_samples = 60          # Number of EOT samples (for stochastic defense)
    epsilon = 1e-6            # Tiny constant to avoid numerical instability

    # Step size scheduler using cosine annealing
    def cosine_lr(step, total_steps, max_lr):
        return max_lr * 0.5 * (1 + torch.cos(torch.tensor(step / total_steps * 3.141592)))

    max_step_size = 0.008     # Maximum learning rate (step size)
    momentum = 0.9            # Momentum factor for gradient smoothing
    velocity = torch.zeros_like(x_arr).to(device)  # Initialize momentum buffer

    # Initialize perturbation with random noise within L∞ bound
    delta = ((torch.rand_like(x_arr) * 2 - 1) * adv_l_inf).clamp(-adv_l_inf + epsilon, adv_l_inf - epsilon)
    delta.requires_grad_()

    # Define the stochastic defense: RandomResizedCrop
    transform_fn = transforms.RandomResizedCrop(size=32, scale=(0.20, 0.50))

    # Begin PGD loop
    for step in range(num_steps):
        total_loss = 0.0

        # Expectation Over Transformation (EOT): average over stochastic defenses
        for _ in range(eot_samples):
            # Apply transformation to each image in the batch independently
            transformed_x = torch.stack([
                transform_fn((x + delta[i]).clamp(0, 1)) for i, x in enumerate(x_arr)
            ])

            # Normalize transformed images before feeding to model
            normed_input = utils.normalize(transformed_x, mean, std)
            logits = target_model(normed_input)

            # Composite loss: encourage high target class logit + suppress others
            target_logit = logits[range(len(logits)), adv_target_arr]
            mask = F.one_hot(adv_target_arr, num_classes=10).bool()
            max_other_logit = logits.masked_fill(mask, float('-inf')).max(dim=1).values

            # Margin loss encourages separation between target and other logits
            margin_loss = -(target_logit - max_other_logit).mean()
            ce_loss = F.cross_entropy(logits, adv_target_arr)

            # Combine both losses
            loss = 0.5 * margin_loss + 0.5 * ce_loss
            total_loss += loss

        # Average the loss over EOT samples and backpropagate
        total_loss = total_loss / eot_samples
        total_loss.backward()

        # Compute adaptive step size for current step
        step_size = cosine_lr(step, num_steps, max_step_size)

        # Apply momentum-based PGD update
        grad = delta.grad.data
        velocity = momentum * velocity + grad / grad.abs().mean(dim=(1,2,3), keepdim=True).clamp(min=1e-8)
        delta.data -= step_size * velocity.sign()

        # Project perturbation back into valid L∞ ball and [0, 1] image range
        delta.data = torch.clamp(delta.data, -adv_l_inf, adv_l_inf)
        delta.data = torch.clamp(x_arr + delta.data, 0.0, 1.0) - x_arr

        # Reset gradients for next iteration
        delta.grad.zero_()

    # Return final adversarial examples (perturbed inputs)
    adv_x = (x_arr + delta.detach()).clamp(0.0, 1.0)
    return adv_x
