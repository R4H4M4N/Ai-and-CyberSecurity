"""
Modified https://github.com/Harry24k/PGD-pytorch/blob/master/PGD.ipynb
"""

import torch

def project(x, original_x, epsilon, proj_type='l_inf'):

    if proj_type == 'l_inf':
        max_x = original_x + epsilon
        min_x = original_x - epsilon

        x = torch.max(torch.min(x, max_x), min_x)

    elif proj_type == 'l_2':
        dist = x - original_x
        dist = dist.view(x.shape[0], -1)
        dist_norm = torch.norm(dist, dim=1, keepdim=True)

        # figure out which perturbations need to be changed
        mask = (dist_norm > epsilon).unsqueeze(2).unsqueeze(3)

        # calculate the maximum allowed distance without changing direction.
        dist_bound = (dist / dist_norm) * epsilon
        dist_bound = dist_bound.view(x.shape)

        # use the maximum allowed perturbations for those exceeding the bounds.
        x = (original_x + dist_bound) * mask.float() + x * (1 - mask.float())

    else:
        raise NotImplementedError

    return x


def pgd_attack(
        model,
        images,                 # Clean images
        preprocess_func,        # function to preprocess images before inputing to the model. Can be set to None.
        targeted_attack,        # True for targeted attack. False for untargeted attack.
        labels,                 # Target labels for targeted attack. Ground truths labels for untargeted attack.
        loss,                   # The loss function to train the model
        eps,                    # Bounds of perturbations
        alpha,                  # Value multiplied by the gradient sign
        iters,                  # Number of iterations to run
        random_start_eps,       # The size of uniform noise that is added to the clean images. Set it to None if not needed.
        clip_min=0,
        clip_max=1,
        proj_type='l_inf',       # "l_inf" or "l_2" for bounding perturbations
):
    # Save original images for bounding perturbations
    org_images = images.clone()

    # The adversaries created from random close points to the original data
    if random_start_eps is not None:
        rand_perturb = (torch.rand(images.shape) * 2 - 1) * random_start_eps
        rand_perturb = rand_perturb.to(images.get_device())
        images = images + rand_perturb
        images = torch.clamp(images, min=clip_min, max=clip_max)

    for i in range(iters):
        images.requires_grad = True

        if preprocess_func is None:
            outputs = model(images)
        else:
            outputs = model(preprocess_func(images))

        # Set gradients of the model and images to 0.
        # Otherwise, gradients will accumulate.
        model.zero_grad()
        assert images.grad is None, "detached images should not have gradients."

        cost = loss(outputs, labels)
        cost.backward()

        if targeted_attack:
            adv_images = images - alpha * images.grad.sign()
        else:
            adv_images = images + alpha * images.grad.sign()

        # Bound the perturbations
        images = project(adv_images, org_images, eps, proj_type)
        images = torch.clamp(images, min=0, max=1)
        images = images.detach()

    return images







