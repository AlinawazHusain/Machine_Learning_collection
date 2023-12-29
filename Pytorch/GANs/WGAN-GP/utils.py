import torch

def GradientPanelty(real , fake , critic , device = 'cpu'):
    B , C , H , W = real.shape

    alpha = torch.rand((B , 1 , 1, 1)).repeat(1 , C , H , W).to(device)

    interpolated = real*alpha + fake*(1-alpha)

    mixed_score = critic(interpolated)

    gradient = torch.autograd.grad(
        inputs = interpolated,
        outputs = mixed_score,
        create_graph = True,
        retain_graph = True,
        grad_outputs = torch.ones_like(mixed_score)
    )[0]
    
    gradient = gradient.view(gradient.shape[0] , -1)
    gradient_norm = gradient.norm(2 , dim = 1)
    gradient_panelty = torch.mean((gradient_norm-1)**2)
    return gradient_panelty
