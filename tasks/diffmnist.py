import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import functools

from torch.optim import Adam
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
import tqdm
from torch import autocast 
from torch.cuda.amp import GradScaler 

from scipy import integrate
import matplotlib.pyplot as plt
import time

from torchvision.utils import make_grid

class DiffMNIST:
    def __init__(self, batch_size_train = 128,
                           num_workers = 4,
                           physical_batchsize = 1024,
                           precison = "amp_half") -> None:
        
        self.batch_size_train = batch_size_train
        self.physical_batchsize = physical_batchsize

        train_dataset = MNIST('.', train=True, transform=transforms.ToTensor(), download=True)
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, num_workers=4)

        self.train_data_len = len(train_dataset)

        self.precison = precison

        if self.precison == "amp_half":
            self.scaler = GradScaler()

        self.start_time = time.time()
        self.avg_loss = 0
        self.num_items = 0

    def get_train_loader(self) -> DataLoader:
        return self.train_loader

    def get_train_data_len(self) -> int:
        return self.train_data_len 

    def loss_and_step(self, model, optimizer, batch_data, device) -> float:
        images, _ = batch_data

        batch_data_len = len(images)

        if self.physical_batchsize < batch_data_len:

            if (batch_data_len % self.physical_batchsize) == 0:
                num_steps = (batch_data_len // self.physical_batchsize)  
            else: 
                num_steps = (batch_data_len // self.physical_batchsize)   + 1
                
            for j in range(num_steps):
                if j == num_steps - 1:
                    input_var = images[j*self.physical_batchsize:].to(device)
                else:
                    input_var = images[(j*self.physical_batchsize):((j+1)*self.physical_batchsize)].to(device)

                input_var = input_var.half() if self.precison == "half" else input_var

                if self.precison == "amp_half":
                    with autocast(device_type='cuda', dtype=torch.float16):
                        # compute output
                        loss = loss_fn(model, input_var, model.marginal_prob_std_fn)
                    self.scaler.scale(loss).backward()

                else:
                    loss = loss_fn(model, input_var, model.marginal_prob_std_fn)
                    loss.backward()

                self.avg_loss += loss.item() * input_var.shape[0]
                self.num_items += input_var.shape[0]

        else:
            input_var = images.to(device)
            input_var = input_var.half() if self.precison == "half" else input_var
            
            if self.precison == "amp_half":
                with autocast(device_type='cuda', dtype=torch.float16):
                    # compute output
                    loss = loss_fn(model, input_var, model.marginal_prob_std_fn)
                self.scaler.scale(loss).backward()
            else:
            # compute output
                loss = loss_fn(model, input_var, model.marginal_prob_std_fn)
                loss.backward()

            self.avg_loss += loss.item() * input_var.shape[0]
            self.num_items += input_var.shape[0]

        if self.precison == "amp_half":
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            optimizer.step()  

        return loss.item()

    def eval_model_with_log(self,model, writer, ep, steps, lr_now, device) -> None:
        dt = {
            'epoch': ep,
            'avg_loss': self.avg_loss / self.num_items,
            'lr': lr_now,
            'time': time.time() - self.start_time}

        print("epoch:", ep, "steps:",steps, "avg_loss:", dt["avg_loss"], "lr", lr_now, "time", dt["time"])

        if writer is not None:
            
            for k, v in dt.items():
                writer.add_scalar("stats/" + k, v, steps)

        sample_size = 16
        samples = Euler_Maruyama_sampler(model, 
                  model.marginal_prob_std_fn,
                  model.diffusion_coeff_fn, 
                  sample_size, 
                  device=device)
        samples = samples.clamp(0.0, 1.0)

        sample_grid = make_grid(samples, nrow= 4)
        writer.add_image('sampled_images', sample_grid.cpu(), steps)

        return dt

        

def loss_fn(model, x, marginal_prob_std, eps=1e-5):
    """The loss function for training score-based generative models.

    Args:
        model: A PyTorch model instance that represents a 
        time-dependent score-based model.
        x: A mini-batch of training data.    
        marginal_prob_std: A function that gives the standard deviation of 
        the perturbation kernel.
        eps: A tolerance value for numerical stability.
    """
    random_t = torch.rand(x.shape[0], device=x.device) * (1. - eps) + eps  
    z = torch.randn_like(x)
    std = marginal_prob_std(random_t)
    perturbed_x = x + z * std[:, None, None, None]
    score = model(perturbed_x, random_t)
    loss = torch.mean(torch.sum((score * std[:, None, None, None] + z)**2, dim=(1,2,3)))
    return loss



def ode_sampler(score_model,
                marginal_prob_std,
                diffusion_coeff,
                batch_size=64, 
                atol=1e-5, 
                rtol=1e-5, 
                device='cuda', 
                z=None,
                eps=1e-3):
    """Generate samples from score-based models with black-box ODE solvers.

    Args:
    score_model: A PyTorch model that represents the time-dependent score-based model.
    marginal_prob_std: A function that returns the standard deviation 
        of the perturbation kernel.
    diffusion_coeff: A function that returns the diffusion coefficient of the SDE.
    batch_size: The number of samplers to generate by calling this function once.
    atol: Tolerance of absolute errors.
    rtol: Tolerance of relative errors.
    device: 'cuda' for running on GPUs, and 'cpu' for running on CPUs.
    z: The latent code that governs the final sample. If None, we start from p_1;
        otherwise, we start from the given z.
    eps: The smallest time step for numerical stability.
    """
    t = torch.ones(batch_size, device=device)
    # Create the latent code
    if z is None:
        init_x = torch.randn(batch_size, 1, 28, 28, device=device) \
        * marginal_prob_std(t)[:, None, None, None]
    else:
        init_x = z
    
    shape = init_x.shape

    def score_eval_wrapper(sample, time_steps):
        """A wrapper of the score-based model for use by the ODE solver."""
        sample = torch.tensor(sample, device=device, dtype=torch.float32).reshape(shape)
        time_steps = torch.tensor(time_steps, device=device, dtype=torch.float32).reshape((sample.shape[0], ))    
        with torch.no_grad():    
            score = score_model(sample, time_steps)
        return score.cpu().numpy().reshape((-1,)).astype(np.float64)
  
    def ode_func(t, x):        
        """The ODE function for use by the ODE solver."""
        time_steps = np.ones((shape[0],)) * t    
        g = diffusion_coeff(torch.tensor(t)).cpu().numpy()
        return  -0.5 * (g**2) * score_eval_wrapper(x, time_steps)
  
    # Run the black-box ODE solver.
    res = integrate.solve_ivp(ode_func, (1., eps), init_x.reshape(-1).cpu().numpy(), rtol=rtol, atol=atol, method='RK45')  
    #print(f"Number of function evaluations: {res.nfev}")
    x = torch.tensor(res.y[:, -1], device=device).reshape(shape)

    return x

def Euler_Maruyama_sampler(score_model, 
                           marginal_prob_std,
                           diffusion_coeff, 
                           batch_size=64, 
                           num_steps=1000, 
                           device='cuda', 
                           eps=1e-3):
    """Generate samples from score-based models with the Euler-Maruyama solver.

    Args:
    score_model: A PyTorch model that represents the time-dependent score-based model.
    marginal_prob_std: A function that gives the standard deviation of
        the perturbation kernel.
    diffusion_coeff: A function that gives the diffusion coefficient of the SDE.
    batch_size: The number of samplers to generate by calling this function once.
    num_steps: The number of sampling steps. 
        Equivalent to the number of discretized time steps.
    device: 'cuda' for running on GPUs, and 'cpu' for running on CPUs.
    eps: The smallest time step for numerical stability.

    Returns:
    Samples.    
    """
    t = torch.ones(batch_size, device=device)
    init_x = torch.randn(batch_size, 1, 28, 28, device=device) \
    * marginal_prob_std(t)[:, None, None, None]
    time_steps = torch.linspace(1., eps, num_steps, device=device)
    step_size = time_steps[0] - time_steps[1]
    x = init_x
    with torch.no_grad():
        for time_step in time_steps:      
            batch_time_step = torch.ones(batch_size, device=device) * time_step
            g = diffusion_coeff(batch_time_step)
            mean_x = x + (g**2)[:, None, None, None] * score_model(x, batch_time_step) * step_size
            x = mean_x + torch.sqrt(step_size) * g[:, None, None, None] * torch.randn_like(x)      
    # Do not include any noise in the last sampling step.
    return mean_x

def pc_sampler(score_model, 
               marginal_prob_std,
               diffusion_coeff,
               batch_size=64, 
               num_steps=500, 
               snr=0.16,                
               device='cuda',
               eps=1e-3):
    """Generate samples from score-based models with Predictor-Corrector method.

    Args:
    score_model: A PyTorch model that represents the time-dependent score-based model.
    marginal_prob_std: A function that gives the standard deviation
        of the perturbation kernel.
    diffusion_coeff: A function that gives the diffusion coefficient 
        of the SDE.
    batch_size: The number of samplers to generate by calling this function once.
    num_steps: The number of sampling steps. 
        Equivalent to the number of discretized time steps.    
    device: 'cuda' for running on GPUs, and 'cpu' for running on CPUs.
    eps: The smallest time step for numerical stability.

    Returns: 
    Samples.
    """
    t = torch.ones(batch_size, device=device)
    init_x = torch.randn(batch_size, 1, 28, 28, device=device) * marginal_prob_std(t)[:, None, None, None]
    time_steps = np.linspace(1., eps, num_steps)
    step_size = time_steps[0] - time_steps[1]
    x = init_x
    with torch.no_grad():
        for time_step in time_steps:      
            batch_time_step = torch.ones(batch_size, device=device) * time_step
            # Corrector step (Langevin MCMC)
            grad = score_model(x, batch_time_step)
            grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
            noise_norm = np.sqrt(np.prod(x.shape[1:]))
            langevin_step_size = 2 * (snr * noise_norm / grad_norm)**2
            x = x + langevin_step_size * grad + torch.sqrt(2 * langevin_step_size) * torch.randn_like(x)      

            # Predictor step (Euler-Maruyama)
            g = diffusion_coeff(batch_time_step)
            x_mean = x + (g**2)[:, None, None, None] * score_model(x, batch_time_step) * step_size
            x = x_mean + torch.sqrt(g**2 * step_size)[:, None, None, None] * torch.randn_like(x)      

    # The last step does not include any noise
    return x_mean