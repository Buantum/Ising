import torch
import matplotlib.pyplot as plt

def initialize_spin_lattice(n, p=0.5, device='cpu'):
    """
    Initialize a spin lattice with random spins.
    
    Parameters:
    n (int): Size of the lattice (n x n).
    p (float): Probability of spin being 1.
    device (str): Device to use ('cpu' or 'cuda').
    
    Returns:
    torch.Tensor: Initialized spin lattice.
    """
    lattice = torch.bernoulli(torch.full((n, n), p)).to(torch.int8) * 2 - 1
    return lattice.to(device)

def calculate_energy(lattice, J=1.0):
    """
    Calculate the total energy of the lattice.
    
    Parameters:
    lattice (torch.Tensor): Spin lattice.
    J (float): Interaction strength.
    
    Returns:
    float: Total energy of the lattice.
    """
    neighbors_sum = (
        torch.roll(lattice, shifts=1, dims=0) +
        torch.roll(lattice, shifts=-1, dims=0) +
        torch.roll(lattice, shifts=1, dims=1) +
        torch.roll(lattice, shifts=-1, dims=1)
    )
    energy = -J * torch.sum(lattice * neighbors_sum) / 2  # Each pair counted twice
    return energy.item()

def ising_update(lattice, beta=1.0):
    """
    Perform a single update of the Ising model using the checkerboard algorithm.
    
    Parameters:
    lattice (torch.Tensor): Spin lattice.
    beta (float): Inverse temperature (1/kT).
    """
    n = lattice.size(0)
    
    # Update black and white sites separately
    for color in [0, 1]:
        indices = torch.stack(torch.meshgrid(
            torch.arange(n, device=lattice.device), 
            torch.arange(n, device=lattice.device),
            indexing='ij'
        ), dim=-1)
        indices = indices[(indices.sum(dim=-1) % 2) == color]
        i, j = indices[:, 0], indices[:, 1]
        
        neighbors_sum = (
            lattice[(i+1) % n, j] +
            lattice[(i-1) % n, j] +
            lattice[i, (j+1) % n] +
            lattice[i, (j-1) % n]
        )
        
        delta_E = 2 * lattice[i, j] * neighbors_sum
        probability = torch.exp(-delta_E/beta)
        flip_condition = (delta_E < 0) | (torch.rand(len(i), device=lattice.device) < torch.exp(-delta_E / beta))
        
        flip = flip_condition.to(lattice.dtype) * -2 + 1
        lattice[i, j] *= flip


def simulate_ising(n, steps, beta=1.0, device='cpu'):
    """
    Simulate the Ising model.
    
    Parameters:
    n (int): Size of the lattice (n x n).
    steps (int): Number of simulation steps.
    beta (float): Inverse temperature (1/kT).
    device (str): Device to use ('cpu' or 'cuda').
    
    Returns:
    torch.Tensor: Final spin lattice.
    """
    lattice = initialize_spin_lattice(n, device=device)
    for _ in range(steps):
        ising_update(lattice, beta)
    return lattice

def plot_lattice(lattice):
    """
    Plot the Ising model lattice.
    
    Parameters:
    lattice (torch.Tensor): Spin lattice.
    """
    plt.imshow(lattice.cpu(), cmap='gray', vmin=-1, vmax=1)
    #plt.colorbar(ticks=[-1, 1], label='Spin')
    plt.title('Ising Model Lattice')
    plt.show()

# Parameters
n = 1000  # Lattice size
steps = 2000  # Number of steps
beta = 1  # temperature
#device = 'cuda' if torch.cuda.is_available() else 'cpu'  # Use CUDA if available
device = 'cuda'
# Simulate Ising model
# Initialize lists to store results
import numpy as np
temperature_range = np.linspace(0.5, 5.0, 1)
temperatures = []
magnetizations = []

# Function to calculate total magnetization
def calculate_magnetization(lattice):
    """
    Calculate the total magnetization of the lattice.
    
    Parameters:
    lattice (torch.Tensor): Spin lattice.
    
    Returns:
    float: Total magnetization of the lattice.
    """
    return torch.sum(lattice)
from torch import vmap
def simulate_ising_vmap(lattices, steps, beta):
    """
    Simulate the Ising model using vmap for parallel updates.
    
    Parameters:
    lattices (torch.Tensor): Batch of spin lattices.
    steps (int): Number of simulation steps.
    beta (float): Inverse temperature (1/kT).
    
    Returns:
    torch.Tensor: Final batch of spin lattices.
    """
    def single_simulation(lattice):
        for _ in range(steps):
            ising_update(lattice, beta)
        return lattice

    return vmap(single_simulation, randomness='different')(lattices)


lattice=simulate_ising(100, steps, beta=1, device='cuda')
for i in range(100):

    lattice=simulate_ising(100, steps, beta=1, device='cuda')
    plt.imshow(lattice.cpu(), cmap='gray', vmin=-1, vmax=1)
    #plt.colorbar(ticks=[-1, 1], label='Spin')
    plt.title('Ising Model Lattice')
    plt.savefig('ising'+str(i)+'.png')
    plt.close()
    print(i)