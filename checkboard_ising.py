import torch
import matplotlib.pyplot as plt
import numpy as np
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

# Parameters
steps = 200000  # Number of steps
temperature_range = np.linspace(1, 3, 40)
device = 'cuda'

# Initialize lists to store results
temperatures_all = []
magnetizations_all = []
lattice_sizes_all = []

# Lattice sizes to simulate
lattice_sizes = [16, 32, 64, 128]

# Run simulations for each lattice size and temperature
num_samples = 100
for n in lattice_sizes:
    for T in temperature_range:
        beta = T
        lattices = torch.stack([initialize_spin_lattice(n, device=device) for _ in range(num_samples)])
        final_lattices = simulate_ising_vmap(lattices, steps, beta)
        magnetizations_batch = torch.vmap(calculate_magnetization)(final_lattices)
        normalized_magnetizations = magnetizations_batch / (n**2)
        
        temperatures_all.extend([T] * num_samples)
        magnetizations_all.extend(normalized_magnetizations.cpu().numpy())
        lattice_sizes_all.extend([n] * num_samples)
        print(f"Lattice Size: {n}, Temperature: {T:.2f}, Average Magnetization: {normalized_magnetizations.mean():.4f}")

# Convert lists to numpy arrays for easier manipulation
temperatures_all = np.array(temperatures_all)
magnetizations_all = np.array(magnetizations_all)
lattice_sizes_all = np.array(lattice_sizes_all)

# Initialize lists to store average and standard deviation values for M^2
m2_avg_all = {n: [] for n in lattice_sizes}
m2_std_all = {n: [] for n in lattice_sizes}

# Calculate average and standard deviation of M^2 for each temperature and lattice size
for n in lattice_sizes:
    for T in temperature_range:
        indices = (temperatures_all == T) & (lattice_sizes_all == n)
        
        if indices.any():
            m2_values = magnetizations_all[indices]**2
            
            m2_avg_all[n].append(np.mean(m2_values))
            m2_std_all[n].append(np.std(m2_values))
        else:
            m2_avg_all[n].append(np.nan)
            m2_std_all[n].append(np.nan)

# Plot M^2 for different lattice sizes
plt.figure(figsize=(12, 8))
colors = ['blue', 'green', 'red', 'orange']

for idx, n in enumerate(lattice_sizes):
    plt.errorbar(temperature_range, m2_avg_all[n], yerr=m2_std_all[n], fmt='o', label=f'Lattice Size {n}', color=colors[idx])

plt.xlabel('Temperature')
plt.ylabel('M^2')
plt.title('Magnetization Squared (M^2) vs Temperature')
plt.legend()
plt.grid(True)
plt.savefig('magnetization_squared_vs_temperature.png', dpi=300)
# plt.show()
