import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader

# Set random seed for reproducibility
torch.manual_seed(42)

# Hyperparameters
latent_dim = 50
image_size = 32
channels = 3
num_epochs = 10
batch_size = 128
lr = 0.0002
beta1 = 0.5

# Configure device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Generator Network
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        
        self.main = nn.Sequential(
            # Input is latent_dim x 1 x 1
            nn.ConvTranspose2d(latent_dim, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # State size: 512 x 4 x 4
            
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # State size: 256 x 8 x 8
            
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # State size: 128 x 16 x 16
            
            nn.ConvTranspose2d(128, channels, 4, 2, 1, bias=False),
            nn.Tanh()
            # Output size: channels x 32 x 32
        )

    def forward(self, x):
        return self.main(x)

# Discriminator Network
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        self.main = nn.Sequential(
            # Input is channels x 32 x 32
            nn.Conv2d(channels, 128, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # State size: 128 x 16 x 16
            
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # State size: 256 x 8 x 8
            
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            # State size: 512 x 4 x 4
            
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
            # Output size: 1 x 1 x 1
        )

    def forward(self, x):
        return self.main(x).view(-1, 1).squeeze(1)

# Load and preprocess CIFAR-10 dataset
def load_data():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                          download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size,
                          shuffle=True, num_workers=2)
    
    return dataloader

# Initialize weights
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# Training function
def train_dcgan():
    # Create the models
    netG = Generator().to(device)
    netD = Discriminator().to(device)
    
    # Initialize weights
    netG.apply(weights_init)
    netD.apply(weights_init)
    
    # Setup optimizers
    criterion = nn.BCELoss()
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))
    
    # Load data
    dataloader = load_data()
    
    # Training loop
    G_losses = []
    D_losses = []
    
    print("Starting Training Loop...")
    for epoch in range(num_epochs):
        for i, data in enumerate(dataloader, 0):
            ############################
            # Update Discriminator
            ###########################
            netD.zero_grad()
            real = data[0].to(device)
            batch_size = real.size(0)
            label_real = torch.ones(batch_size).to(device)
            label_fake = torch.zeros(batch_size).to(device)

            # Train with real
            output = netD(real)
            errD_real = criterion(output, label_real)
            D_x = output.mean().item()

            # Train with fake
            noise = torch.randn(batch_size, latent_dim, 1, 1, device=device)
            fake = netG(noise)
            output = netD(fake.detach())
            errD_fake = criterion(output, label_fake)
            D_G_z1 = output.mean().item()
            
            errD = errD_real + errD_fake
            errD.backward()
            optimizerD.step()

            ############################
            # Update Generator
            ###########################
            netG.zero_grad()
            output = netD(fake)
            errG = criterion(output, label_real)
            errG.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()

            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())

            if i % 100 == 0:
                print(f'[{epoch}/{num_epochs}][{i}/{len(dataloader)}] '
                      f'Loss_D: {errD.item():.4f} Loss_G: {errG.item():.4f} '
                      f'D(x): {D_x:.4f} D(G(z)): {D_G_z1:.4f}/{D_G_z2:.4f}')

    return netG, netD, G_losses, D_losses

# Visualization functions
def visualize_results(netG, real_batch):
    # Generate fake images
    with torch.no_grad():
        fake = netG(torch.randn(64, latent_dim, 1, 1, device=device)).cpu()
    
    # Plot real images
    plt.figure(figsize=(15, 15))
    plt.subplot(1, 2, 1)
    plt.axis("off")
    plt.title("Real Images")
    plt.imshow(np.transpose(torchvision.utils.make_grid(
        real_batch[:64], padding=2, normalize=True).cpu(), (1, 2, 0)))
    
    # Plot fake images
    plt.subplot(1, 2, 2)
    plt.axis("off")
    plt.title("Fake Images")
    plt.imshow(np.transpose(torchvision.utils.make_grid(
        fake, padding=2, normalize=True), (1, 2, 0)))
    plt.show()

def plot_training_losses(G_losses, D_losses):
    plt.figure(figsize=(10,5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses, label="Generator")
    plt.plot(D_losses, label="Discriminator")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import torch.nn.functional as F

def generate_latent_points(latent_dim, n_samples):
    """Generate random points in the latent space."""
    return torch.randn(n_samples, latent_dim, 1, 1)

def interpolate_points(p1, p2, n_steps=10):
    """
    Perform linear interpolation between two points in latent space.
    Args:
        p1: Starting point
        p2: Ending point
        n_steps: Number of interpolation steps
    Returns:
        Tensor of interpolated points
    """
    # Create interpolation coefficients
    ratios = torch.linspace(0, 1, n_steps)
    
    # Generate interpolated points
    vectors = []
    for ratio in ratios:
        # Linear interpolation: v = (1-r)*v1 + r*v2
        interpolated = p1 * (1 - ratio) + p2 * ratio
        vectors.append(interpolated)
    
    return torch.cat(vectors, dim=0)

def visualize_interpolation(generator, latent_dim, device, n_steps=10):
    """
    Generate and visualize interpolation between two random points in latent space.
    Args:
        generator: Trained generator model
        latent_dim: Dimension of latent space
        device: torch device (cuda/cpu)
        n_steps: Number of interpolation steps
    """
    # Set generator to eval mode
    generator.eval()
    
    # Generate two random points in latent space
    z1 = generate_latent_points(latent_dim, 1).to(device)
    z2 = generate_latent_points(latent_dim, 1).to(device)
    
    # Interpolate between the points
    interpolated = interpolate_points(z1, z2, n_steps).to(device)
    
    # Generate images from interpolated points
    with torch.no_grad():
        generated_images = generator(interpolated)
    
    # Convert images for visualization
    images = (generated_images.cpu() + 1) / 2  # Denormalize from [-1,1] to [0,1]
    
    # Create figure
    fig = plt.figure(figsize=(20, 4))
    for i in range(n_steps):
        plt.subplot(1, n_steps, i + 1)
        plt.axis('off')
        plt.imshow(np.transpose(images[i], (1, 2, 0)))
        if i == 0:
            plt.title('Start')
        elif i == n_steps - 1:
            plt.title('End')
        else:
            plt.title(f'Step {i}')
    
    plt.suptitle('Latent Space Interpolation', fontsize=16)
    plt.tight_layout()
    plt.show()

def analyze_interpolation_smoothness(generator, latent_dim, device, n_steps=10, n_pairs=5):
    """
    Analyze the smoothness of interpolation by measuring the average 
    difference between consecutive images.
    Args:
        generator: Trained generator model
        latent_dim: Dimension of latent space
        device: torch device (cuda/cpu)
        n_steps: Number of interpolation steps
        n_pairs: Number of random pairs to analyze
    """
    generator.eval()
    smoothness_scores = []
    
    for pair in range(n_pairs):
        # Generate pair of latent vectors
        z1 = generate_latent_points(latent_dim, 1).to(device)
        z2 = generate_latent_points(latent_dim, 1).to(device)
        
        # Interpolate
        interpolated = interpolate_points(z1, z2, n_steps).to(device)
        
        # Generate images
        with torch.no_grad():
            images = generator(interpolated)
        
        # Calculate differences between consecutive images
        diffs = []
        for i in range(len(images)-1):
            # Calculate MSE between consecutive images
            diff = F.mse_loss(images[i], images[i+1]).item()
            diffs.append(diff)
        
        # Average difference for this interpolation
        avg_diff = np.mean(diffs)
        smoothness_scores.append(avg_diff)
    
    return np.mean(smoothness_scores), np.std(smoothness_scores)

def interpolate_between_classes(generator, latent_dim, device, n_steps=10):
    """
    Generate interpolation between two meaningful directions in latent space
    that might correspond to different classes/features.
    """
    generator.eval()
    
    # Generate multiple random vectors and pick two that generate distinct features
    n_candidates = 20
    candidates = generate_latent_points(latent_dim, n_candidates).to(device)
    
    with torch.no_grad():
        candidate_images = generator(candidates)
    
    # Visualize candidate images to manually select interesting pairs
    plt.figure(figsize=(20, 4))
    for i in range(n_candidates):
        plt.subplot(2, n_candidates//2, i+1)
        plt.axis('off')
        plt.imshow(np.transpose((candidate_images[i].cpu() + 1) / 2, (1, 2, 0)))
    plt.show()
    
    # Let's pick two vectors that generate visually distinct images
    # (You can modify these indices based on the generated candidates)
    z1 = candidates[0:1]
    z2 = candidates[1:2]
    
    # Perform interpolation
    interpolated = interpolate_points(z1, z2, n_steps).to(device)
    
    # Generate and visualize interpolated images
    with torch.no_grad():
        generated_images = generator(interpolated)
    
    # Visualize interpolation
    plt.figure(figsize=(20, 4))
    for i in range(n_steps):
        plt.subplot(1, n_steps, i+1)
        plt.axis('off')
        plt.imshow(np.transpose((generated_images[i].cpu() + 1) / 2, (1, 2, 0)))
    plt.suptitle('Interpolation Between Distinct Features', fontsize=16)
    plt.tight_layout()
    plt.show()

# Example usage (assuming we have a trained generator)
def explore_latent_space(generator, latent_dim, device):
    """
    Comprehensive exploration of the latent space.
    """
    print("1. Basic Interpolation between random points:")
    visualize_interpolation(generator, latent_dim, device)
    
    print("\n2. Analyzing interpolation smoothness:")
    mean_smoothness, std_smoothness = analyze_interpolation_smoothness(
        generator, latent_dim, device)
    print(f"Average smoothness score: {mean_smoothness:.6f} ± {std_smoothness:.6f}")
    
    print("\n3. Interpolating between distinct features:")
    interpolate_between_classes(generator, latent_dim, device)

# Main execution
if __name__ == "__main__":
    # Train the model
    netG, netD, G_losses, D_losses = train_dcgan()
    
    # Get a batch of real images for comparison
    dataloader = load_data()
    real_batch = next(iter(dataloader))[0]
    
    # Visualize results
    visualize_results(netG, real_batch)
    plot_training_losses(G_losses, D_losses)

    # Load the trained generator (assuming we have it saved)
    generator = Generator().to(device)
    # generator.load_state_dict(torch.load('generator.pth'))  # Uncomment to load saved model
    
    # Explore latent space
    explore_latent_space(generator, latent_dim, device)