import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# Set random seed for reproducibility
torch.manual_seed(42)

# Hyperparameters
latent_dim = 100
hidden_dim = 256
image_dim = 784  # 28x28
batch_size = 64
num_epochs = 100
lr = 0.0002
beta1 = 0.5

# Generator Network
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, image_dim),
            nn.Tanh()
        )
    
    def forward(self, z):
        return self.model(z)

# Discriminator Network
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(image_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.model(x)

# Load MNIST Dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize networks and optimizers
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
generator = Generator().to(device)
discriminator = Discriminator().to(device)

g_optimizer = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))
d_optimizer = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999))

criterion = nn.BCELoss()

# Lists to store losses for plotting
g_losses = []
d_losses = []

# Training loop
for epoch in range(num_epochs):
    for i, (real_images, _) in enumerate(dataloader):
        batch_size = real_images.size(0)
        real_images = real_images.reshape(-1, image_dim).to(device)
        
        # Labels
        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)
        
        # Train Discriminator
        d_optimizer.zero_grad()
        # Real images
        d_output_real = discriminator(real_images)
        d_loss_real = criterion(d_output_real, real_labels)
        
        # Fake images
        z = torch.randn(batch_size, latent_dim).to(device)
        fake_images = generator(z)
        d_output_fake = discriminator(fake_images.detach())
        d_loss_fake = criterion(d_output_fake, fake_labels)
        
        # Combined discriminator loss
        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        d_optimizer.step()
        
        # Train Generator
        g_optimizer.zero_grad()
        d_output_fake = discriminator(fake_images)
        g_loss = criterion(d_output_fake, real_labels)
        g_loss.backward()
        g_optimizer.step()
        
        if i == 0:
            print(f'Epoch [{epoch}/{num_epochs}] d_loss: {d_loss.item():.4f} g_loss: {g_loss.item():.4f}')
            g_losses.append(g_loss.item())
            d_losses.append(d_loss.item())

# Plot the loss curves
plt.figure(figsize=(10, 5))
plt.plot(g_losses, label='Generator Loss')
plt.plot(d_losses, label='Discriminator Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('GAN Training Loss Curves')
plt.grid(True)
plt.show()

# Generate and display sample images
with torch.no_grad():
    z = torch.randn(16, latent_dim).to(device)
    generated_images = generator(z).cpu()
    generated_images = generated_images.reshape(-1, 28, 28)

plt.figure(figsize=(10, 10))
for i in range(16):
    plt.subplot(4, 4, i+1)
    plt.imshow(generated_images[i], cmap='gray')
    plt.axis('off')
plt.show()