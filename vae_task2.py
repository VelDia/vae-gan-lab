import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc
import seaborn as sns

class VAE(nn.Module):
    def __init__(self, latent_dim=20):
        super(VAE, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(784, 400),
            nn.ReLU(),
            nn.Linear(400, 200),
            nn.ReLU()
        )
        
        # Latent space
        self.fc_mu = nn.Linear(200, latent_dim)
        self.fc_var = nn.Linear(200, latent_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 200),
            nn.ReLU(),
            nn.Linear(200, 400),
            nn.ReLU(),
            nn.Linear(400, 784),
            nn.Sigmoid()
        )
        
    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_var(h)
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        mu, log_var = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var

def loss_function(recon_x, x, mu, log_var):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return BCE + KLD

def train_vae(model, train_loader, optimizer, device, epochs=50):
    model.train()
    train_losses = []
    
    for epoch in range(epochs):
        epoch_loss = 0
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            
            recon_batch, mu, log_var = model(data)
            loss = loss_function(recon_batch, data, mu, log_var)
            
            loss.backward()
            epoch_loss += loss.item()
            optimizer.step()
            
        avg_loss = epoch_loss / len(train_loader.dataset)
        train_losses.append(avg_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}/{epochs} Average loss: {avg_loss:.4f}')
    
    return train_losses

def add_noise(images, noise_factor=0.5):
    noisy_images = images + noise_factor * torch.randn(*images.shape)
    return torch.clamp(noisy_images, 0., 1.)

def compute_reconstruction_error(model, data_loader, device):
    model.eval()
    reconstruction_errors = []
    
    with torch.no_grad():
        for data, _ in data_loader:
            data = data.to(device)
            recon_batch, _, _ = model(data)
            error = F.mse_loss(recon_batch, data.view(-1, 784), reduction='none').sum(dim=1)
            reconstruction_errors.extend(error.cpu().numpy())
    
    return np.array(reconstruction_errors)

def main():
    # Set random seed and device
    torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    
    train_dataset = torchvision.datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST('./data', train=False, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    # Initialize model
    model = VAE().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # Train the model
    print("Training VAE...")
    train_losses = train_vae(model, train_loader, optimizer, device)
    
    # Plot training losses
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses)
    plt.title('VAE Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig('vae_training_loss.png')
    plt.close()
    
    # Generate anomalous data
    print("Generating anomalous data...")
    test_data = test_dataset.data.float() / 255.0
    noisy_test_data = add_noise(test_data)
    
    normal_dataset = torch.utils.data.TensorDataset(test_data, torch.zeros(len(test_data)))
    anomaly_dataset = torch.utils.data.TensorDataset(noisy_test_data, torch.ones(len(noisy_test_data)))
    
    normal_loader = DataLoader(normal_dataset, batch_size=128, shuffle=False)
    anomaly_loader = DataLoader(anomaly_dataset, batch_size=128, shuffle=False)
    
    # Compute reconstruction errors
    print("Computing reconstruction errors...")
    normal_errors = compute_reconstruction_error(model, normal_loader, device)
    anomaly_errors = compute_reconstruction_error(model, anomaly_loader, device)
    
    # Plot distribution of reconstruction errors
    plt.figure(figsize=(12, 6))
    plt.hist(normal_errors, bins=50, alpha=0.5, label='Normal', density=True)
    plt.hist(anomaly_errors, bins=50, alpha=0.5, label='Anomalous', density=True)
    plt.xlabel('Reconstruction Error')
    plt.ylabel('Density')
    plt.title('Distribution of Reconstruction Errors')
    plt.legend()
    plt.savefig('reconstruction_error_distribution.png')
    plt.close()
    
    # Calculate ROC curve and optimal threshold
    labels = np.concatenate([np.zeros(len(normal_errors)), np.ones(len(anomaly_errors))])
    scores = np.concatenate([normal_errors, anomaly_errors])
    
    fpr, tpr, thresholds = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)
    
    # Find optimal threshold using Youden's J statistic
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    
    # Plot ROC curve
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig('roc_curve.png')
    plt.close()
    
    # Evaluate results using optimal threshold
    normal_predictions = (normal_errors > optimal_threshold).astype(int)
    anomaly_predictions = (anomaly_errors > optimal_threshold).astype(int)
    
    normal_accuracy = (normal_predictions == 0).mean()
    anomaly_accuracy = (anomaly_predictions == 1).mean()
    
    print("\nResults:")
    print(f"Optimal threshold: {optimal_threshold:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    print(f"Normal data classification accuracy: {normal_accuracy:.4f}")
    print(f"Anomaly data classification accuracy: {anomaly_accuracy:.4f}")
    
    # Visualize some examples
    model.eval()
    with torch.no_grad():
        # Get some normal and anomalous images
        normal_images = test_data[:8]
        anomalous_images = noisy_test_data[:8]
        
        # Get reconstructions
        normal_recon, _, _ = model(normal_images.view(-1, 784).to(device))
        anomalous_recon, _, _ = model(anomalous_images.view(-1, 784).to(device))
        
        # Convert to numpy for plotting
        normal_images = normal_images.cpu().numpy()
        normal_recon = normal_recon.view(-1, 28, 28).cpu().numpy()
        anomalous_images = anomalous_images.cpu().numpy()
        anomalous_recon = anomalous_recon.view(-1, 28, 28).cpu().numpy()
        
        # Plot
        fig, axes = plt.subplots(4, 8, figsize=(20, 10))
        for i in range(8):
            # Plot normal images
            axes[0, i].imshow(normal_images[i], cmap='gray')
            axes[0, i].axis('off')
            axes[1, i].imshow(normal_recon[i], cmap='gray')
            axes[1, i].axis('off')
            
            # Plot anomalous images
            axes[2, i].imshow(anomalous_images[i], cmap='gray')
            axes[2, i].axis('off')
            axes[3, i].imshow(anomalous_recon[i], cmap='gray')
            axes[3, i].axis('off')
        
        axes[0, 0].set_ylabel('Normal\nOriginal')
        axes[1, 0].set_ylabel('Normal\nReconstructed')
        axes[2, 0].set_ylabel('Anomalous\nOriginal')
        axes[3, 0].set_ylabel('Anomalous\nReconstructed')
        
        plt.suptitle('Original vs Reconstructed Images')
        plt.tight_layout()
        plt.savefig('reconstruction_examples.png')
        plt.close()

if __name__ == '__main__':
    main()