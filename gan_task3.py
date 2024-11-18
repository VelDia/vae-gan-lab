import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import torch.nn.functional as F

# Reuse Generator architecture from previous implementation
class Generator(nn.Module):
    def __init__(self, latent_dim=100, hidden_dim=256, image_dim=784):
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

def interpolate_latent_space(generator, z1, z2, num_steps=10):
    """
    Perform linear interpolation between two latent vectors and generate corresponding images.
    
    Args:
        generator: Trained generator model
        z1, z2: Start and end latent vectors
        num_steps: Number of interpolation steps
    
    Returns:
        List of generated images
    """
    generator.eval()
    interpolated_images = []
    
    # Create interpolation steps
    alphas = np.linspace(0, 1, num_steps)
    
    with torch.no_grad():
        for alpha in alphas:
            # Linear interpolation: z = (1-α)z1 + αz2
            z = (1 - alpha) * z1 + alpha * z2
            # Generate image from interpolated latent vector
            fake_image = generator(z)
            interpolated_images.append(fake_image.cpu())
    
    return interpolated_images

def visualize_interpolation(images, num_steps, save_path='interpolation.png'):
    """
    Create a visualization of the interpolated images.
    """
    # Reshape images to 28x28
    images = [img.reshape(28, 28) for img in images]
    
    # Create figure
    fig, axes = plt.subplots(1, num_steps, figsize=(20, 4))
    plt.subplots_adjust(wspace=0.05)
    
    # Plot each image
    for i, ax in enumerate(axes):
        ax.imshow(images[i], cmap='gray')
        ax.axis('off')
        if i == 0:
            ax.set_title('Start', pad=10)
        elif i == num_steps-1:
            ax.set_title('End', pad=10)
        
    plt.suptitle('Latent Space Interpolation', y=1.05, fontsize=16)
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()

def analyze_interpolation_smoothness(images):
    """
    Analyze the smoothness of transitions between interpolated images.
    Returns metrics about the interpolation quality.
    """
    smoothness_metrics = {}
    
    # Convert images to numpy arrays
    images_np = [img.numpy() for img in images]
    
    # Calculate average pixel-wise difference between consecutive images
    consecutive_diffs = []
    for i in range(len(images_np)-1):
        diff = np.mean(np.abs(images_np[i+1] - images_np[i]))
        consecutive_diffs.append(diff)
    
    smoothness_metrics['avg_consecutive_diff'] = np.mean(consecutive_diffs)
    smoothness_metrics['max_consecutive_diff'] = np.max(consecutive_diffs)
    smoothness_metrics['std_consecutive_diff'] = np.std(consecutive_diffs)
    
    return smoothness_metrics

# Main execution
def main():
    # Initialize device and model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    latent_dim = 100
    generator = Generator(latent_dim).to(device)
    
    # Load the trained generator weights (assuming they're saved)
    try:
        generator.load_state_dict(torch.load('generator.pth'))
    except:
        print("No saved model found. Please train the GAN first.")
        return
    
    # Generate two random latent vectors
    z1 = torch.randn(1, latent_dim).to(device)
    z2 = torch.randn(1, latent_dim).to(device)
    
    # Perform interpolation
    num_steps = 10
    interpolated_images = interpolate_latent_space(generator, z1, z2, num_steps)
    
    # Visualize interpolation
    visualize_interpolation(interpolated_images, num_steps)
    
    # Analyze interpolation smoothness
    smoothness_metrics = analyze_interpolation_smoothness(interpolated_images)
    
    print("\nInterpolation Analysis:")
    print(f"Average consecutive image difference: {smoothness_metrics['avg_consecutive_diff']:.4f}")
    print(f"Maximum consecutive image difference: {smoothness_metrics['max_consecutive_diff']:.4f}")
    print(f"Standard deviation of differences: {smoothness_metrics['std_consecutive_diff']:.4f}")
    
    # Generate multiple interpolation examples
    num_examples = 5
    plt.figure(figsize=(20, 4*num_examples))
    
    for i in range(num_examples):
        z1 = torch.randn(1, latent_dim).to(device)
        z2 = torch.randn(1, latent_dim).to(device)
        interpolated_images = interpolate_latent_space(generator, z1, z2, num_steps)
        
        plt.subplot(num_examples, 1, i+1)
        grid = make_grid(torch.stack([img.reshape(1, 28, 28) for img in interpolated_images]), 
                        nrow=num_steps, padding=2, normalize=True)
        plt.imshow(grid.permute(1, 2, 0), cmap='gray')
        plt.axis('off')
        
    plt.suptitle('Multiple Interpolation Examples', y=1.02, fontsize=16)
    plt.tight_layout()
    plt.savefig('multiple_interpolations.png', bbox_inches='tight', dpi=150)
    plt.close()

if __name__ == "__main__":
    main()