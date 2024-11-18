# # import torch; torch.manual_seed(0)
# # import torch.nn as nn
# # import torch.nn.functional as F
# # import torch.utils
# # import torch.distributions
# # import torchvision
# # import numpy as np
# # import matplotlib.pyplot as plt; plt.rcParams['figure.dpi'] = 200

# # device = 'mps' #'cpu'
# # class VariationalEncoder(nn.Module):
# #     def __init__(self, latent_dims):
# #         super(VariationalEncoder, self).__init__()
# #         self.linear1 = nn.Linear(784, 512)
# #         self.linear2 = nn.Linear(512, latent_dims)
# #         self.linear3 = nn.Linear(512, latent_dims)

# #         self.N = torch.distributions.Normal(0, 1)
# #         self.N.loc = self.N.loc.cuda() # hack to get sampling on the GPU
# #         self.N.scale = self.N.scale.cuda()
# #         self.kl = 0

# #     def forward(self, x):
# #         x = torch.flatten(x, start_dim=1)
# #         x = F.relu(self.linear1(x))
# #         mu =  self.linear2(x)
# #         sigma = torch.exp(self.linear3(x))
# #         z = mu + sigma*self.N.sample(mu.shape)
# #         self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
# #         return z
# # class VariationalAutoencoder(nn.Module):
# #     def __init__(self, latent_dims):
# #         super(VariationalAutoencoder, self).__init__()
# #         self.encoder = VariationalEncoder(latent_dims)
# #         self.decoder = Decoder(latent_dims)

# #     def forward(self, x):
# #         z = self.encoder(x)
# #         return self.decoder(z)

# # def train(autoencoder, data, epochs=20):
# #     opt = torch.optim.Adam(autoencoder.parameters())
# #     for epoch in range(epochs):
# #         for x, y in data:
# #             x = x.to(device) # GPU
# #             opt.zero_grad()
# #             x_hat = autoencoder(x)
# #             loss = ((x - x_hat)**2).sum() + autoencoder.encoder.kl
# #             loss.backward()
# #             opt.step()
# #     return autoencoder

# # latent_dims = 2
# # vae = VariationalAutoencoder(latent_dims).to(device) # GPU
# # vae = train(vae, data)

# import torch; torch.manual_seed(0)
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.utils
# import torch.distributions
# import torchvision
# import numpy as np
# import matplotlib.pyplot as plt; plt.rcParams['figure.dpi'] = 200

# class VariationalEncoder(nn.Module):
#     def __init__(self, latent_dims):
#         super(VariationalEncoder, self).__init__()
#         self.linear1 = nn.Linear(784, 512)
#         self.linear2 = nn.Linear(512, latent_dims)
#         self.linear3 = nn.Linear(512, latent_dims)
#         self.N = torch.distributions.Normal(0, 1)
#         self.kl = 0

#     def forward(self, x):
#         x = torch.flatten(x, start_dim=1)
#         x = F.relu(self.linear1(x))
#         mu = self.linear2(x)
#         sigma = torch.exp(self.linear3(x))
#         z = mu + sigma * self.N.sample(mu.shape)
#         self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
#         return z

# class Decoder(nn.Module):
#     def __init__(self, latent_dims):
#         super(Decoder, self).__init__()
#         self.linear1 = nn.Linear(latent_dims, 512)
#         self.linear2 = nn.Linear(512, 784)

#     def forward(self, z):
#         z = F.relu(self.linear1(z))
#         z = torch.sigmoid(self.linear2(z))
#         return z.reshape((-1, 1, 28, 28))

# class VariationalAutoencoder(nn.Module):
#     def __init__(self, latent_dims):
#         super(VariationalAutoencoder, self).__init__()
#         self.encoder = VariationalEncoder(latent_dims)
#         self.decoder = Decoder(latent_dims)

#     def forward(self, x):
#         z = self.encoder(x)
#         return self.decoder(z)

# def train(autoencoder, data, epochs=20):
#     opt = torch.optim.Adam(autoencoder.parameters())
#     for epoch in range(epochs):
#         for x, y in data:
#             opt.zero_grad()
#             x_hat = autoencoder(x)
#             loss = ((x - x_hat)**2).sum() + autoencoder.encoder.kl
#             loss.backward()
#             opt.step()
#     return autoencoder

# # Example usage:
# latent_dims = 2
# vae = VariationalAutoencoder(latent_dims)

# # Example data loading
# transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
# data = torch.utils.data.DataLoader(
#     torchvision.datasets.MNIST('./data', transform=transform, download=True),
#     batch_size=128,
#     shuffle=True)

# vae = train(vae, data)

import torch; torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.distributions
import torchvision
import numpy as np
import matplotlib.pyplot as plt; plt.rcParams['figure.dpi'] = 200

# Custom dataset wrapper to ensure writeable arrays
class WriteableDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        
    def __getitem__(self, index):
        data, target = self.dataset[index]
        # Make a copy to ensure it's writeable
        if isinstance(data, np.ndarray):
            data = np.array(data, copy=True)
        elif isinstance(data, torch.Tensor):
            data = data.clone()
        return data, target
    
    def __len__(self):
        return len(self.dataset)

class VariationalEncoder(nn.Module):
    def __init__(self, latent_dims):
        super(VariationalEncoder, self).__init__()
        self.linear1 = nn.Linear(784, 512)
        self.linear2 = nn.Linear(512, latent_dims)
        self.linear3 = nn.Linear(512, latent_dims)
        self.N = torch.distributions.Normal(0, 1)
        self.kl = 0

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.linear1(x))
        mu = self.linear2(x)
        sigma = torch.exp(self.linear3(x))
        z = mu + sigma * self.N.sample(mu.shape)
        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
        return z

class Decoder(nn.Module):
    def __init__(self, latent_dims):
        super(Decoder, self).__init__()
        self.linear1 = nn.Linear(latent_dims, 512)
        self.linear2 = nn.Linear(512, 784)

    def forward(self, z):
        z = F.relu(self.linear1(z))
        z = torch.sigmoid(self.linear2(z))
        return z.reshape((-1, 1, 28, 28))

class VariationalAutoencoder(nn.Module):
    def __init__(self, latent_dims):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = VariationalEncoder(latent_dims)
        self.decoder = Decoder(latent_dims)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

def train(autoencoder, data, epochs=100):
    opt = torch.optim.Adam(autoencoder.parameters())
    for epoch in range(epochs):
        for x, y in data:
            opt.zero_grad()
            x_hat = autoencoder(x)
            loss = ((x - x_hat)**2).sum() + autoencoder.encoder.kl
            loss.backward()
            opt.step()
        if (epoch + 1) % 5 == 0:
            print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}')
    return autoencoder

# Example usage with the fixed data loading:
transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
])

# Load MNIST dataset and wrap it with WriteableDataset
mnist_dataset = torchvision.datasets.MNIST('./data', transform=transform, download=True)
writeable_dataset = WriteableDataset(mnist_dataset)

# Create data loader with the wrapped dataset
data = torch.utils.data.DataLoader(
    writeable_dataset,
    batch_size=128,
    shuffle=True,
    num_workers=0  # Set to 0 to avoid potential multiprocessing issues
)

# Initialize and train the model
latent_dims = 2
vae = VariationalAutoencoder(latent_dims)
vae = train(vae, data)