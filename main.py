import os
import torch
import tarfile
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import LinearLR
from sklearn.decomposition import PCA


def extract_zep(unzipped_path, data_path):
    os.makedirs(unzipped_path, exist_ok=True)
    # Extract the .tgz file
    try:
        with tarfile.open(data_path, "r:gz") as tar:
            tar.extractall(path=unzipped_path)
        print(f"Files have been extracted to: {unzipped_path}")
    except FileNotFoundError:
        print(f"The file {unzipped_path} does not exist.")
    except Exception as e:
        print(f"An error occurred: {e}")


class VAE(nn.Module):
    """
    Variational Autoencoder (VAE) class.
    """
    def __init__(self, latent_dim=128):
        """
        Initialize the VAE model.
        :param latent_dim: Dimension of the latent space.
        """
        super(VAE, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),  # 96 -> 48
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 48 -> 24

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 24 -> 12
            nn.ReLU(),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # 12 -> 6
            nn.ReLU(),

            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),  # 6 -> 3
            nn.ReLU(),

            nn.Flatten()
        )

        # Adjusted input size to match encoder output
        self.fc_mu = nn.Linear(512 * 3 * 3, latent_dim)
        self.fc_var = nn.Linear(512 * 3 * 3, latent_dim)

        # Decoder
        self.fc_decode = nn.Linear(latent_dim, 512 * 3 * 3)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),  # 3 -> 6
            nn.ReLU(),

            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 6 -> 12
            nn.ReLU(),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 12 -> 24
            nn.ReLU(),

            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # 24 -> 48
            nn.ReLU(),

            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),  # 48 -> 96
        )

    def encode(self, x):
        """
        Encode the input image.
        :param x: Input image.

        :return: expectation and log variance of the latent space distribution.
        """
        h = self.encoder(x)
        mu = self.fc_mu(h)
        var = self.fc_var(h)
        return mu, var

    def reparameterize(self, mu, var):
        """
        Reparameterization trick to sample from the latent space distribution.
        :param mu: Expectation of the latent space distribution.
        :param var: Log variance of the latent space distribution.
        :return: Sample from the latent space distribution.
        """
        eps = torch.randn_like(var)
        return mu + eps * torch.exp(var)

    def decode(self, z):
        """
        Decode a latent space sample.
        :param z: Latent space sample.
        :return: Reconstructed image.
        """
        h = self.fc_decode(z).view(-1, 512, 3, 3)  # Adjusted for new encoder output
        x = self.decoder(h)
        return x

    def forward(self, x):
        """
        Forward pass through the VAE.
        :param x: Input image.
        :return: Reconstructed image, expectation, and variance of the latent space distribution.
        """
        mu, var = self.encode(x)
        z = self.reparameterize(mu, var)
        x_reconstructed = self.decode(z)
        return x_reconstructed, mu, var


# Loss function
def vae_loss(reconstructed, original, mu, var):
    """
    Compute the basic VAE loss.
    :param reconstructed: Reconstructed image.
    :param original: Original image.
    :param mu: Expectation of the latent space distribution.
    :param var: Log variance of the latent space distribution.
    :return: Total loss, reconstruction loss, and KL divergence.
    """
    recon_loss = nn.functional.mse_loss(reconstructed, original, reduction='sum') / reconstructed.size()[0]
    kld_loss = -0.5 * torch.sum(1 + var - mu.pow(2) - torch.exp(var))
    return recon_loss + kld_loss, recon_loss, kld_loss


def train_model(vae, data_loader, dataset):
    """
    Train the VAE model.
    :param vae: VAE model to train.
    :param data_loader: DataLoader for the dataset.
    :param dataset: Dataset for training.
    :return: Trained VAE model and loss history.
    """
    # Training setup
    latent_dim = 128
    epochs = 30
    learning_rate = 0.001
    optimizer = optim.AdamW(vae.parameters(), lr=learning_rate)
    scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=0.01, total_iters=epochs)

    loss_history = []
    recon_loss_history = []
    kld_loss_history = []

    # Training loop
    vae.train()
    for epoch in range(epochs):
        train_loss = 0
        recon_loss, kld_loss = 0, 0
        for batch in data_loader:
            images, _ = batch
            images = images.to(device)

            optimizer.zero_grad()
            reconstructed, mu, var = vae(images)
            loss, recon, kld = vae_loss(reconstructed, images, mu, var)
            loss.backward()
            optimizer.step()
            kld_loss += kld.item()
            recon_loss += recon.item()
            train_loss += loss.item()

        scheduler.step()
        recon_loss_history.append(recon_loss / len(dataset))
        kld_loss_history.append(kld_loss / len(dataset))
        loss_history.append(train_loss / len(dataset))

        # Save trained weights
        torch.save(vae.state_dict(), '/home/student/model_weights.pkl')

        return vae, loss_history, recon_loss_history, kld_loss_history


def plot_train_results(loss_history, recon_loss_history, kld_loss_history, epochs=30):
    # Plot loss vs epochs
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs + 1), loss_history, marker='o', linestyle='-', label='Total Loss')
    plt.plot(range(1, epochs + 1), recon_loss_history, marker='x', linestyle='--', label='Reconstruction Loss')
    plt.plot(range(1, epochs + 1), kld_loss_history, marker='s', linestyle='-.', label='KL Divergence')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.title('VAE Training Loss vs Epochs')
    plt.legend()
    plt.grid()
    plt.show()


def plot_results(vae, data_loader, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], latent_dim=128):
    """
    Plot several evaluations of the trained VAE model:
    1. Random samples from the latent space.
    2. Original and reconstructed images.
    3. Linear interpolation between pairs of images.
    :param vae: Trained VAE model.
    :param data_loader: DataLoader for the dataset.
    :param mean: Mean to denormalize images.
    :param std: Standard deviation to denormalize images.
    :param latent_dim: Dimension of the latent space.
    """
    vae.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Convert mean and std to tensors
    mean = torch.tensor(mean).view(1, 3, 1, 1).to(device)
    std = torch.tensor(std).view(1, 3, 1, 1).to(device)
    # Plot sample images
    with torch.no_grad():
        sample = torch.randn(8, latent_dim).to(device)
        generated_images = vae.decode(sample)

        # Denormalize images
        generated_images = generated_images * std + mean

        fig, axes = plt.subplots(2, 4, figsize=(10, 10))
        for i, ax in enumerate(axes.flatten()):
            img = generated_images[i].cpu().permute(1, 2, 0)
            ax.imshow(img)
            ax.axis("off")
        plt.suptitle("Rnadomly Samaple from Laten Space")
        plt.tight_layout()
        plt.show()

    # Show results for 5 images
    vae.eval()
    with torch.no_grad():
        images, _ = next(iter(data_loader))
        images = images[:5].to(device)
        reconstructed, _, _ = vae(images)

        # Visualization
        fig, axes = plt.subplots(5, 2, figsize=(10, 15))
        den_images = images * std + mean
        reconstructed = reconstructed * std + mean
        for i in range(5):
            axes[i, 0].imshow(den_images[i].cpu().permute(1, 2, 0))
            axes[i, 0].set_title("Original")
            axes[i, 0].axis("off")

            axes[i, 1].imshow(reconstructed[i].cpu().permute(1, 2, 0))
            axes[i, 1].set_title("Reconstructed")
            axes[i, 1].axis("off")
        plt.tight_layout()
        plt.show()

    # Linear interpolation between 3 pairs
    with torch.no_grad():
        pairs = [(0, 1), (1, 2), (2, 3)]  # Indices of pairs
        fig, axes = plt.subplots(len(pairs), 8, figsize=(20, 8))

        for idx, (i, j) in enumerate(pairs):
            z1, _ = vae.encode(images[i].unsqueeze(0))
            z2, _ = vae.encode(images[j].unsqueeze(0))

            # Decode and normalize the original images
            img1 = images[i] * std.squeeze(0) + mean.squeeze(0)
            img2 = images[j] * std.squeeze(0) + mean.squeeze(0)

            # Plot the original images at the first and last positions
            axes[idx, 0].imshow(img1.cpu().permute(1, 2, 0))
            axes[idx, 0].axis("off")
            axes[idx, 0].set_title("Original Image 1")

            axes[idx, -1].imshow(img2.cpu().permute(1, 2, 0))
            axes[idx, -1].axis("off")
            axes[idx, -1].set_title("Original Image 2")

            for alpha_idx, alpha in enumerate(torch.linspace(0, 1, steps=6)):
                z_interp = (1 - alpha) * z1 + alpha * z2
                img_interp = vae.decode(z_interp)
                img_interp = img_interp * std + mean
                img_interp = img_interp.squeeze(0)
                axes[idx, alpha_idx + 1].imshow(img_interp.cpu().permute(1, 2, 0))
                axes[idx, alpha_idx + 1].axis("off")

        plt.tight_layout()
        plt.suptitle("Move through Latent Space")
        plt.show()


def encode_and_visualize_pca(vae, dataloader, num_images=10, num_samples_per_image=50, device="cuda"):
    """
    Encodes `num_images` images, samples multiple points from their distributions,
    applies PCA for dimensionality reduction, and plots the latent space.

    Args:
        vae (nn.Module): Trained VAE model.
        dataloader (DataLoader): DataLoader to get images.
        num_images (int): Number of images to encode.
        num_samples_per_image (int): Number of samples per latent distribution.
        device (str): Device for computation ("cuda" or "cpu").
    """

    vae.eval()  # Set model to evaluation mode
    images, _ = next(iter(dataloader))  # Get a batch of images
    images = images[:num_images].to(device)  # Select the first `num_images`

    # Encode images to get latent distributions
    with torch.no_grad():
        mu, var = vae.encode(images)

    # Sample from each latent distribution
    sampled_points = []
    labels = []  # Track which image each sample belongs to
    for i in range(num_images):
        mu_i = mu[i]  # Mean for image i
        var_i = var[i]  # Variance for image i
        std_i = torch.exp(0.5 * var_i)  # Convert log-variance to std deviation

        # Sample multiple points from the latent distribution
        samples = mu_i + std_i * torch.randn(num_samples_per_image, mu.shape[1]).to(device)
        sampled_points.append(samples.cpu())
        labels.extend([i] * num_samples_per_image)  # Assign a unique label per image

    # Convert to tensor for PCA
    sampled_points = torch.cat(sampled_points, dim=0).numpy()

    # Apply PCA to reduce dimensions to 2 for visualization
    pca = PCA(n_components=2)
    reduced_points = pca.fit_transform(sampled_points)

    # Plot the PCA-transformed latent space
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(reduced_points[:, 0], reduced_points[:, 1], c=labels, cmap="tab10", alpha=0.5, marker="o")
    plt.xlabel("PCA Dimension 1")
    plt.ylabel("PCA Dimension 2")
    plt.title("PCA Visualization of Sampled Latent Space")
    plt.grid()
    plt.show()


def save_model(vae, save_path):
    """
    Save the trained VAE model.
    :param vae: Trained VAE model.
    :param save_path: Path to save the model.
    """
    torch.save(vae.state_dict(), save_path)


def main():
    # Load the trained weights
    save_model_path = 'model_weights.pkl'
    unzipped_path = '102flowers'
    data_path = '102flowers.tgz'
    extract_zep(unzipped_path, data_path)
    # Transform and DataLoader
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    transform = transforms.Compose([
        transforms.Resize((96, 96)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    batch_size = 128
    latent_dim = 128
    dataset = datasets.ImageFolder(unzipped_path, transform=transform)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Train the model and evaluate:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vae = VAE(latent_dim=latent_dim).to(device)
    vae, loss_history, recon_loss_history, kld_loss_history = train_model(vae, data_loader, dataset)
    plot_train_results(loss_history, recon_loss_history, kld_loss_history)
    vae.eval()
    plot_results(vae, data_loader, mean=mean, std=std, latent_dim=latent_dim)
    encode_and_visualize_pca(vae, data_loader, num_images=10, num_samples_per_image=50, device=device)

    # Save model:
    save_model(vae, save_model_path)


if __name__ == '__main__':
    main()