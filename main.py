from datasets import load_from_disk
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt

# ------ DATA -------
dataset = load_from_disk("data/images")
transform = transforms.Compose([
    transforms.Resize((64,64)),
    transforms.ToTensor(),
])

def apply_transform(batch):
    return{"pixel_values": [transform(img.convert("RGB")) for img in batch["image"]]}

dataset.set_transform(apply_transform)
loader = DataLoader(dataset["train"], batch_size=32, shuffle=True) #dataloader helps with shuffling amongst other things

class Encoder(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)

        self.flatten_size = 256 * 4 * 4
        self.fc_mu = nn.Linear(self.flatten_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_size, latent_dim)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.conv2(x), 0.2)
        x = F.leaky_relu(self.conv3(x), 0.2)
        x = F.leaky_relu(self.conv4(x), 0.2)
        x = x.view(-1, self.flatten_size)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar 


class Decoder(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.flatten_size = 256 * 4 * 4
        self.fc = nn.Linear(latent_dim, self.flatten_size)
        self.deconv1 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.deconv4 = nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1)

    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, 256, 4, 4)
        x = F.leaky_relu(self.deconv1(x), 0.2)
        x = F.leaky_relu(self.deconv2(x), 0.2)
        x = F.leaky_relu(self.deconv3(x), 0.2)
        x = torch.sigmoid(self.deconv4(x))
        return x

class VAE(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decoder(z)
        return x_hat, mu, logvar

def vae_loss(x, x_hat, mu, logvar):
    recon = F.mse_loss(x_hat, x, reduction='sum') / x.shape[0]
    kl = -0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp()) / x.shape[0]
    return recon + kl, recon, kl


    
# -------- training ---------
device = "cuda" if torch.cuda.is_available() else "cpu"
model = VAE(latent_dim=128).to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

#sanity check first batch
first = next(iter(loader))["pixel_values"]
assert torch.is_tensor(first) and first.shape[1:] == (3, 64, 64), f"expected stacked tensor [B,3,64,64], got {type(first)} {getattr(first, 'shape', None)}"\

for epoch in range(20):
    model.train()
    total = 0.0
    for batch in loader:
        x = batch["pixel_values"].to(device)
        opt.zero_grad()
        x_hat, mu, logvar = model(x)
        loss, recon, kl = vae_loss(x, x_hat, mu, logvar)
        loss.backward()
        opt.step()
        total += loss.item()
    print(f"epoch {epoch:2d}: loss={total/len(loader):7.1f} recon={recon.item():7.1f} kl={kl.item():6.1f}")

torch.save(model.state_dict(), "vae_weights.pth")