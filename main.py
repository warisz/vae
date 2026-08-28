from datasets import load_from_disk
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from model import VAE

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