import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from datasets import load_from_disk
from torch.utils.data import DataLoader
from torchvision import transforms
from model import VAE

device = "cuda" if torch.cuda.is_available() else "cpu"
model = VAE(latent_dim=128).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

dataset = load_from_disk("data/images")
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])
def apply_transform(batch):
    return {"pixel_values": [transform(img.convert("RGB")) for img in batch["image"]]}

dataset.set_transform(apply_transform)
loader = DataLoader(dataset["train"], batch_size=32, shuffle=True)

history_loss, history_recon, history_kl = [], [], []
epochs = 20

print("Starting training...")
for epoch in range(epochs):
    model.train()
    
    epoch_loss_sum = 0
    epoch_recon_sum = 0
    epoch_kl_sum = 0
    
    for batch in loader:
        x = batch["pixel_values"].to(device)
        
        optimizer.zero_grad()
        x_hat, mu, logvar = model(x)
        
        recon_loss = torch.nn.functional.mse_loss(x_hat, x, reduction='sum') / x.size(0)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
        loss = recon_loss + kl_loss
        
        loss.backward()
        optimizer.step()
        
        epoch_loss_sum += loss.item()
        epoch_recon_sum += recon_loss.item()
        epoch_kl_sum += kl_loss.item()
        
    num_batches = len(loader)
    avg_loss = epoch_loss_sum / num_batches
    avg_recon = epoch_recon_sum / num_batches
    avg_kl = epoch_kl_sum / num_batches
    
    history_loss.append(avg_loss)
    history_recon.append(avg_recon)
    history_kl.append(avg_kl)
    
    print(f"epoch {epoch:2d}: loss= {avg_loss:6.1f} recon= {avg_recon:6.1f} kl= {avg_kl:6.1f}")

torch.save(model.state_dict(), "vae_weights.pth")
print("Saved vae_weights.pth!")

plt.figure(figsize=(10, 6))
plt.plot(history_loss, label='Total Loss', color='blue', linewidth=2)
plt.plot(history_recon, label='Reconstruction Loss', color='green', linewidth=2)
plt.plot(history_kl, label='KL Divergence', color='red', linewidth=2)

plt.title('VAE Training Loss Over Time (True Averages)', fontsize=14, fontweight='bold')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss Value', fontsize=12)
plt.xticks(range(0, epochs, 2))
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=11)

plt.tight_layout()
plt.savefig('loss_graph.png')
print("Saved loss_graph.png!")