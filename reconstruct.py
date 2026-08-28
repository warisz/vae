import torch
from torchvision.utils import save_image
from datasets import load_from_disk
from torch.utils.data import DataLoader
from torchvision import transforms

device = "cuda" if torch.cuda.is_available() else "cpu"
model = VAE(latent_dim=128).to(device)

model.load_state_dict(torch.load("vae_weights.pth", map_location=device))
model.eval()

dataset = load_from_disk("data/images") 
transform = transforms.Compose([
    transforms.Resize((64,64)),
    transforms.ToTensor(),
])
def apply_transform(batch):
    return{"pixel_values": [transform(img.convert("RGB")) for img in batch["image"]]}
dataset.set_transform(apply_transform)

# Changed batch_size to 50
test_loader = DataLoader(dataset["train"], batch_size=50, shuffle=True)

with torch.no_grad():
    x = next(iter(test_loader))["pixel_values"].to(device)
    x_hat, _, _ = model(x)
    
    # Stack groups them into pairs: [50, 2, 3, 64, 64]
    # View flattens the pairs into a single list of 100 images: [100, 3, 64, 64]
    comparison = torch.stack([x, x_hat], dim=1).view(-1, 3, 64, 64)
    
    # nrow=10 means 10 images per row (exactly 5 pairs per row)
    save_image(comparison, "50_side_by_side.png", nrow=10)
    
