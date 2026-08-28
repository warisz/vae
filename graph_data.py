import matplotlib.pyplot as plt
import pandas as pd
import re

data = """
epoch  0: loss=  407.7 recon=  293.8 kl=  51.5
epoch  1: loss=  301.1 recon=  231.3 kl=  51.8
epoch  2: loss=  269.9 recon=  172.2 kl=  60.8
epoch  3: loss=  257.4 recon=  196.6 kl=  61.3
epoch  4: loss=  250.4 recon=  171.4 kl=  57.3
epoch  5: loss=  245.3 recon=  185.2 kl=  65.0
epoch  6: loss=  242.2 recon=  183.4 kl=  63.6
epoch  7: loss=  240.2 recon=  187.2 kl=  68.5
epoch  8: loss=  238.0 recon=  187.4 kl=  67.4
epoch  9: loss=  236.5 recon=  162.9 kl=  63.5
epoch 10: loss=  236.6 recon=  150.4 kl=  65.2
epoch 11: loss=  234.4 recon=  178.7 kl=  67.8
epoch 12: loss=  234.0 recon=  175.7 kl=  69.6
epoch 13: loss=  233.0 recon=  196.0 kl=  72.3
epoch 14: loss=  232.4 recon=  153.1 kl=  70.2
epoch 15: loss=  231.5 recon=  168.1 kl=  65.2
epoch 16: loss=  231.1 recon=  156.6 kl=  69.7
epoch 17: loss=  230.5 recon=  177.1 kl=  69.4
epoch 18: loss=  230.1 recon=  168.1 kl=  70.2
epoch 19: loss=  229.7 recon=  162.4 kl=  66.6
"""

epochs, losses, recons, kls = [], [], [], []
for line in data.strip().split('\n'):
    parts = re.findall(r'[\d\.]+', line)
    if len(parts) >= 4:
        epochs.append(int(parts[0]))
        losses.append(float(parts[1]))
        recons.append(float(parts[2]))
        kls.append(float(parts[3]))

df = pd.DataFrame({'Epoch': epochs, 'Loss': losses, 'Recon Loss': recons, 'KL Divergence': kls})

plt.figure(figsize=(10, 6))
plt.plot(df['Epoch'], df['Loss'], label='Total Loss', color='blue', linewidth=2)
plt.plot(df['Epoch'], df['Recon Loss'], label='Reconstruction Loss', color='green', linewidth=2)
plt.plot(df['Epoch'], df['KL Divergence'], label='KL Divergence', color='red', linewidth=2)

plt.title('VAE Training Loss Over 20 Epochs', fontsize=14, fontweight='bold')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss Value', fontsize=12)
plt.xticks(range(0, 20, 2))
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=11)

plt.tight_layout()
plt.show()