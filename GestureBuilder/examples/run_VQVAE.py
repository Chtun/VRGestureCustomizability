import torch
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader

# Import your dataset and model
from GestureBuilder.model import VQVAE, HandDataset
from GestureBuilder.utilities.naming_conventions import get_hand_joint_list

# === Paths ===
data_folder = Path("C:\\Users\\chtun\\AppData\\LocalLow\\DefaultCompany\\Unity_VR_Template")
val_path = data_folder / "test.csv"
output_folder = Path("../output")
input_VQVAE_model = output_folder / "vqvae_hand_model.pt"

# === Load validation dataset ===
val_dataset = HandDataset(val_path)
val_loader = DataLoader(val_dataset, batch_size=4096, shuffle=False)

# === Load state dict ===
state_dict = torch.load(input_VQVAE_model, map_location="cpu")
num_embeddings = state_dict['vq.embeddings'].shape[0]
print(f"Loaded num_embeddings from state dict: {num_embeddings}")

# === Model parameters ===
joints_list = get_hand_joint_list()
num_joints = len(joints_list)
input_dim = num_joints * 3
latent_dim = 8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# === Initialize and load model ===
vqvae_model = VQVAE(input_dim=input_dim, hidden_dim=128, latent_dim=latent_dim, num_embeddings=num_embeddings)
vqvae_model.load_state_dict(torch.load(input_VQVAE_model, map_location=DEVICE))
vqvae_model.to(DEVICE)
vqvae_model.eval()
# === Extract latent embeddings for all validation samples ===
all_latents = []

with torch.no_grad():
    for batch in val_loader:
        batch = batch.to(DEVICE)
        z_q = vqvae_model.encode(batch)  # output of encoder
        # Optional: use embedding vectors if VQVAE returns indices
        if isinstance(z_q, tuple):  # sometimes VQVAE returns (quantized, indices)
            z_q = z_q[0]
        all_latents.append(z_q)

# Stack all embeddings
all_latents = torch.cat(all_latents, dim=0).cpu().numpy()
print(f"Latent embeddings shape: {all_latents.shape}")

import numpy as np
from scipy.spatial.distance import pdist, squareform

# === Compute pairwise Euclidean distances between all latent embeddings ===
dist_matrix = squareform(pdist(all_latents, metric='euclidean'))

# === Summary statistics ===
print("\n=== Pairwise Euclidean Distance Summary ===")
print(f"Mean distance: {dist_matrix.mean():.4f}")
print(f"Min distance:  {dist_matrix.min():.4f}")
print(f"Max distance:  {dist_matrix.max():.4f}")
print(f"Std deviation: {dist_matrix.std():.4f}")

# === t-SNE projection to 2D ===
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
latents_2d = tsne.fit_transform(all_latents)

# === Plot ===
plt.figure(figsize=(8, 8))
plt.scatter(latents_2d[:, 0], latents_2d[:, 1], c='blue', s=20, alpha=0.6)
plt.title("t-SNE of VQ-VAE Latent Embeddings (Test Set)")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.grid(True)
plt.show()
