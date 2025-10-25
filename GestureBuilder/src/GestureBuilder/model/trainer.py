import torch
from torch.utils.data import DataLoader
import torch.optim as optim

from ..utilities.hand_computations import cosine_denoised_distance
from .VQ_VAE import VQVAE

# ==========================================================
# 🔹 Trainer Class with Validation
# ==========================================================
class VQVAETrainer:
    def __init__(self, model: VQVAE, dataset, batch_size=32, lr=1e-3, device=None, loss_fn=None, val_dataset=None):
        """
        Handles training of the VQ-VAE model with optional validation.
        
        Args:
            model: your VQVAE model
            dataset: HandDataset for training
            batch_size: Number of examples per batch.
            lr: Learning rate.
            device: The device to train on.
            loss_fn: custom loss function, takes (x_recon, x_gt) and returns scalar
            val_dataset: optional HandDataset for validation
        """
        self.device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.dataset = dataset
        self.loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        # Loss function
        self.loss_fn = loss_fn if loss_fn is not None else cosine_denoised_distance

        # Validation
        self.val_dataset = val_dataset
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2) if val_dataset else None

    def train_epoch(self):
        self.model.train()
        total_loss = 0.0
        total_recon = 0.0
        total_vq = 0.0

        for batch_idx, batch in enumerate(self.loader):
            batch = batch.to(self.device)

            # Forward pass
            x_recon, vq_loss, _ = self.model(batch)

            # Flatten for custom loss
            num_joints = x_recon.shape[1] // 3
            x_recon_vec = x_recon.view(-1, num_joints, 3)
            batch_vec = batch.view(-1, num_joints, 3)

            # Compute loss
            recon_loss = self.loss_fn(x_recon_vec, batch_vec)
            loss = recon_loss + vq_loss

            # Backprop
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Track losses
            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_vq += vq_loss.item()

        num_batches = len(self.loader)
        return {
            "loss": total_loss / num_batches,
            "recon_loss": total_recon / num_batches,
            "vq_loss": total_vq / num_batches
        }

    def validate_epoch(self):
        if self.val_loader is None:
            return None

        self.model.eval()
        total_loss = 0.0
        total_recon = 0.0
        total_vq = 0.0

        with torch.no_grad():
            for batch in self.val_loader:
                batch = batch.to(self.device)
                x_recon, vq_loss, _ = self.model(batch)

                num_joints = x_recon.shape[1] // 3
                x_recon_vec = x_recon.view(-1, num_joints, 3)
                batch_vec = batch.view(-1, num_joints, 3)

                recon_loss = self.loss_fn(x_recon_vec, batch_vec)
                loss = recon_loss + vq_loss

                total_loss += loss.item()
                total_recon += recon_loss.item()
                total_vq += vq_loss.item()

        num_batches = len(self.val_loader)
        return {
            "loss": total_loss / num_batches,
            "recon_loss": total_recon / num_batches,
            "vq_loss": total_vq / num_batches
        }

    def train(self, epochs=10, recon_threshold=None):
        """
        Train the VQ-VAE model and trigger secondary clustering
        when reconstruction loss drops below recon_threshold.
        """
        reclustered = False

        for epoch in range(1, epochs + 1):
            stats = self.train_epoch()
            print(f"Epoch {epoch}/{epochs} | "
                f"Train Loss: {stats['loss']:.6f} | "
                f"Recon Loss: {stats['recon_loss']:.6f} | "
                f"VQ Loss: {stats['vq_loss']:.6f}")

            # --- Validation ---
            val_stats = None
            if self.val_loader is not None:
                val_stats = self.validate_epoch()
                if val_stats:
                    print(f"              | "
                        f"Val Loss: {val_stats['loss']:.6f} | "
                        f"Recon: {val_stats['recon_loss']:.6f} | "
                        f"VQ: {val_stats['vq_loss']:.6f}")

            # --- 🔹 Log embedding distances ---
            with torch.no_grad():
                embeddings = self.model.vq.embeddings.detach()
                if embeddings.ndim == 2 and embeddings.shape[0] > 1:
                    dists = torch.cdist(embeddings, embeddings)
                    mean_dist = dists.mean().item()
                    min_dist = dists.min().item()
                    max_dist = dists.max().item()
                    print(f"Codebook distances — Mean: {mean_dist:.4f} | Min: {min_dist:.4f} | Max: {max_dist:.4f}")

                    # --- Trigger reclustering if threshold is reached ---
                    if (not reclustered) and (recon_threshold is not None):
                        current_recon = val_stats['recon_loss'] if val_stats else stats['recon_loss']

                        if current_recon < recon_threshold:
                            print(f"Reconstruction loss {current_recon:.6f} < {recon_threshold}. "
                                f"Triggering secondary clustering...")

                            # 🔹 Adaptive threshold: 0.4 * mean distance
                            adaptive_threshold = 0.3 * mean_dist
                            self.model.vq.secondary_cluster(distance_threshold=adaptive_threshold)
                            print(f"Codebook reclustered using K-means++ with adaptive threshold {adaptive_threshold:.4f}.")

                            reclustered = True  # prevent triggering multiple times
                else:
                    print("Codebook embeddings not available or only one vector — skipping distance log.")

