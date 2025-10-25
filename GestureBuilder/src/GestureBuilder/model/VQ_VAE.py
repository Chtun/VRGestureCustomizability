import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans

# ==========================================================
# 🔹 Encoder Network
# ==========================================================
class Encoder(nn.Module):
    def __init__(self, input_dim=72, hidden_dim=128, latent_dim=32):
        """
        input_dim = num of joints × 3 coords
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )

    def forward(self, x):
        # x: (batch, 66)
        return self.net(x)


# ==========================================================
# 🔹 Decoder Network
# ==========================================================
class Decoder(nn.Module):
    def __init__(self, latent_dim=32, hidden_dim=128, output_dim=72):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, z_q):
        # z_q: quantized latent (batch, latent_dim)
        return self.net(z_q)


# ===================================================================
# 🔹 Vector Quantizer (VQ layer) with Secondary K-Means++ Clustering
# ===================================================================
class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings=512, embedding_dim=32, commitment_cost=0.25):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost

        # Initialize codebook (K × D)
        self.embeddings = nn.Parameter(torch.randn(num_embeddings, embedding_dim))
        self._last_cluster_epoch = -1  # Track last epoch clustered (optional)

    def forward(self, z_e):
        """
        z_e: encoder output (batch, D)
        """
        # Compute distances between z_e and all embeddings
        z_e_expanded = z_e.unsqueeze(1)  # (B, 1, D)
        e_expanded = self.embeddings.unsqueeze(0)  # (1, K, D)
        distances = torch.sum((z_e_expanded - e_expanded) ** 2, dim=-1)  # (B, K)

        # Find nearest embedding index
        encoding_indices = torch.argmin(distances, dim=1)  # (B,)
        z_q = self.embeddings[encoding_indices]  # (B, D)

        # Compute VQ losses
        commitment_loss = F.mse_loss(z_e.detach(), z_q)
        embedding_loss = F.mse_loss(z_e, z_q.detach())
        vq_loss = embedding_loss + self.commitment_cost * commitment_loss

        # Straight-through estimator
        z_q = z_e + (z_q - z_e).detach()

        return z_q, vq_loss, encoding_indices

    # ======================================================
    # 🔹 Secondary Clustering Step (K-means++)
    # ======================================================
    @torch.no_grad()
    def secondary_cluster(self, distance_threshold=0.1, min_clusters=10, verbose=True):
        """
        Performs K-means++ clustering on the codebook to merge nearby embeddings.

        Args:
            distance_threshold: clusters with centers closer than this (Euclidean) are merged
            min_clusters: minimum number of clusters to maintain
            verbose: print clustering info
        """
        embeddings = self.embeddings.detach().cpu().numpy()  # (K, D)
        K = len(embeddings)

        if K <= min_clusters:
            if verbose:
                print(f"[Clustering] Skipped -- codebook already at min size ({K}).")
            return

        # Start with current number of embeddings and reduce gradually
        for k_try in range(K, min_clusters - 1, -1):
            kmeans = KMeans(
                n_clusters=k_try,
                init="k-means++",
                n_init=10,
                random_state=42
            )
            kmeans.fit(embeddings)
            centers = kmeans.cluster_centers_

            # Compute pairwise distances between centers
            centers_t = torch.tensor(centers)
            dist = torch.cdist(centers_t, centers_t)

            # Mask out self-distances
            mask = (dist > 0) & (dist < distance_threshold)
            close_pairs = mask.sum().item() // 2  # each pair counted twice

            # Get actual distances of the close pairs
            close_distances = dist[mask].tolist()

            if verbose:
                print(f"[Clustering] k_try={k_try} | close cluster pairs < {distance_threshold}: {close_pairs}")
                if close_distances:
                    print(f"-> Distances of close pairs: {close_distances}")

            # Stop if no close clusters remain or reached minimum clusters
            if close_pairs == 0 or k_try == min_clusters:
                if verbose:
                    print(f"[Clustering] Reduced codebook: {K} -> {k_try}")
                new_embeddings = torch.tensor(centers, dtype=self.embeddings.dtype)
                self.embeddings = nn.Parameter(new_embeddings.to(self.embeddings.device))
                self.num_embeddings = k_try
                break


# ==========================================================
# 🔹 Full VQ-VAE Model with encode() method
# ==========================================================
class VQVAE(nn.Module):
    def __init__(self, input_dim=72, hidden_dim=128, latent_dim=32, num_embeddings=512):
        super().__init__()
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
        self.vq = VectorQuantizer(num_embeddings, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim)

    def forward(self, x):
        # Standard forward: encode -> quantize -> decode
        z_e = self.encoder(x)
        z_q, vq_loss, indices = self.vq(z_e)
        x_recon = self.decoder(z_q)
        return x_recon, vq_loss, indices

    # ======================================================
    # 🔹 Encode method for extracting latent embeddings
    # ======================================================
    @torch.no_grad()
    def encode(self, x):
        """
        Encode input x into quantized latent embeddings.

        Args:
            x: (batch, input_dim)
            return_indices (bool): if True, also return VQ indices

        Returns:
            z_q: (batch, latent_dim) quantized embeddings
        """
        z_e = self.encoder(x)
        z_q, _, _ = self.vq(z_e)
        return z_q