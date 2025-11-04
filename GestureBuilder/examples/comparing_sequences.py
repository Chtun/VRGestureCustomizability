from pathlib import Path
import torch
import pandas as pd
from GestureBuilder.utilities.naming_conventions import get_hand_joint_list
from GestureBuilder.utilities.sequence_comparison import sequence_distance, VQVAE
from GestureBuilder.model.hand_dataset import load_hand_tensors_from_csv  # same function used on server

# === Paths ===
data_folder = Path("C:\\Users\\chtun\\AppData\\LocalLow\\DefaultCompany\\Unity_VR_Template")

csv_paths = [
    data_folder / "hand_data-right_jab-1.csv",
    data_folder / "hand_data-right_jab-2.csv",
    data_folder / "hand_data-right_jab-3.csv",
    data_folder / "hand_data-right_jab-Evan-1.csv",
    data_folder / "hand_data-table_bang-1.csv",
]

output_folder = Path("../output")
input_VQVAE_model = output_folder / "vqvae_hand_model.pt"

# === Load VQ-VAE model ===
state_dict = torch.load(input_VQVAE_model, map_location="cpu")
num_embeddings = state_dict['vq.embeddings'].shape[0]
print(f"Loaded num_embeddings from state dict: {num_embeddings}")

# === Model parameters ===
joints_list = get_hand_joint_list()
num_joints = len(joints_list)
input_dim = num_joints * 3
latent_dim = 8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

vqvae_model = VQVAE(input_dim=input_dim, hidden_dim=128, latent_dim=latent_dim, num_embeddings=num_embeddings)
vqvae_model.load_state_dict(state_dict)
vqvae_model.to(DEVICE)
vqvae_model.eval()

# === Load all gesture sequences ===
gesture_data = [load_hand_tensors_from_csv(p, joints_list) for p in csv_paths]

# === Compute pairwise DTW distances ===
num_gestures = len(gesture_data)
for i in range(num_gestures):
    for j in range(i + 1, num_gestures):
        left_seq1, right_seq1, lw1, rw1, _, _ = gesture_data[i]
        left_seq2, right_seq2, lw2, rw2, _, _ = gesture_data[j]

        # print("First sequence shapes:")
        # print(left_seq1.shape)
        # print(right_seq1.shape)
        # print(lw1.shape)
        # print(rw1.shape)

        # print("Second sequence shapes:")
        # print(left_seq2.shape)
        # print(right_seq2.shape)
        # print(lw2.shape)
        # print(rw2.shape)

        # print("Left sequences of first and second gesture:")
        # print(left_seq1)
        # print(left_seq2)

        # print("Right sequences of first and second gesture:")
        # print(right_seq1)
        # print(right_seq2)

        # print("Left wrist positions of first and second gesture:")
        # print(lw1)
        # print(rw1)

        # print("Right wrist positions of first and second gesture:")
        # print(rw1)
        # print(rw2)

        import time

        start_time = time.time()

        dist = sequence_distance(
            vqvae_model,
            left_seq1, right_seq1,
            lw1, rw1,
            left_seq2, right_seq2,
            lw2, rw2
        )

        end_time = time.time()

        print(f"DTW Distance between gesture {i+1} and {j+1}: {dist:.4f}")
        print(f"Time to process: {end_time - start_time}")
