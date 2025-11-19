from pathlib import Path
import torch
import pandas as pd
import yaml
from GestureBuilder.utilities.naming_conventions import get_hand_joint_list
from GestureBuilder.utilities.sequence_comparison import sequence_distance, VQVAE
from GestureBuilder.model.hand_dataset import load_hand_tensors_from_csv  # same function used on server
from GestureBuilder.utilities.file_operations import load_gestures_from_json

# === Paths ===

# Try to load gesture template names and filenames from the server config
try:
    config_file = Path(__file__).resolve().parents[1] / "server" / "config" / "config.yaml"
    cfg = yaml.safe_load(config_file.read_text())
    templates = cfg.get("gesture_template_paths", []) if isinstance(cfg, dict) else []
    data_folder = Path(cfg.get("paths", Path()).get("data_folder", Path())) if isinstance(cfg, dict) else Path()

    gesture_template_json = cfg.get("paths", {}).get("gesture_template_json", "") if isinstance(cfg, dict) else ""

    MATCH_THRESHOLD = cfg.get("gesture_settings", {}).get("MATCH_THRESHOLD", 1.5) if isinstance(cfg, dict) else 1.5
except Exception:
    templates = []
    raise ValueError("Ur chopped")

# === Model parameters ===
joints_list = get_hand_joint_list()
num_joints = len(joints_list)
input_dim = num_joints * 3
latent_dim = 8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# === Load all gesture sequences ===
if gesture_template_json:
    stored_data = load_gestures_from_json(gesture_template_json)
    gesture_data = []
    csv_names = []

    for gesture_label in stored_data:
        print(f"Loaded gesture label: {gesture_label}")

        

        for hand_data in stored_data[gesture_label]:
            data = []
            csv_names.append(gesture_label)
            for key in hand_data:
                data.append(hand_data[key])
            
            print(len(data))
            gesture_data.append(data)


elif templates:
    # Use the name from config (human-readable) and build full paths from data_folder + path
    csv_names = [t.get("name", Path(t.get("path", "")).stem) for t in templates]
    csv_paths = [data_folder / t.get("path") for t in templates]
    gesture_data = [load_hand_tensors_from_csv(p, joints_list) for p in csv_paths]
else:
    raise ValueError("Ur chopped")

output_folder = Path("../output")
input_VQVAE_model = output_folder / "vqvae_hand_model.pt"

# === Load VQ-VAE model ===
state_dict = torch.load(input_VQVAE_model, map_location="cpu")
num_embeddings = state_dict['vq.embeddings'].shape[0]
print(f"Loaded num_embeddings from state dict: {num_embeddings}")

vqvae_model = VQVAE(input_dim=input_dim, hidden_dim=128, latent_dim=latent_dim, num_embeddings=num_embeddings)
vqvae_model.load_state_dict(state_dict)
vqvae_model.to(DEVICE)
vqvae_model.eval()

# === Compute pairwise DTW distances ===
num_gestures = len(gesture_data)
for i in range(num_gestures):
    for j in range(i + 1, num_gestures):
        left_seq1, right_seq1, _, _, lw1, rw1, _, _ = gesture_data[i]
        left_seq2, right_seq2, _, _, lw2, rw2, _, _ = gesture_data[j]

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
            lw2, rw2,
            debug_statements=True,
            visualize_metrics=False,
        )

        end_time = time.time()

        print(f"DTW Distance between gesture {csv_names[i]} and {csv_names[j]}: {dist:.4f}")
        # print(f"Time to process: {end_time - start_time}")

        if dist >= MATCH_THRESHOLD and csv_names[i] == csv_names[j]:
            print(f"Gestures {csv_names[i]} and {csv_names[j]} are not in match range! (distance: {dist:.4f} < threshold: {MATCH_THRESHOLD})")
        elif dist < MATCH_THRESHOLD and csv_names[i] != csv_names[j]:
            print(f"Gestures {csv_names[i]} and {csv_names[j]} are falsely matched! (distance: {dist:.4f} >= threshold: {MATCH_THRESHOLD})")
