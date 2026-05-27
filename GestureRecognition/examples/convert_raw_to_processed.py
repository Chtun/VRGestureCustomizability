from pathlib import Path
import torch
import pandas as pd
import yaml
from typing import Dict, List

from GestureBuilder.utilities.naming_conventions import get_hand_joint_list
from GestureBuilder.utilities.sequence_comparison import sequence_distance, VQVAE
from GestureBuilder.model.hand_dataset import load_hand_tensors_from_csv  # same function used on server
from GestureBuilder.utilities.file_operations import load_gestures_from_json, save_gestures_to_json

# === Paths ===

# Try to load gesture template names and filenames from the server config
try:
    config_file = Path(__file__).resolve().parents[1] / "server" / "config" / "config.yaml"
    cfg = yaml.safe_load(config_file.read_text())
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


JOINTS_LIST = get_hand_joint_list()
default_gesture_templates: Dict[str, List[Dict[str, torch.Tensor]]] = {}

if not cfg['gesture_template_paths'] is None:
    gesture_template_paths = [
        (item['name'], data_folder / item['path'])
        for item in cfg['gesture_template_paths']
    ]

    for pair in gesture_template_paths:
        gesture_key =  pair[0]
        template_path = pair[1]

        if gesture_key not in default_gesture_templates:
            default_gesture_templates[gesture_key] = []

        (left_hand_vectors, right_hand_vectors, left_joint_rotations, right_joint_rotations,
        left_wrist_positions, right_wrist_positions,
        left_wrist_rotations, right_wrist_rotations) = load_hand_tensors_from_csv(template_path, JOINTS_LIST)
        gesture_dict = {
            "left_hand_vectors": left_hand_vectors, # shape: (num_frames, 72)
            "right_hand_vectors": right_hand_vectors, # shape: (num_frame   s, 72)
            "left_joint_rotations": left_joint_rotations, # shape: (num_frames, 24, 4)
            "right_joint_rotations": right_joint_rotations, # shape: (num_frames, 24, 4)
            "left_wrist_positions": left_wrist_positions, # shape: (num_frames, 3)
            "right_wrist_positions": right_wrist_positions, # shape: (num_frames, 3)
            "left_wrist_rotations": left_wrist_rotations, # shape: (num_frames, 4)
            "right_wrist_rotations": right_wrist_rotations # shape: (num_frames, 4)
        }

        default_gesture_templates[gesture_key].append(gesture_dict)
else:
    raise ValueError("Ur chopped")

save_gestures_to_json(default_gesture_templates, gesture_template_json)