import torch
import json
from typing import List, Dict, Union
from pathlib import Path

# Default file path
GESTURE_TEMPLATES_FILE = Path("./database/stored_gesture_templates.json")


def save_gestures_to_json(
    gesture_templates: Dict[str, List[Dict[str, torch.Tensor]]],
    path: Union[str, Path] = GESTURE_TEMPLATES_FILE,
):
    """Save gesture_templates dict to a JSON file (accepts str or Path)."""
    try:
        path = Path(path)  # Normalize to Path

        serializable_gestures = {}
        for gesture_key, templates in gesture_templates.items():
            serializable_gestures[gesture_key] = []
            for template in templates:
                serializable_gestures[gesture_key].append({
                    "left_hand_vectors": template["left_hand_vectors"].cpu().tolist(),
                    "right_hand_vectors": template["right_hand_vectors"].cpu().tolist(),
                    "left_wrist_root": template["left_wrist_root"].cpu().tolist(),
                    "right_wrist_root": template["right_wrist_root"].cpu().tolist(),
                })

        # Ensure parent directory exists
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            json.dump(serializable_gestures, f, indent=2)

        print(f"[INFO] Saved {len(serializable_gestures)} gesture templates to {path}")

    except Exception as e:
        print(f"[ERROR] Failed to save gestures: {e}")


def load_gestures_from_json(
    path: Union[str, Path] = GESTURE_TEMPLATES_FILE,
) -> Dict[str, List[Dict[str, torch.Tensor]]]:
    """Load gesture_templates dict from a JSON file (accepts str or Path)."""
    path = Path(path)  # Normalize to Path

    if not path.exists():
        print(f"[INFO] No gesture template file found at {path}.")
        return {}

    try:
        with open(path, "r") as f:
            data = json.load(f)

        loaded_gestures = {}
        for gesture_key, templates in data.items():
            loaded_gestures[gesture_key] = []
            for t in templates:
                loaded_gestures[gesture_key].append({
                    "left_hand_vectors": torch.tensor(t["left_hand_vectors"], dtype=torch.float32),
                    "right_hand_vectors": torch.tensor(t["right_hand_vectors"], dtype=torch.float32),
                    "left_wrist_root": torch.tensor(t["left_wrist_root"], dtype=torch.float32),
                    "right_wrist_root": torch.tensor(t["right_wrist_root"], dtype=torch.float32),
                })

        print(f"[INFO] Loaded {len(loaded_gestures)} gesture templates from JSON.")
        return loaded_gestures

    except Exception as e:
        print(f"[ERROR] Failed to load gestures: {e}")
        return {}
