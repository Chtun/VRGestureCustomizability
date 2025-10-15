import torch
from torch.utils.data import Dataset
import pandas as pd
from pathlib import Path

from GestureBuilder.utilities.naming_conventions import get_hand_joint_list
from GestureBuilder.utilities.hand_computations import convert_joint_to_hand_vector

# ==========================================================
# 🔹 Hand Dataset
# ==========================================================
class HandDataset(Dataset):
    def __init__(self, csv_path: Path):
        """
        Reads the CSV once and converts all left/right hand frames into hand vectors.
        Each hand is treated as a separate example.
        """
        self.df = pd.read_csv(csv_path)
        
        # joint names in order
        self.joint_names = get_hand_joint_list()
        num_joints = len(self.joint_names)
        
        # Precompute all hand vectors
        all_vectors = []
        for hand_prefix in ["L_", "R_"]:
            cols = []
            for j in self.joint_names:
                cols.extend([f"{hand_prefix}{j}_posX",
                             f"{hand_prefix}{j}_posY",
                             f"{hand_prefix}{j}_posZ"])
            
            # Extract joint positions
            JA = torch.tensor(self.df[cols].values, dtype=torch.float32)  # (num_rows, num_joints * 3)
            JA = JA.view(-1, num_joints, 3)  # (num_rows, num_joints, 3)

            # Convert all joints to hand vectors in batch
            VA = convert_joint_to_hand_vector(JA)  # (num_rows, num_joints, 3)

            # Flatten for VQVAE input
            VA_flat = VA.view(VA.shape[0], -1)  # (num_rows, num_joints * 3)
            
            all_vectors.append(VA_flat)
        
        # Concatenate left and right hands into one long tensor
        self.data = torch.cat(all_vectors, dim=0)  # (num_rows * 2, num_joints * 3)
    
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        return self.data[idx]


def joints_dict_to_tensor(joints_dict, joint_names):
    """
    Converts a dictionary of joint_name -> [x, y, z] into a tensor of relative hand vectors.

    Steps:
        1. Extracts joint positions from the input dictionary following the canonical joint order.
        2. Converts the list of positions into a tensor of shape (num_joints, 3).
        3. Converts the absolute joint positions into relative hand vectors
           using `convert_joint_to_hand_vector`, consistent with real-time processing.

    Args:
        joints_dict (dict): Dictionary mapping joint names to [x, y, z] positions.
        joint_names (list[str]): Ordered list of joint names to preserve consistent indexing.

    Returns:
        torch.Tensor: Tensor of relative hand vectors with shape (num_joints, 3).
    """
    joints_list = []
    for jname in joint_names:
        pos = joints_dict[jname]  # Each entry is [x, y, z]
        joints_list.append(pos)

    # Convert to tensor of shape (num_joints, 3)
    J = torch.tensor(joints_list, dtype=torch.float32)

    # Add batch/frame dimension for compatibility with converter
    J = J.unsqueeze(0)  # (1, num_joints, 3)

    # Convert absolute joint positions to relative hand vectors
    V = convert_joint_to_hand_vector(J)  # (1, num_joints, 3)

    # Remove batch dimension -> (num_joints, 3)
    V = V.squeeze(0)

    return V

def load_hand_tensors_from_csv(csv_path, joints_list):
    """
    Loads hand joint data and converts it into flattened hand vectors
    consistent with live server data format.

    Args:
        csv_path (Path or str): Path to the gesture CSV file.
        joints_list (list[str]): Ordered list of joint names (same as server order).

    Returns:
        left_flat (Tensor): (num_frames, num_joints * 3)
        right_flat (Tensor): (num_frames, num_joints * 3)
        left_wrist (Tensor): (num_frames, 3)
        right_wrist (Tensor): (num_frames, 3)
    """
    df = pd.read_csv(csv_path)
    num_joints = len(joints_list)

    def extract_hand_joints(prefix):
        # --- Build column names ---
        cols = []
        for j in joints_list:
            cols.extend([
                f"{prefix}{j}_posX",
                f"{prefix}{j}_posY",
                f"{prefix}{j}_posZ"
            ])
        
        # --- Extract joint positions ---
        JA = torch.tensor(df[cols].values, dtype=torch.float32)  # (num_frames, num_joints * 3)
        JA = JA.view(-1, num_joints, 3)  # (num_frames, num_joints, 3)

        # --- Convert to relative hand vectors (as used in real-time server) ---
        VA = convert_joint_to_hand_vector(JA)  # (num_frames, num_joints, 3)

        # --- Flatten for VQVAE input ---
        VA_flat = VA.view(VA.shape[0], -1)  # (num_frames, num_joints * 3)
        return VA_flat

    # --- Extract both hands ---
    left_flat = extract_hand_joints("L_")
    right_flat = extract_hand_joints("R_")

    # --- Extract wrist positions (consistent with server-side keys) ---
    left_wrist = torch.tensor(
        df[["L_Root_posX", "L_Root_posY", "L_Root_posZ"]].values,
        dtype=torch.float32
    )
    right_wrist = torch.tensor(
        df[["R_Root_posX", "R_Root_posY", "R_Root_posZ"]].values,
        dtype=torch.float32
    )

    return left_flat, right_flat, left_wrist, right_wrist