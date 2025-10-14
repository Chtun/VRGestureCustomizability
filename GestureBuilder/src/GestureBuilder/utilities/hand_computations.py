import torch
import torch.nn.functional as F

from GestureBuilder.utilities.naming_conventions import *

# ==========================================================
# 🔹 Hand Distance Function: Denoised Cosine Distance
# ==========================================================
def cosine_denoised_distance(VA, VB, pt=0.5) -> torch.Tensor:
    """
    Compute denoised cosine distance between two hand postures (as defined in GestureBuilder paper).
    
    Args:
        VA, VB: tensors of shape (batch, 24, 3)
        pt: threshold for truncation
    
    Returns:
        The average cosine-denoised distance over the entire batch.
    """
    VA_norm = F.normalize(VA, dim=-1)
    VB_norm = F.normalize(VB, dim=-1)
    
    batch_size = VA.shape[0]
    total_dist = torch.zeros(batch_size, device=VA.device)
    
    finger_indices = get_finger_indices_list()

    # Sum over fingers
    for finger in finger_indices:
        # Compute cosine similarity for joints in this finger
        cos_sim = torch.sum(VA_norm[:, finger, :] * VB_norm[:, finger, :], dim=-1)  # (batch, Nj)
        csim = 1 - cos_sim  # CSIM(v_a, v_b)
        
        # Apply truncation
        csim = torch.where(csim < pt, torch.zeros_like(csim), csim)
        
        # Sum over joints in finger
        finger_dist = torch.sum(csim, dim=-1)  # (batch,)
        total_dist += finger_dist
    
    # Return mean over batch
    return torch.mean(total_dist)


# ==========================================================
# 🔹 Hand Vector Conversion
# ==========================================================
def convert_joint_to_hand_vector(JA: torch.Tensor) -> torch.Tensor:
    """
    Converts Joint points into hand vectors along the skeleton.

    Each vector corresponds to the vector from the previous joint to the current joint.
    If a joint has no previous joint (e.g., root), its vector is just its absolute position.

    Args:
        JA: Tensor of shape (batch, 24, 3), joint positions in wrist space.

    Returns:
        VA: Tensor of shape (batch, 24, 3), vectors along the hand skeleton.
    """
    batch_size, num_joints, _ = JA.shape
    connected = get_connected_indices_list()  # dict: current_joint -> previous_joint

    VA = torch.zeros_like(JA)

    for joint_idx, prev_idx in connected.items():
        if prev_idx is not None:
            # Vector from previous joint to current joint
            VA[:, joint_idx, :] = JA[:, joint_idx, :] - JA[:, prev_idx, :]
        else:
            # No previous joint; use absolute position
            VA[:, joint_idx, :] = JA[:, joint_idx, :]

    return VA

