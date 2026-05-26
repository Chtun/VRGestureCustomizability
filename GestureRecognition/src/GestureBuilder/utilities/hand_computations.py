import torch
import torch.nn.functional as F

from GestureBuilder.utilities.naming_conventions import get_finger_indices_list, get_connected_indices_list, get_next_joint_indices_list

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

def quat_inverse_xyzw(q: torch.Tensor) -> torch.Tensor:
    """
    Inverse/conjugate for a unit quaternion stored as (x, y, z, w).
    For unit quaternions inverse = conjugate = (-x, -y, -z, w)
    q: shape (4,) or (...,4)
    returns tensor with same shape
    """
    # keep dtype/device
    return torch.stack([-q[..., 0], -q[..., 1], -q[..., 2], q[..., 3]], dim=-1)

def quat_mult_xyzw(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """
    Multiply q1 * q2 where quaternions are stored as (x, y, z, w).
    Uses the formula where scalar is w = q[3] and vector is v = (x,y,z).
    Inputs q1, q2: shape (...,4) or (4,)
    Returns product in same shape/order (x,y,z,w).
    """
    # extract components
    x1, y1, z1, w1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    x2, y2, z2, w2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]

    # scalar part
    w = w1*w2 - (x1*x2 + y1*y2 + z1*z2)

    # vector part: v = w1*v2 + w2*v1 + v1 x v2
    vx = w1*x2 + w2*x1 + (y1*z2 - z1*y2)
    vy = w1*y2 + w2*y1 + (z1*x2 - x1*z2)
    vz = w1*z2 + w2*z1 + (x1*y2 - y1*x2)

    return torch.stack([vx, vy, vz, w], dim=-1)

def compute_parent_relative_rotations(joint_rotations: torch.Tensor) -> torch.Tensor:
    """
    joint_rotations: (num_frames, num_joints, 4) with ordering (x,y,z,w)
    Returns relative rotations in the same ordering (x,y,z,w).
    We compute q_relative = q_child * inverse(q_parent) (both in wrist space),
    so the relative rotation maps from parent frame into child frame.
    """
    num_frames, num_joints, _ = joint_rotations.shape
    relative_rotations = torch.zeros_like(joint_rotations)
    connected_indices = get_connected_indices_list()  # previous mapping: current -> previous (parent)

    for f in range(num_frames):
        for j in range(num_joints):
            parent_idx = connected_indices[j]  # may be None
            # Use detach().clone() to avoid warnings and keep device/dtype
            q = joint_rotations[f, j].detach().clone()  # (4,) in (x,y,z,w)
            if parent_idx is None:
                # If there's no parent in the finger chain, keep as-is
                relative_rotations[f, j] = q
            else:
                parent_q = joint_rotations[f, parent_idx].detach().clone()
                parent_q_inv = quat_inverse_xyzw(parent_q)
                # q_relative = q_child * inverse(parent_q)
                q_relative = quat_mult_xyzw(q, parent_q_inv)
                relative_rotations[f, j] = q_relative

    return relative_rotations
