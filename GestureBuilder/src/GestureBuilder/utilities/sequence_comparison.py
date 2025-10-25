import torch
import torch.nn.functional as F

import matplotlib.pyplot as plt

from ..model.VQ_VAE import VQVAE

def compute_frame_distance_batch(
    latent_left_A, latent_right_A,
    latent_left_B, latent_right_B,
    left_wrist_pos_A, right_wrist_pos_A,
    left_wrist_pos_B, right_wrist_pos_B,
    left_wrist_vel_A, right_wrist_vel_A,
    left_wrist_vel_B, right_wrist_vel_B,
    alpha_wrist: float = 0.5,
    LATENT_MAX: float = 6.0
):
    """
    Compute pairwise frame-wise distances between two sequences of hand frames in batch.
    Matches exactly the behavior of compute_frame_distance (single-frame version).

    Velocities are T-1, positions are T.

    Returns:
        frame_matrix: (T1-1, T2-1)
    """
    # Slice positions to match velocities
    left_wrist_pos_A = left_wrist_pos_A[1:]
    right_wrist_pos_A = right_wrist_pos_A[1:]
    left_wrist_pos_B = left_wrist_pos_B[1:]
    right_wrist_pos_B = right_wrist_pos_B[1:]
    latent_left_A = latent_left_A[1:]
    latent_right_A = latent_right_A[1:]
    latent_left_B = latent_left_B[1:]
    latent_right_B = latent_right_B[1:]

    # Latent distance
    latent_left_A_exp = latent_left_A.unsqueeze(1)
    latent_left_B_exp = latent_left_B.unsqueeze(0)
    latent_right_A_exp = latent_right_A.unsqueeze(1)
    latent_right_B_exp = latent_right_B.unsqueeze(0)
    latent_dist_left = torch.norm(latent_left_A_exp - latent_left_B_exp, dim=-1) / LATENT_MAX
    latent_dist_right = torch.norm(latent_right_A_exp - latent_right_B_exp, dim=-1) / LATENT_MAX
    latent_dist = (latent_dist_left + latent_dist_right) / 2.0

    # Wrist velocity distance: |norm(A) - norm(B)|
    left_vel_A_exp = left_wrist_vel_A.unsqueeze(1)
    left_vel_B_exp = left_wrist_vel_B.unsqueeze(0)
    right_vel_A_exp = right_wrist_vel_A.unsqueeze(1)
    right_vel_B_exp = right_wrist_vel_B.unsqueeze(0)
    left_wrist_dist = torch.abs(torch.norm(left_vel_A_exp, dim=-1) - torch.norm(left_vel_B_exp, dim=-1))
    right_wrist_dist = torch.abs(torch.norm(right_vel_A_exp, dim=-1) - torch.norm(right_vel_B_exp, dim=-1))

    # Wrist position distances: |norm(left-right)_A - norm(left-right)_B|
    wrist_dist_A = torch.norm(left_wrist_pos_A - right_wrist_pos_A, dim=-1)  # (T1-1)
    wrist_dist_B = torch.norm(left_wrist_pos_B - right_wrist_pos_B, dim=-1)  # (T2-1)
    mag_diff = torch.abs(wrist_dist_A.unsqueeze(1) - wrist_dist_B.unsqueeze(0))

    # Wrist angles: |angle_A - angle_B|
    cos_angle_A = F.cosine_similarity(left_wrist_pos_A, right_wrist_pos_A, dim=-1)  # (T1-1)
    cos_angle_B = F.cosine_similarity(left_wrist_pos_B, right_wrist_pos_B, dim=-1)  # (T2-1)
    angle_A = torch.acos(torch.clamp(cos_angle_A, -1.0, 1.0))
    angle_B = torch.acos(torch.clamp(cos_angle_B, -1.0, 1.0))
    angle_diff = torch.abs(angle_A.unsqueeze(1) - angle_B.unsqueeze(0))

    wrist_dist = mag_diff + angle_diff
    wrist_dist = 20 * (wrist_dist / 5.0 + (left_wrist_dist + right_wrist_dist) / 2.0)

    frame_matrix = alpha_wrist * wrist_dist + (1 - alpha_wrist) * latent_dist
    return frame_matrix

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

def plot_wrist_metrics(
    left_wrist_seq1: torch.Tensor,
    right_wrist_seq1: torch.Tensor,
    left_wrist_seq2: torch.Tensor,
    right_wrist_seq2: torch.Tensor,
    dtw_path: list = None
):
    """
    Plot wrist metrics: velocities, wrist distances, and angles between wrists.
    If dtw_path is provided, align seq2 to seq1 using the DTW path.

    Args:
        left_wrist_seq1, right_wrist_seq1: T1 x 3 tensors for seq1
        left_wrist_seq2, right_wrist_seq2: T2 x 3 tensors for seq2
        dtw_path: optional list of (i, j) index tuples from DTW alignment
    """
    device = left_wrist_seq1.device

    # Align seq2 based on DTW path if provided
    if dtw_path is not None:
        seq2_aligned_indices = [j for i, j in dtw_path]
        aligned_left_seq2 = left_wrist_seq2[torch.tensor(seq2_aligned_indices, device=device)]
        aligned_right_seq2 = right_wrist_seq2[torch.tensor(seq2_aligned_indices, device=device)]
    else:
        aligned_left_seq2 = left_wrist_seq2
        aligned_right_seq2 = right_wrist_seq2

    # Compute wrist velocities
    vel_left_seq1 = left_wrist_seq1[1:] - left_wrist_seq1[:-1]
    vel_right_seq1 = right_wrist_seq1[1:] - right_wrist_seq1[:-1]
    vel_left_seq2 = aligned_left_seq2[1:] - aligned_left_seq2[:-1]
    vel_right_seq2 = aligned_right_seq2[1:] - aligned_right_seq2[:-1]

    # Compute magnitudes
    mag_left_seq1 = torch.norm(vel_left_seq1, dim=-1)
    mag_right_seq1 = torch.norm(vel_right_seq1, dim=-1)
    mag_left_seq2 = torch.norm(vel_left_seq2, dim=-1)
    mag_right_seq2 = torch.norm(vel_right_seq2, dim=-1)

    # Compute wrist distances
    wrist_distance_seq1 = torch.norm(left_wrist_seq1 - right_wrist_seq1, dim=-1)
    wrist_distance_seq2 = torch.norm(aligned_left_seq2 - aligned_right_seq2, dim=-1)

    # Compute angles between left and right wrists
    cos_angle_seq1 = F.cosine_similarity(left_wrist_seq1, right_wrist_seq1, dim=-1)
    cos_angle_seq2 = F.cosine_similarity(aligned_left_seq2, aligned_right_seq2, dim=-1)
    angle_seq1 = torch.acos(torch.clamp(cos_angle_seq1, -1.0, 1.0))
    angle_seq2 = torch.acos(torch.clamp(cos_angle_seq2, -1.0, 1.0))

    # Plot metrics
    plt.figure(figsize=(12, 6))
    plt.plot(mag_left_seq1.cpu(), label='Left seq1 magnitude', marker='o')
    plt.plot(mag_right_seq1.cpu(), label='Right seq1 magnitude', marker='o')
    plt.plot(mag_left_seq2.cpu(), label='Left seq2 magnitude', marker='x')
    plt.plot(mag_right_seq2.cpu(), label='Right seq2 magnitude', marker='x')

    plt.plot(wrist_distance_seq1.cpu(), label='Seq1 wrist distance', linestyle='--')
    plt.plot(wrist_distance_seq2.cpu(), label='Seq2 wrist distance', linestyle='--')

    plt.plot(angle_seq1.cpu(), label='Seq1 angle (left-right)', linestyle=':')
    plt.plot(angle_seq2.cpu(), label='Seq2 angle (left-right)', linestyle=':')

    plt.xlabel('Frame')
    plt.ylabel('Value')
    plt.title('Seq1 vs Seq2 Wrist Metrics')
    plt.legend()
    plt.grid(True)
    plt.show()


def dtw_subsequence_full_alignment(
    latent_left_seq1: torch.Tensor,
    latent_right_seq1: torch.Tensor,
    latent_left_seq2: torch.Tensor,
    latent_right_seq2: torch.Tensor,
    left_wrist_seq1: torch.Tensor,
    right_wrist_seq1: torch.Tensor,
    left_wrist_seq2: torch.Tensor,
    right_wrist_seq2: torch.Tensor,
    alpha_wrist: float = 0.5,
    device: str = "cpu"
):
    """
    DTW distance for a subsequence match where seq1 must fully fit inside seq2.
    Uses velocity-based rotation-invariant motion comparison.
    """

    T1 = latent_left_seq1.shape[0]
    T2 = latent_left_seq2.shape[0]
    # print(f"\n[DTW] Sequence lengths: seq1={T1}, seq2={T2}")

    if T1 > T2:
        raise ValueError("seq1 must be shorter than or equal to seq2 for full alignment.")

    # Compute wrist velocities (frame-to-frame motion)
    vel_left_seq1 = left_wrist_seq1[1:] - left_wrist_seq1[:-1]
    vel_right_seq1 = right_wrist_seq1[1:] - right_wrist_seq1[:-1]
    vel_left_seq2 = left_wrist_seq2[1:] - left_wrist_seq2[:-1]
    vel_right_seq2 = right_wrist_seq2[1:] - right_wrist_seq2[:-1]

    # plot_wrist_metrics(
    #     left_wrist_seq1=left_wrist_seq1,
    #     left_wrist_seq2=left_wrist_seq2,
    #     right_wrist_seq1=right_wrist_seq1,
    #     right_wrist_seq2=right_wrist_seq2,
    # )

    # Adjust lengths (velocities are one shorter)
    T1_vel = T1 - 1
    T2_vel = T2 - 1

    # Compute full frame distance matrix at once
    frame_matrix = compute_frame_distance_batch(
        latent_left_seq1, latent_right_seq1,
        latent_left_seq2, latent_right_seq2,
        left_wrist_seq1, right_wrist_seq1,
        left_wrist_seq2, right_wrist_seq2,
        vel_left_seq1, vel_right_seq1,
        vel_left_seq2, vel_right_seq2,
        alpha_wrist=alpha_wrist
    )  # shape: (T1, T2)

    # # Precompute frame distance matrix
    # frame_matrix = torch.zeros((T1_vel, T2_vel))
    # for i in range(T1_vel):
    #     for j in range(T2_vel):
    #         frame_matrix[i, j] = compute_frame_distance(
    #             latent_left_seq1[i + 1], latent_right_seq1[i + 1],
    #             latent_left_seq2[j + 1], latent_right_seq2[j + 1],
    #             left_wrist_seq1[i+1], right_wrist_seq1[i+1],
    #             left_wrist_seq2[j+1], right_wrist_seq2[j+1],
    #             vel_left_seq1[i], vel_right_seq1[i],
    #             vel_left_seq2[j], vel_right_seq2[j],
    #             alpha_latent=alpha_latent,
    #             alpha_wrist=alpha_wrist
    #         )

    # # Visualize the Frame cost matrix
    # plt.figure(figsize=(8, 6))
    # plt.imshow(frame_matrix.cpu().numpy(), origin='lower', cmap='viridis', aspect='auto')
    # plt.colorbar(label='Accumulated Cost')
    # plt.xlabel('Sequence 2 Index')
    # plt.ylabel('Sequence 1 Index')
    # plt.title('DTW Cost Matrix')
    # plt.show()

    # DTW matrix
    T1_vel, T2_vel = frame_matrix.shape
    D = torch.full((T1_vel + 1, T2_vel + 1), float('inf'), device=device)
    D[0, :] = 0.0
    D[0, 0] = 0.0

    backtrack = torch.zeros((T1_vel + 1, T2_vel + 1), dtype=torch.int, device=device)
    penalty = 0.005

    costs = torch.empty(3, device=device)

    for i in range(1, T1_vel + 1):
        diag = D[i - 1, :-1]
        up   = D[i - 1, 1:] + penalty
        left = D[i, :-1] + penalty

        costs = torch.stack([diag, up, left], dim=0)
        min_costs, min_idx = costs.min(dim=0)

        D[i, 1:] = frame_matrix[i - 1, :] + min_costs
        backtrack[i, 1:] = min_idx


    # # Visualize the DTW cost matrix
    # plt.figure(figsize=(8, 6))
    # plt.imshow(D.cpu().numpy(), origin='lower', cmap='viridis', aspect='auto')
    # plt.colorbar(label='Accumulated Cost')
    # plt.xlabel('Sequence 2 Index')
    # plt.ylabel('Sequence 1 Index')
    # plt.title('DTW Cost Matrix')
    # plt.show()

    # Subsequence alignment: find min in last row
    dtw_costs = D[T1_vel, 1:]
    dtw_dist_norm, end_idx = dtw_costs.min(0)
    dtw_dist_norm = dtw_dist_norm.item()
    end_idx = end_idx.item() + 1  # add 1 because we sliced D[:,1:]

    # reconstruct the alignment path
    i, j = T1_vel, end_idx
    path = []
    while i > 0 and j > 0:
        path.append((i-1, j-1))
        step = backtrack[i, j].item()
        if step == 0:      # diag
            i -= 1
            j -= 1
        elif step == 1:    # up
            i -= 1
        elif step == 2:    # left
            j -= 1
    path.reverse()
    best_start = path[0]

    # plot_wrist_metrics(
    #     left_wrist_seq1=left_wrist_seq1,
    #     left_wrist_seq2=left_wrist_seq2,
    #     right_wrist_seq1=right_wrist_seq1,
    #     right_wrist_seq2=right_wrist_seq2,
    #     dtw_path=path
    # )
    

    # print(path)
    seq2_span = path[-1][1] - path[0][1] + 1  # how many frames in seq2 the match spanned

    if seq2_span > 0:
        dtw_dist_norm /= seq2_span

    return dtw_dist_norm, best_start




def sequence_distance(
    vqvae_model: VQVAE,
    left_hand_seq1: torch.Tensor,
    right_hand_seq1: torch.Tensor,
    left_wrist_seq1: torch.Tensor,
    right_wrist_seq1: torch.Tensor,
    left_hand_seq2: torch.Tensor,
    right_hand_seq2: torch.Tensor,
    left_wrist_seq2: torch.Tensor,
    right_wrist_seq2: torch.Tensor,
    alpha_wrist: float = 0.3,
):
    """
    Compute a DTW-based distance between two gesture sequences using both hands and wrist positions.

    The function automatically ensures the shorter sequence is treated as `seq1` for
    full subsequence alignment and uses the device of the model automatically.

    Args:
        vqvae_model (VQVAE): Pre-trained VQVAE model used to encode hand joint sequences.
        left_hand_seq1, right_hand_seq1 (torch.Tensor): Left and right hand sequences for gesture 1
            with shape (num_frames, num_joints * 3).
        left_hand_seq2, right_hand_seq2 (torch.Tensor): Left and right hand sequences for gesture 2
            with shape (num_frames, num_joints * 3).
        left_wrist_seq1, right_wrist_seq1, left_wrist_seq2, right_wrist_seq2 (torch.Tensor):
            Wrist sequences for both gestures with shape (num_frames, 3).
        alpha_latent (float): Weight for the latent space distance (default 0.7).
        alpha_wrist (float): Weight for the wrist distance (default 0.7).

    Raises:
        ValueError: If any hand sequence does not have shape (num_frames, num_joints * 3)
                    or if any wrist sequence does not have shape (num_frames, 3).

    Returns:
        float: The computed DTW distance between the two gestures.
    """
    vqvae_model.eval()

    # Automatically get the model's device
    device = next(vqvae_model.parameters()).device

    # Determine expected feature dimension for hands
    encoder_input_size = vqvae_model.encoder.net[0].in_features

    # Validate hand sequence shapes
    for seq_name, seq in [
        ("left_hand_seq1", left_hand_seq1),
        ("right_hand_seq1", right_hand_seq1),
        ("left_hand_seq2", left_hand_seq2),
        ("right_hand_seq2", right_hand_seq2)
    ]:
        if seq.ndim != 2 or seq.shape[1] != encoder_input_size:
            raise ValueError(
                f"{seq_name} has invalid shape {tuple(seq.shape)}, expected (num_frames, {encoder_input_size})"
            )

    # Validate wrist sequence shapes
    for seq_name, seq in [
        ("left_wrist_seq1", left_wrist_seq1),
        ("right_wrist_seq1", right_wrist_seq1),
        ("left_wrist_seq2", left_wrist_seq2),
        ("right_wrist_seq2", right_wrist_seq2)
    ]:
        if seq.ndim != 2 or seq.shape[1] != 3:
            raise ValueError(
                f"{seq_name} has invalid shape {tuple(seq.shape)}, expected (num_frames, 3)"
            )

    # print(f"\n=== Sequence Distance Debug ===")
    # print(f"Input sizes: {encoder_input_size}-dim encoder, device={device}")
    # print(f"Left seq1 shape: {tuple(left_hand_seq1.shape)}, Right seq1: {tuple(right_hand_seq1.shape)}")
    # print(f"Left seq2 shape: {tuple(left_hand_seq2.shape)}, Right seq2: {tuple(right_hand_seq2.shape)}")
    # print(f"Wrist seq shapes: left1 {tuple(left_wrist_seq1.shape)}, right1 {tuple(right_wrist_seq1.shape)}, left2 {tuple(left_wrist_seq2.shape)}, right2 {tuple(right_wrist_seq2.shape)}")

    # Move all inputs to the same device as the model
    left_hand_seq1 = left_hand_seq1.to(device)
    right_hand_seq1 = right_hand_seq1.to(device)
    left_hand_seq2 = left_hand_seq2.to(device)
    right_hand_seq2 = right_hand_seq2.to(device)
    left_wrist_seq1 = left_wrist_seq1.to(device)
    right_wrist_seq1 = right_wrist_seq1.to(device)
    left_wrist_seq2 = left_wrist_seq2.to(device)
    right_wrist_seq2 = right_wrist_seq2.to(device)

    # Encode
    with torch.no_grad():
        latent_left_seq1 = vqvae_model.encode(left_hand_seq1)
        latent_right_seq1 = vqvae_model.encode(right_hand_seq1)
        latent_left_seq2 = vqvae_model.encode(left_hand_seq2)
        latent_right_seq2 = vqvae_model.encode(right_hand_seq2)

    # print(f"[Encode] Latent mean/std:")
    # for name, latent in [
    #     ("latent_left_seq1", latent_left_seq1),
    #     ("latent_right_seq1", latent_right_seq1),
    #     ("latent_left_seq2", latent_left_seq2),
    #     ("latent_right_seq2", latent_right_seq2),
    # ]:
    #     print(f"  {name}: mean={latent.mean().item():.6f}, std={latent.std().item():.6f}")

    # --- Check sequence lengths and reorder if necessary ---
    len1 = latent_left_seq1.shape[0]
    len2 = latent_left_seq2.shape[0]
    if len1 > len2:
        # print(f"[Sequence Swap] seq1 is longer than seq2 ({len1} > {len2}), swapping sequences for DTW.")
        latent_left_seq1, latent_left_seq2 = latent_left_seq2, latent_left_seq1
        latent_right_seq1, latent_right_seq2 = latent_right_seq2, latent_right_seq1
        left_wrist_seq1, left_wrist_seq2 = left_wrist_seq2, left_wrist_seq1
        right_wrist_seq1, right_wrist_seq2 = right_wrist_seq2, right_wrist_seq1

    

    # DTW
    dtw_distance, index_start = dtw_subsequence_full_alignment(
        latent_left_seq1, latent_right_seq1,
        latent_left_seq2, latent_right_seq2,
        left_wrist_seq1, right_wrist_seq1,
        left_wrist_seq2, right_wrist_seq2,
        alpha_wrist=alpha_wrist,
        device=str(device)
    )

    # print(f"[Final] DTW distance = {dtw_distance:.6f}, best start index in seq2 = {index_start}")
    return dtw_distance
