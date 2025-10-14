import torch
import torch.nn.functional as F
import numpy as np
from ..model.VQ_VAE import VQVAE


def compute_frame_distance(
    latent_left_A, latent_right_A,
    latent_left_B, latent_right_B,
    left_wrist_A, right_wrist_A,
    left_wrist_B, right_wrist_B,
    alpha_latent: float = 0.7,
    alpha_wrist: float = 0.5,
    LATENT_MAX: float = 4.5
):
    """
    Compute frame-wise distance between two hands' latent embeddings + wrist relative vectors.
    """
    # --------------------------
    # Latent distances per hand (normalized)
    # --------------------------
    latent_dist_left = torch.norm(latent_left_A - latent_left_B, p=2) / LATENT_MAX
    latent_dist_right = torch.norm(latent_right_A - latent_right_B, p=2) / LATENT_MAX
    latent_dist = alpha_latent * latent_dist_left + (1 - alpha_latent) * latent_dist_right

    # --------------------------
    # Wrist relative vector (right -> left) distances
    # --------------------------
    vec_A = left_wrist_A - right_wrist_A
    vec_B = left_wrist_B - right_wrist_B
    wrist_dist = torch.norm(vec_A - vec_B, p=2).item()  # L2 distance between vectors

    # Combine latent + wrist distances
    combined = alpha_wrist * latent_dist + (1 - alpha_wrist) * wrist_dist

    # Debug prints
    print(f"\nFrame Distance Debug:")
    print(f"  latent_left_dist = {latent_dist_left.item():.6f}, latent_right_dist = {latent_dist_right.item():.6f}")
    print(f"  latent_combined = {latent_dist.item():.6f}")
    print(f"  wrist_vec_A = {vec_A.cpu().numpy()}, wrist_vec_B = {vec_B.cpu().numpy()}, wrist_dist = {wrist_dist:.6f}")
    print(f"  final_combined_distance = {combined:.6f}")

    return combined


def dtw_subsequence_full_alignment(
    latent_left_seq1: torch.Tensor,
    latent_right_seq1: torch.Tensor,
    latent_left_seq2: torch.Tensor,
    latent_right_seq2: torch.Tensor,
    left_wrist_seq1: torch.Tensor,
    right_wrist_seq1: torch.Tensor,
    left_wrist_seq2: torch.Tensor,
    right_wrist_seq2: torch.Tensor,
    alpha_latent: float = 0.7,
    alpha_wrist: float = 0.7
):
    """
    DTW distance for a subsequence match where seq1 must fully fit inside seq2.
    Computes the best alignment of seq1 inside seq2 using both hands' latent vectors
    and wrist-relative motion. Returns the normalized DTW cost and best start index in seq2.

    Args:
        latent_left_seq1, latent_right_seq1: (T1, latent_dim)
        latent_left_seq2, latent_right_seq2: (T2, latent_dim)
        left_wrist_seq1, right_wrist_seq1: (T1, 3)
        left_wrist_seq2, right_wrist_seq2: (T2, 3)
        alpha_latent: weight for latent distance
        alpha_wrist: weight for wrist distance

    Returns:
        dtw_dist_norm: normalized DTW distance
        best_start: index in seq2 where seq1 best aligns
    """
    T1 = latent_left_seq1.shape[0]
    T2 = latent_left_seq2.shape[0]
    print(f"\n[DTW] Sequence lengths: seq1={T1}, seq2={T2}")

    if T1 > T2:
        raise ValueError("seq1 must be shorter than or equal to seq2 for full alignment.")

    dtw_costs = []
    for start in range(T2 - T1 + 1):
        cost = 0.0
        for i in range(T1):

            # print(f"Computing the frame distance for start: {start} and i: {i}")

            frame_dist = compute_frame_distance(
                latent_left_seq1[i], latent_right_seq1[i],
                latent_left_seq2[start + i], latent_right_seq2[start + i],
                left_wrist_seq1[i], right_wrist_seq1[i],
                left_wrist_seq2[start + i], right_wrist_seq2[start + i],
                alpha_latent=alpha_latent,
                alpha_wrist=alpha_wrist
            )
            cost += frame_dist

        dtw_costs.append(cost / T1)

    dtw_dist_norm = min(dtw_costs)
    best_start = dtw_costs.index(dtw_dist_norm)
    print(f"[DTW] Costs for each valid start: {dtw_costs}")
    print(f"[DTW] Best start index in seq2: {best_start}, normalized distance: {dtw_dist_norm:.6f}")

    return dtw_dist_norm, best_start



def sequence_distance(
    vqvae_model: VQVAE,
    left_joints_seq1: torch.Tensor,
    right_joints_seq1: torch.Tensor,
    left_joints_seq2: torch.Tensor,
    right_joints_seq2: torch.Tensor,
    left_wrist_seq1: torch.Tensor,
    right_wrist_seq1: torch.Tensor,
    left_wrist_seq2: torch.Tensor,
    right_wrist_seq2: torch.Tensor,
    alpha_latent: float = 0.7,
    alpha_wrist: float = 0.7,
    device: str = "cpu"
):
    """
    Compute DTW distance between two gesture sequences using both hands.
    Automatically ensures the shorter sequence is treated as seq1 for full subsequence alignment.
    """
    vqvae_model.eval()

    encoder_input_size = vqvae_model.encoder.net[0].in_features

    print(f"\n=== Sequence Distance Debug ===")
    print(f"Input sizes: {encoder_input_size}-dim encoder, device={device}")
    print(f"Left seq1 shape: {tuple(left_joints_seq1.shape)}, Right seq1: {tuple(right_joints_seq1.shape)}")
    print(f"Left seq2 shape: {tuple(left_joints_seq2.shape)}, Right seq2: {tuple(right_joints_seq2.shape)}")

    # Move to device
    left_joints_seq1 = left_joints_seq1.to(device)
    right_joints_seq1 = right_joints_seq1.to(device)
    left_joints_seq2 = left_joints_seq2.to(device)
    right_joints_seq2 = right_joints_seq2.to(device)
    left_wrist_seq1 = left_wrist_seq1.to(device)
    right_wrist_seq1 = right_wrist_seq1.to(device)
    left_wrist_seq2 = left_wrist_seq2.to(device)
    right_wrist_seq2 = right_wrist_seq2.to(device)

    # Encode
    with torch.no_grad():
        latent_left_seq1 = vqvae_model.encode(left_joints_seq1)
        latent_right_seq1 = vqvae_model.encode(right_joints_seq1)
        latent_left_seq2 = vqvae_model.encode(left_joints_seq2)
        latent_right_seq2 = vqvae_model.encode(right_joints_seq2)

    print(f"[Encode] Latent mean/std:")
    for name, latent in [
        ("latent_left_seq1", latent_left_seq1),
        ("latent_right_seq1", latent_right_seq1),
        ("latent_left_seq2", latent_left_seq2),
        ("latent_right_seq2", latent_right_seq2),
    ]:
        print(f"  {name}: mean={latent.mean().item():.6f}, std={latent.std().item():.6f}")

    # --- Check sequence lengths and reorder if necessary ---
    len1 = latent_left_seq1.shape[0]
    len2 = latent_left_seq2.shape[0]
    if len1 > len2:
        print(f"[Sequence Swap] seq1 is longer than seq2 ({len1} > {len2}), swapping sequences for DTW.")
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
        alpha_latent=alpha_latent,
        alpha_wrist=alpha_wrist
    )

    print(f"[Final] DTW distance = {dtw_distance:.6f}, best start index in seq2 = {index_start}")
    return dtw_distance
