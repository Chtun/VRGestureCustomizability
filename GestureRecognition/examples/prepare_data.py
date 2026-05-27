import pandas as pd
import numpy as np
import glob
import os
from pathlib import Path
from sklearn.model_selection import train_test_split


def load_hand_data_from_folder(folder_path, align_timestamps=True):
    """
    Loads multiple CSVs of hand tracking data into a dictionary or combined DataFrame,
    excluding columns with '_rotX', '_rotY', '_rotZ', or '_rotW'.

    Args:
        folder_path (str): Path to folder containing CSVs.
        align_timestamps (bool): If True, normalizes timestamps to start at 0 for each recording.

    Returns:
        dict[str, pd.DataFrame]: Dictionary mapping filename -> DataFrame
        pd.DataFrame: Concatenated DataFrame (with 'source_file' column)
    """

    all_data = {}
    combined = []

    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
    if not csv_files:
        print(f"No CSV files found in {folder_path}")
        return None, None

    for csv_path in csv_files:
        print(f"Loading {os.path.basename(csv_path)} ...")
        try:
            df = pd.read_csv(csv_path, encoding='utf-8-sig')

            if "Timestamp" not in df.columns:
                raise ValueError(f"Missing 'Timestamp' in {csv_path}")

            if align_timestamps:
                start_time = df["Timestamp"].iloc[0]
                df["Timestamp"] -= start_time

            # Drop any columns that contain rotation components
            rotation_cols = [c for c in df.columns if any(rot in c for rot in ["_rotX", "_rotY", "_rotZ", "_rotW"])]
            if rotation_cols:
                df = df.drop(columns=rotation_cols)

            # Drop any columns for WristRoot or End
            excluded_joint_cols = [c for c in df.columns if any(joint in c for joint in ["WristRoot", "End", "Start", "Palm"])]
            if excluded_joint_cols:
                df = df.drop(columns=excluded_joint_cols)

            df["source_file"] = os.path.basename(csv_path)
            all_data[os.path.basename(csv_path)] = df
            combined.append(df)

        except Exception as e:
            print(f"Error loading {csv_path}: {e}")

    combined_df = pd.concat(combined, ignore_index=True)
    print(f"Loaded {len(csv_files)} files, total {len(combined_df)} frames (rotation columns excluded)")

    return all_data, combined_df

def augment_hand_data(hand_data, n_augmentations=3):
    """
    Augments normalized hand pose data with Gaussian noise and optional left/right mirroring.

    Args:
        df (pd.DataFrame): Original dataset (already cleaned and normalized).
        n_augmentations (int): Number of noisy copies to generate per frame.
        mirror_hands (bool): Whether to include mirrored (left/right swapped) copies.

    Returns:
        pd.DataFrame: Augmented dataset (includes original + augmentations).
    """
    augmented_data = [hand_data]

    # Identify all joint coordinate columns
    joint_cols = [c for c in hand_data.columns if any(k in c for k in ["_posX", "_posY", "_posZ"])]

    # --- Gaussian noise augmentation ---
    for i in range(n_augmentations):
        aug_df = hand_data.copy()

        # Add small Gaussian noise (2% variation)
        noise_scale = 0.003
        noise = np.random.normal(0, noise_scale, size=aug_df[joint_cols].shape)
        aug_df[joint_cols] += noise

        aug_df["augmentation_id"] = f"noise_{i+1}"
        augmented_data.append(aug_df)

    # --- Combine all ---
    full_data_with_augmentations = pd.concat(augmented_data, ignore_index=True)
    print(f"Original: {len(hand_data)} frames | Augmented: {len(full_data_with_augmentations)} frames")

    return full_data_with_augmentations


if __name__ == "__main__":
    # Example usage
    data_recordings_folder = Path("C:\\Users\\chtun\\AppData\\LocalLow\\DefaultCompany\\Unity_VR_Template")
    real_data_output_path = data_recordings_folder / "real_dataset.csv"

    train_data_output_path = data_recordings_folder / "train.csv"
    test_data_output_path = data_recordings_folder / "test.csv"
    full_train_output_path = data_recordings_folder / "train_with_augs.csv"
    full_test_output_path = data_recordings_folder / "test_with_augs.csv"

    num_augmentations = 3

    all_data, combined_df = load_hand_data_from_folder(data_recordings_folder)

    if combined_df is not None and all_data is not None:
        print("\nFiles loaded:", list(all_data.keys()))
        print("\nCombined shape:", combined_df.shape)
        print("\nColumns:", combined_df.columns.tolist())
        
        combined_df.to_csv(real_data_output_path, index=False)

        # Split combined_df into 80% train and 20% test
        train_df, test_df = train_test_split(combined_df, test_size=0.2, random_state=42, shuffle=True)

        # Save train and test to CSV
        train_df.to_csv(train_data_output_path, index=False)
        test_df.to_csv(test_data_output_path, index=False)

        # Also add augmented examples to dataset, then save full dataset to CSV
        train_df_with_augmentations = augment_hand_data(train_df, n_augmentations=num_augmentations)
        train_df_with_augmentations.to_csv(full_train_output_path, index=False)