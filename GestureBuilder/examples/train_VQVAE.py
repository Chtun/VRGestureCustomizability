import torch
from pathlib import Path
import sys
import datetime


# Import your dataset and trainer
from GestureBuilder.model import VQVAE, HandDataset, VQVAETrainer
from GestureBuilder.utilities.naming_conventions import get_hand_joint_list
from GestureBuilder.utilities.logger import Logger

if __name__ == "__main__":
    # === Settings ===
    data_folder = Path("C:\\Users\\chtun\\AppData\\LocalLow\\DefaultCompany\\Unity_VR_Template")
    train_path = data_folder / "train_with_augs.csv"
    val_path = data_folder / "test.csv"
    output_folder = Path("../output")
    output_folder.mkdir(parents=True, exist_ok=True)

    weights_output_path = output_folder / "vqvae_hand_model.pt"
    logs_output_folder = output_folder / "logs"
    logs_output_folder.mkdir(parents=True, exist_ok=True)

    # --- Configurations ---
    BATCH_SIZE = 4096
    LR = 1e-5
    EPOCHS = 2000
    RECON_THRESHOLD = 0.6

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # log filename with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"training_log_{timestamp}.txt"


    # --- Create logger instances ---
    stdout_logger = Logger(sys.stdout, output_folder=logs_output_folder, filename=log_filename)
    stderr_logger = Logger(sys.stderr, output_folder=logs_output_folder, filename=log_filename)

    sys.stdout = stdout_logger
    sys.stderr = stderr_logger

    # === Load datasets ===
    train_dataset = HandDataset(train_path)
    val_dataset = HandDataset(val_path)

    joints_list = get_hand_joint_list()
    num_joints = len(joints_list)

    print(f"Training examples: {len(train_dataset)}")
    print(f"Validation examples: {len(val_dataset)}")

    # === Initialize model ===
    input_dim = num_joints * 3  # num of joints × 3 coords
    model = VQVAE(input_dim=input_dim, hidden_dim=128, latent_dim=8, num_embeddings=150)

    # === Trainer ===
    trainer = VQVAETrainer(model, train_dataset, batch_size=BATCH_SIZE, lr=LR, val_dataset=val_dataset)

    # === Train ===
    trainer.train(epochs=EPOCHS, recon_threshold=RECON_THRESHOLD)

    # === Save Trained Model ===
    torch.save(model.state_dict(), weights_output_path)
    print(f"Model saved to: {weights_output_path}")

    
    stdout_logger.close()
    stderr_logger.close()