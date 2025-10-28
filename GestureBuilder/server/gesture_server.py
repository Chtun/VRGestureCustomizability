import torch
import json
from collections import deque
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pydantic import BaseModel
from typing import List, Dict
from pathlib import Path
import asyncio
from concurrent.futures import ThreadPoolExecutor
import time
import yaml

# Import your dataset and model
from GestureBuilder.model.VQ_VAE import VQVAE
from GestureBuilder.utilities.naming_conventions import get_hand_joint_list
from GestureBuilder.utilities.sequence_comparison import sequence_distance
from GestureBuilder.model.hand_dataset import joints_dict_to_tensor, load_hand_tensors_from_csv, convert_joint_to_hand_vector

# === FastAPI setup ===
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# Load YAML
config_path = "./config/config.yaml"
with open(config_path, "r") as f:
    cfg = yaml.safe_load(f)

# === Paths ===
data_folder = Path(cfg['paths']['data_folder'])
output_folder = Path(cfg['paths']['output_folder'])
input_VQVAE_model = output_folder / cfg['paths']['input_VQVAE_model']

# === Gesture template paths ===
gesture_template_paths = [
    (item['name'], data_folder / item['path'])
    for item in cfg['gesture_template_paths']
]

# === Gesture settings ===
BUFFER_MAX_LEN = cfg['gesture_settings']['BUFFER_MAX_LEN']
MATCH_THRESHOLD = cfg['gesture_settings']['MATCH_THRESHOLD']

# === Load state dict ===
state_dict = torch.load(input_VQVAE_model, map_location="cpu")
num_embeddings = state_dict['vq.embeddings'].shape[0]
print(f"Loaded num_embeddings from state dict: {num_embeddings}")

# === Model parameters ===
JOINTS_LIST = get_hand_joint_list()
num_joints = len(JOINTS_LIST)
input_dim = num_joints * 3
latent_dim = cfg['model_params']['latent_dim']
hidden_dim = cfg['model_params']['hidden_dim']
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# === Initialize and load model ===
vqvae_model = VQVAE(input_dim=input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim, num_embeddings=num_embeddings)
vqvae_model.load_state_dict(torch.load(input_VQVAE_model, map_location=DEVICE))
vqvae_model.to(DEVICE)
vqvae_model.eval()

# === Gesture templates storage: dict of lists ===
gesture_templates: Dict[str, List[Dict[str, torch.Tensor]]] = {}
num_templates = 0
MAX_NUM_TEMPLATES = 15

# === Load all gesture sequences ===
for pair in gesture_template_paths:
    gesture_key =  pair[0]
    template_path = pair[1]

    if gesture_key not in gesture_templates:
        gesture_templates[gesture_key] = []

    left_seq, right_seq, lw, rw = load_hand_tensors_from_csv(template_path, JOINTS_LIST)
    gesture_dict = {
        "left_hand_vectors": left_seq,
        "right_hand_vectors": right_seq,
        "left_wrist_root": lw,
        "right_wrist_root": rw
    }

    gesture_templates[gesture_key].append(gesture_dict)

print(gesture_templates.keys())

# === HTTP Endpoint to add a new gesture ===
class GestureInput(BaseModel):
    label: str
    right_joints: List[List[List[float]]]
    left_joints: List[List[List[float]]]
    left_wrist: List[List[float]]
    right_wrist: List[List[float]]

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    # Maintain separate rolling buffers
    buffer_left_hands = deque(maxlen=BUFFER_MAX_LEN)
    buffer_right_hands = deque(maxlen=BUFFER_MAX_LEN)
    buffer_left_wrist = deque(maxlen=BUFFER_MAX_LEN)
    buffer_right_wrist = deque(maxlen=BUFFER_MAX_LEN)

    try:
        while True:
            message = await websocket.receive_text()
            data = json.loads(message)

            # Parse per-frame data
            left_hands_frame = joints_dict_to_tensor(data["left_hand"], JOINTS_LIST)
            right_hands_frame = joints_dict_to_tensor(data["right_hand"], JOINTS_LIST)
            left_wrist_frame = torch.tensor(data["left_wrist"], dtype=torch.float32)
            right_wrist_frame = torch.tensor(data["right_wrist"], dtype=torch.float32)


            # Add to rolling buffers
            buffer_left_hands.append(left_hands_frame)
            buffer_right_hands.append(right_hands_frame)
            buffer_left_wrist.append(left_wrist_frame)
            buffer_right_wrist.append(right_wrist_frame)

            # Start comparing when enough frames are buffered
            min_buffer_length = 5
            if len(buffer_left_hands) >= min_buffer_length:
                seq_left_hands = torch.stack(list(buffer_left_hands), dim=0)
                seq_right_hands = torch.stack(list(buffer_right_hands), dim=0)
                seq_left_wrist = torch.stack(list(buffer_left_wrist), dim=0)
                seq_right_wrist = torch.stack(list(buffer_right_wrist), dim=0)

                # Flatten for input
                seq_left_hands_flat = seq_left_hands.view(seq_left_hands.shape[0], -1)   # (num_frames, num_joints * 3)
                seq_right_hands_flat = seq_right_hands.view(seq_right_hands.shape[0], -1) # (num_frames, num_joints * 3)

                loop = asyncio.get_running_loop()
                match_found = False

                start_time = time.time()  # start timer
                lowest_distance = 1000 # Store the lowest dtw distance value.

                # Use ThreadPoolExecutor for CPU-bound DTW
                with ThreadPoolExecutor() as executor:
                    # Dictionary to hold the lowest distance and match status per gesture key
                    gesture_results = {}

                    futures = []
                    template_keys = []

                    # Submit sequence_distance calls for all templates
                    for gesture_key, template_list in gesture_templates.items():
                        for template in template_list:

                            buffer_length = len(buffer_left_hands)
                            template_length = len(template["left_hand_vectors"])

                            if buffer_length >= template_length:
                                future = loop.run_in_executor(
                                    executor,
                                    sequence_distance,
                                    vqvae_model,
                                    seq_left_hands_flat,
                                    seq_right_hands_flat,
                                    seq_left_wrist,
                                    seq_right_wrist,
                                    template["left_hand_vectors"],
                                    template["right_hand_vectors"],
                                    template["left_wrist_root"],
                                    template["right_wrist_root"],
                                )
                                futures.append(future)
                                template_keys.append(gesture_key)
                            elif gesture_key not in gesture_results:
                                gesture_results[gesture_key] = (-1, False)

                    # Process results as they complete
                    for future, gesture_key in zip(futures, template_keys):
                        dtw_dist = await future

                        # Update the lowest distance for this gesture
                        if gesture_key not in gesture_results:
                            gesture_results[gesture_key] = (dtw_dist, dtw_dist < MATCH_THRESHOLD)
                        else:
                            current_lowest, matched = gesture_results[gesture_key]

                            if current_lowest < 0:
                                new_lowest = dtw_dist
                            else:
                                new_lowest = min(current_lowest, dtw_dist)
                            
                            gesture_results[gesture_key] = (new_lowest, new_lowest < MATCH_THRESHOLD)

                    # Reset buffers if any match was found
                    if any(matched for _, matched in gesture_results.values()):
                        buffer_left_hands.clear()
                        buffer_right_hands.clear()
                        buffer_left_wrist.clear()
                        buffer_right_wrist.clear()

                    # Send the full gesture results dictionary to the frontend
                    await websocket.send_json({
                        "gesture_results": {k: v for k, v in gesture_results.items()}
                    })

                end_time = time.time()  # end timer
                # print(f"Time to process all templates: {end_time - start_time:.4f} seconds")


    except WebSocketDisconnect:
        print("Client disconnected.")
        # Nothing else to do, the socket is already closed
    except Exception as e:
        print("WebSocket error:", e)
        # Only attempt to close if it's still open
        if websocket.client_state != "DISCONNECTED":
            await websocket.close()

add_gesture_statuses = {
    "OK": 0,
    "Max Templates Hit": 1,
    "Too Similar To Other": 2,
    "Too Different From Group": 3,
    "Internal Error": -1
}

@app.post("/add_gesture")
async def add_gesture(gesture: GestureInput):
    try:
        # === Check that there are fewer than the max number of templates ===
        if num_templates >= MAX_NUM_TEMPLATES:
            return {
            "status_code": add_gesture_statuses["Max Templates Hit"],
            "message": f"The maximum number of templates in the system has already been hit. Delete one to proceed.",
            }

        # === Validate sequence lengths ===
        seq_len_left_joints = len(gesture.left_joints)
        seq_len_right_joints = len(gesture.right_joints)
        seq_len_left_wrist = len(gesture.left_wrist)
        seq_len_right_wrist = len(gesture.right_wrist)

        if not (seq_len_left_joints == seq_len_right_joints == seq_len_left_wrist == seq_len_right_wrist):
            raise ValueError(
                f"All sequences must have the same length, got: "
                f"left_joints={seq_len_left_joints}, right_joints={seq_len_right_joints}, "
                f"left_wrist={seq_len_left_wrist}, right_wrist={seq_len_right_wrist}"
            )

        # === Convert to tensors ===
        left_joints = torch.tensor(gesture.left_joints, dtype=torch.float32)
        right_joints = torch.tensor(gesture.right_joints, dtype=torch.float32)
        left_wrist_root = torch.tensor(gesture.left_wrist, dtype=torch.float32)
        right_wrist_root = torch.tensor(gesture.right_wrist, dtype=torch.float32)

        # === Validate per-frame structure ===
        if left_joints.ndim != 3 or left_joints.shape[-1] != 3:
            raise ValueError(f"left_joints must have shape (B, 24, 3), got {left_joints.shape}")
        if right_joints.ndim != 3 or right_joints.shape[-1] != 3:
            raise ValueError(f"right_joints must have shape (B, 24, 3), got {right_joints.shape}")

        # === Convert joints to hand vectors ===
        left_hand_vectors = convert_joint_to_hand_vector(left_joints)
        right_hand_vectors = convert_joint_to_hand_vector(right_joints)

        # === Store the gesture ===
        gesture_dict = {
            "left_hand_vectors": left_hand_vectors,
            "right_hand_vectors": right_hand_vectors,
            "left_wrist_root": left_wrist_root,
            "right_wrist_root": right_wrist_root,
        }

        if gesture.label not in gesture_templates:
            gesture_templates[gesture.label] = []


        for gesture_key in gesture_templates.keys():
            # Check that the input gesture is not too close to gestures in other groups.
            if gesture_key != gesture.label:
                for template in gesture_templates[gesture_key]:
                    if sequence_distance(
                        vqvae_model,
                        gesture_dict["left_hand_vectors"],
                        gesture_dict["right_hand_vectors"],
                        gesture_dict["left_wrist_root"],
                        gesture_dict["right_wrist_root"],
                        template["left_hand_vectors"],
                        template["right_hand_vectors"],
                        template["left_wrist_root"],
                        template["right_wrist_root"]
                    ) < 2 * MATCH_THRESHOLD:
                        return {
                            "status_code": add_gesture_statuses["Too Similar To Other"],
                            "message": f"Gesture '{gesture.label}' is too similar to another gesture '{gesture_key}' in the system.",
                        }
            # Check that the input gesture is not close to gestures in its group.
            else:
                for template in gesture_templates[gesture_key]:
                    if sequence_distance(
                        vqvae_model,
                        gesture_dict["left_hand_vectors"],
                        gesture_dict["right_hand_vectors"],
                        gesture_dict["left_wrist_root"],
                        gesture_dict["right_wrist_root"],
                        template["left_hand_vectors"],
                        template["right_hand_vectors"],
                        template["left_wrist_root"],
                        template["right_wrist_root"]
                    ) > 3 * MATCH_THRESHOLD:
                        return {
                            "status_code": add_gesture_statuses["Too Different From Group"],
                            "message": f"Gesture '{gesture.label}' was too far away from another template in its group.",
                        }



        gesture_templates[gesture.label].append(gesture_dict)

        return {
            "status_code": add_gesture_statuses["OK"],
            "message": f"Gesture '{gesture.label}' added.",
        }

    except Exception as e:
        return {
            "status_code": add_gesture_statuses["Internal Error"],
            "message": f"Internal Error: {str(e)}"
        }


@app.get("/get_gestures")
async def get_gestures():
    serializable_gestures = {}

    def tensor_to_list(tensor):
        # Convert PyTorch tensor to nested lists
        return tensor.cpu().tolist()

    for gesture_key, templates in gesture_templates.items():
        serializable_gestures[gesture_key] = []

        for template in templates:
            serializable_template = {
                "left_hand_vectors": tensor_to_list(template["left_hand_vectors"]),
                "right_hand_vectors": tensor_to_list(template["right_hand_vectors"]),
                "left_wrist_root": tensor_to_list(template["left_wrist_root"]),
                "right_wrist_root": tensor_to_list(template["right_wrist_root"]),
            }
            serializable_gestures[gesture_key].append(serializable_template)

    return serializable_gestures



if __name__ == "__main__":
    uvicorn.run("gesture_server:app", host="127.0.0.1", port=8000, reload=True)
