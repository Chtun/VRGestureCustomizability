import torch
import json
from collections import deque
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query
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
from GestureBuilder.utilities.hand_computations import compute_parent_relative_rotations
from GestureBuilder.model.hand_dataset import joints_dict_to_tensor, load_hand_tensors_from_csv, convert_joint_to_hand_vector
from GestureBuilder.utilities.file_operations import load_gestures_from_json, save_gestures_to_json

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
if not cfg['gesture_template_paths'] is None:
    gesture_template_paths = [
        (item['name'], data_folder / item['path'])
        for item in cfg['gesture_template_paths']
    ]
else:
    gesture_template_paths = []

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

# === Default gesture templates storage: dict of lists ===
default_gesture_templates: Dict[str, List[Dict[str, torch.Tensor]]] = {}
num_templates = 0
MAX_NUM_TEMPLATES = 15

# === Load all gesture sequences ===
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
        "right_hand_vectors": right_hand_vectors, # shape: (num_frames, 72)
        "left_joint_rotations": left_joint_rotations, # shape: (num_frames, 24, 4)
        "right_joint_rotations": right_joint_rotations, # shape: (num_frames, 24, 4)
        "left_wrist_positions": left_wrist_positions, # shape: (num_frames, 3)
        "right_wrist_positions": right_wrist_positions, # shape: (num_frames, 3)
        "left_wrist_rotations": left_wrist_rotations, # shape: (num_frames, 4)
        "right_wrist_rotations": right_wrist_rotations # shape: (num_frames, 4)
    }

    default_gesture_templates[gesture_key].append(gesture_dict)

print(f"Default gesture keys: {default_gesture_templates.keys()}")

# === HTTP Endpoint to add a new gesture ===
class GestureInput(BaseModel):
    label: str
    left_joint_positions: List[List[List[float]]] # shape: (num_frames, 24, 3)
    right_joint_positions: List[List[List[float]]] # shape: (num_frames, 24, 3)
    left_joint_rotations: List[List[List[float]]] # shape: (num_frames, 24, 4)
    right_joint_rotations: List[List[List[float]]] # shape: (num_frames, 24, 4)
    left_wrist_positions: List[List[float]] # shape: (num_frames, 3)
    right_wrist_positions: List[List[float]] # shape: (num_frames, 3)
    left_wrist_rotations: List[List[float]] # shape: (num_frames, 4)
    right_wrist_rotations: List[List[float]] # shape: (num_frames, 4)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    global default_gesture_templates
    await websocket.accept()

    gesture_templates: Dict[str, List[Dict[str, torch.Tensor]]] = load_gestures_from_json(cfg['paths']['gesture_template_json'])

    # Maintain separate rolling buffers
    buffer_left_hands = deque(maxlen=BUFFER_MAX_LEN)
    buffer_right_hands = deque(maxlen=BUFFER_MAX_LEN)
    buffer_left_wrist = deque(maxlen=BUFFER_MAX_LEN)
    buffer_right_wrist = deque(maxlen=BUFFER_MAX_LEN)

    try:

        message = await websocket.receive_text()
        data = json.loads(message)

        if data.get("type") == "init":
            use_default_system = data.get("useDefaultSystem", True)
            print(f"Client using default system? {use_default_system}")
        else:
            # Send error message
            await websocket.send_json({
                "error": "Expected 'init' message first."
            })
            # Close the connection
            await websocket.close(code=1003)  # 1003 = unsupported data
            return
        
        if use_default_system:
            print(default_gesture_templates.keys())
        else:
            print(gesture_templates.keys())

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
            min_buffer_length = 20
            if len(buffer_left_hands) >= min_buffer_length:
                seq_left_hands = torch.stack(list(buffer_left_hands), dim=0)
                seq_right_hands = torch.stack(list(buffer_right_hands), dim=0)
                seq_left_wrist = torch.stack(list(buffer_left_wrist), dim=0)
                seq_right_wrist = torch.stack(list(buffer_right_wrist), dim=0)

                # Flatten for input
                seq_left_hands_flat = seq_left_hands.view(seq_left_hands.shape[0], -1)   # (num_frames, num_joints * 3)
                seq_right_hands_flat = seq_right_hands.view(seq_right_hands.shape[0], -1) # (num_frames, num_joints * 3)

                loop = asyncio.get_running_loop()

                start_time = time.time()  # start timer

                # Use ThreadPoolExecutor for CPU-bound DTW
                with ThreadPoolExecutor() as executor:
                    # Dictionary to hold the lowest distance and match status per gesture key
                    gesture_results = {}

                    futures = []
                    template_keys = []

                    if use_default_system:
                        # Submit sequence_distance calls for all default templates
                        for gesture_key, template_list in default_gesture_templates.items():
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
                                        template["left_wrist_positions"],
                                        template["right_wrist_positions"],
                                    )
                                    futures.append(future)
                                    template_keys.append(gesture_key)
                                elif gesture_key not in gesture_results:
                                    gesture_results[gesture_key] = (-1, False)
                    else:
                        # Submit sequence_distance calls for all user-defined templates
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
                                        template["left_wrist_positions"],
                                        template["right_wrist_positions"],
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
    global num_templates, MAX_NUM_TEMPLATES
    try:
        gesture_templates: Dict[str, List[Dict[str, torch.Tensor]]] = load_gestures_from_json(cfg['paths']['gesture_template_json'])

        # === Check that there are fewer than the max number of templates ===
        if num_templates >= MAX_NUM_TEMPLATES:
            return {
            "status_code": add_gesture_statuses["Max Templates Hit"],
            "message": f"The maximum number of templates in the system has already been hit. Delete one to proceed.",
            }

        # === Validate sequence lengths ===
        seq_len_left_joint_positions = len(gesture.left_joint_positions)
        seq_len_right_joint_positions = len(gesture.right_joint_positions)
        seq_len_left_joint_rotations = len(gesture.left_joint_rotations)
        seq_len_right_joint_rotations = len(gesture.right_joint_rotations)
        seq_len_left_wrist_positions = len(gesture.left_wrist_positions)
        seq_len_right_wrist_positions = len(gesture.right_wrist_positions)
        seq_len_left_wrist_rotations = len(gesture.left_joint_rotations)
        seq_len_right_wrist_rotations = len(gesture.right_wrist_rotations)

        if not (seq_len_left_joint_positions == seq_len_right_joint_positions
                == seq_len_left_joint_rotations == seq_len_right_joint_rotations
                == seq_len_left_wrist_positions == seq_len_right_wrist_positions
                == seq_len_left_wrist_rotations == seq_len_right_wrist_rotations):
            raise ValueError(
                f"All sequences must have the same length, got lengths: "
                f"left_joint_positions={seq_len_left_joint_positions}, right_joint_positions={seq_len_right_joint_positions}, "
                f"left_joint_rotations={seq_len_left_joint_rotations}, right_joint_rotations={seq_len_right_joint_rotations}, "
                f"left_wrist_positions={seq_len_left_wrist_positions}, right_wrist_positions={seq_len_right_wrist_positions}, "
                f"left_wrist_rotations={seq_len_left_wrist_rotations}, right_wrist_rotations={seq_len_right_wrist_rotations}"
            )

        # === Convert to tensors ===
        left_joint_positions = torch.tensor(gesture.left_joint_positions, dtype=torch.float32)
        right_joint_positions = torch.tensor(gesture.right_joint_positions, dtype=torch.float32)
        left_joint_rotations = torch.tensor(gesture.left_joint_rotations, dtype=torch.float32)
        right_joint_rotations = torch.tensor(gesture.right_joint_rotations, dtype=torch.float32)
        left_wrist_positions = torch.tensor(gesture.left_wrist_positions, dtype=torch.float32)
        right_wrist_positions = torch.tensor(gesture.right_wrist_positions, dtype=torch.float32)
        left_wrist_rotations = torch.tensor(gesture.left_wrist_rotations, dtype=torch.float32)
        right_wrist_rotations = torch.tensor(gesture.right_wrist_rotations, dtype=torch.float32)

        # === Validate per-frame structure ===
        if left_joint_positions.ndim != 3 or left_joint_positions.shape[-1] != 3:
            raise ValueError(f"left_joint_positions must have shape (B, 24, 3), got {left_joint_positions.shape}")
        if right_joint_positions.ndim != 3 or right_joint_positions.shape[-1] != 3:
            raise ValueError(f"right_joint_positions must have shape (B, 24, 3), got {right_joint_positions.shape}")
        if left_joint_rotations.ndim != 3 or left_joint_rotations.shape[-1] != 4:
            raise ValueError(f"left_joints_rotations must have shape (B, 24, 4), got {left_joint_rotations.shape}")
        if right_joint_rotations.ndim != 3 or right_joint_rotations.shape[-1] != 4:
            raise ValueError(f"right_joints_rotations must have shape (B, 24, 3), got {right_joint_rotations.shape}")
        if left_wrist_positions.ndim != 2 or left_wrist_positions.shape[-1] != 3:
            raise ValueError(f"left_wrist_positions must have shape (B, 3), got {left_wrist_positions.shape}")
        if right_wrist_positions.ndim != 2 or right_wrist_positions.shape[-1] != 3:
            raise ValueError(f"right_wrist_positions must have shape (B, 3), got {right_wrist_positions.shape}")
        if left_wrist_rotations.ndim != 2 or left_wrist_rotations.shape[-1] != 4:
            raise ValueError(f"left_wrist_rotations must have shape (B, 4), got {left_wrist_rotations.shape}")
        if right_wrist_rotations.ndim != 2 or right_wrist_rotations.shape[-1] != 4:
            raise ValueError(f"right_wrist_rotations must have shape (B, 4), got {right_wrist_rotations.shape}")

        # === Convert joints to hand vectors ===
        left_hand_vectors = convert_joint_to_hand_vector(left_joint_positions)
        right_hand_vectors = convert_joint_to_hand_vector(right_joint_positions)

        # === Flatten to shape (num_frames, 72) ===
        left_hand_vectors = left_hand_vectors.reshape(left_hand_vectors.shape[0], -1)
        right_hand_vectors = right_hand_vectors.reshape(right_hand_vectors.shape[0], -1)

        # --- Convert left/right joint rotations to parent-relative ---
        left_joint_rotations = compute_parent_relative_rotations(left_joint_rotations)
        right_joint_rotations = compute_parent_relative_rotations(right_joint_rotations)

        # === Store the gesture ===
        gesture_dict = {
            "left_hand_vectors": left_hand_vectors, # shape: (num_frames, 72)
            "right_hand_vectors": right_hand_vectors, # shape: (num_frames, 72)
            "left_joint_rotations": left_joint_rotations, # shape: (num_frames, 24, 4)
            "right_joint_rotations": right_joint_rotations, # shape: (num_frames, 24, 4)
            "left_wrist_positions": left_wrist_positions, # shape: (num_frames, 3)
            "right_wrist_positions": right_wrist_positions, # shape: (num_frames, 3)
            "left_wrist_rotations": left_wrist_rotations, # shape: (num_frames, 4)
            "right_wrist_rotations": right_wrist_rotations, # shape: (num_frames, 4)
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
                        gesture_dict["left_wrist_positions"],
                        gesture_dict["right_wrist_positions"],
                        template["left_hand_vectors"],
                        template["right_hand_vectors"],
                        template["left_wrist_positions"],
                        template["right_wrist_positions"]
                    ) < 1.4 * MATCH_THRESHOLD:
                        message = f"Gesture '{gesture.label}' was too far away from another template in its group."
                        print(message)

                        return {
                            "status_code": add_gesture_statuses["Too Similar To Other"],
                            "message": message,
                        }
            # Check that the input gesture is not close to gestures in its group.
            else:
                for template in gesture_templates[gesture_key]:
                    if sequence_distance(
                        vqvae_model,
                        gesture_dict["left_hand_vectors"],
                        gesture_dict["right_hand_vectors"],
                        gesture_dict["left_wrist_positions"],
                        gesture_dict["right_wrist_positions"],
                        template["left_hand_vectors"],
                        template["right_hand_vectors"],
                        template["left_wrist_positions"],
                        template["right_wrist_positions"]
                    ) > 2.5 * MATCH_THRESHOLD:
                        message = f"Gesture '{gesture.label}' was too far away from another template in its group."
                        print(message)
                        return {
                            "status_code": add_gesture_statuses["Too Different From Group"],
                            "message": message,
                        }

        gesture_templates[gesture.label].append(gesture_dict)

        print(gesture_templates.keys())

        save_gestures_to_json(gesture_templates, cfg['paths']['gesture_template_json'])

        return {
            "status_code": add_gesture_statuses["OK"],
            "message": f"Gesture '{gesture.label}' added.",
        }

    except Exception as e:
        print(e)
        return {
            "status_code": add_gesture_statuses["Internal Error"],
            "message": f"Internal Error: {str(e)}"
        }

remove_gestures_statuses = {
    "OK": 0,
    "No Label Found": 1,
    "Internal Error": -1
}

@app.delete("/gesture/{gesture_label}")
async def remove_gesture(gesture_label: str):
    global num_templates
    try:
        gesture_templates: Dict[str, List[Dict[str, torch.Tensor]]] = load_gestures_from_json(cfg['paths']['gesture_template_json'])

        # === Check that the gesture label has templates ===
        if gesture_label in gesture_templates:
            num_templates_removed = len(gesture_templates[gesture_label])

            del gesture_templates[gesture_label]

            num_templates -= num_templates_removed

            save_gestures_to_json(gesture_templates, cfg['paths']['gesture_template_json'])

            return {
                "status_code": remove_gestures_statuses["OK"],
                "message": f"Removed {num_templates_removed} templates from gesture with label '{gesture_label}'"
            }
        else:
            return {
                "status_code": remove_gestures_statuses["No Label Found"],
                "message": f"No templates found for gesture with label '{gesture_label}'"
            }

    except Exception as e:
        return {
            "status_code": remove_gestures_statuses["Internal Error"],
            "message": f"Internal Error: {str(e)}"
        }

@app.delete("/gesture/")
async def remove_all_gestures():
    global num_templates
    try:


        gesture_templates = {}
        num_templates = 0
        
        save_gestures_to_json(gesture_templates, cfg['paths']['gesture_template_json'])

        return {
            "status_code": remove_gestures_statuses["OK"],
            "message": f"All gestures removed."
        }

    except Exception as e:
        return {
            "status_code": remove_gestures_statuses["Internal Error"],
            "message": f"Internal Error: {str(e)}"
        }        



@app.get("/get_gestures")
async def get_gestures(useDefaultSystem: bool = Query(False, description="Whether to use default or custom gesture templates.")):
    global default_gesture_templates

    # Load either default or custom gestures
    if useDefaultSystem:
        gesture_templates: Dict[str, List[Dict[str, torch.Tensor]]] = default_gesture_templates
    else:
        gesture_templates: Dict[str, List[Dict[str, torch.Tensor]]] = load_gestures_from_json(cfg['paths']['gesture_template_json'])

    serializable_gestures = {}

    def tensor_to_list(tensor):
        return tensor.cpu().tolist()

    for gesture_key, templates in gesture_templates.items():
        serializable_gestures[gesture_key] = []
        for template in templates:
            left_vectors_reshaped = template["left_hand_vectors"].reshape(-1, 24, 3)
            right_vectors_reshaped = template["right_hand_vectors"].reshape(-1, 24, 3)

            serializable_template = {
                "left_hand_vectors": tensor_to_list(left_vectors_reshaped),
                "right_hand_vectors": tensor_to_list(right_vectors_reshaped),
                "left_joint_rotations": tensor_to_list(template["left_joint_rotations"]),
                "right_joint_rotations": tensor_to_list(template["right_joint_rotations"]),
                "left_wrist_positions": tensor_to_list(template["left_wrist_positions"]),
                "right_wrist_positions": tensor_to_list(template["right_wrist_positions"]),
                "left_wrist_rotations": tensor_to_list(template["left_wrist_rotations"]),
                "right_wrist_rotations": tensor_to_list(template["right_wrist_rotations"]),
            }

            serializable_gestures[gesture_key].append(serializable_template)

    return serializable_gestures



if __name__ == "__main__":
    uvicorn.run("gesture_server:app", host="127.0.0.1", port=8000, reload=True)
