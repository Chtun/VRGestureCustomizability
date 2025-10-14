import torch
import json
from collections import deque
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pydantic import BaseModel
from typing import List, Dict
from pathlib import Path

# Import your dataset and model
from GestureBuilder.model.VQ_VAE import VQVAE
from GestureBuilder.utilities.naming_conventions import get_hand_joint_list
from GestureBuilder.utilities.sequence_comparison import sequence_distance
from GestureBuilder.model.hand_dataset import joints_dict_to_tensor

# === FastAPI setup ===
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# === Paths ===
data_folder = Path("C:\\Users\\chtun\\AppData\\LocalLow\\DefaultCompany\\Unity_VR_Template")
output_folder = Path("../output")
input_VQVAE_model = output_folder / "vqvae_hand_model.pt"

# === Load state dict ===
state_dict = torch.load(input_VQVAE_model, map_location="cpu")
num_embeddings = state_dict['vq.embeddings'].shape[0]
print(f"Loaded num_embeddings from state dict: {num_embeddings}")

# === Model parameters ===
JOINTS_LIST = get_hand_joint_list()
num_joints = len(JOINTS_LIST)
input_dim = num_joints * 3
latent_dim = 8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# === Initialize and load model ===
vqvae_model = VQVAE(input_dim=input_dim, hidden_dim=128, latent_dim=latent_dim, num_embeddings=num_embeddings)
vqvae_model.load_state_dict(torch.load(input_VQVAE_model, map_location=DEVICE))
vqvae_model.to(DEVICE)
vqvae_model.eval()

# === Gesture templates storage: dict of lists ===
gesture_templates: Dict[str, List[Dict[str, torch.Tensor]]] = {}

# === WebSocket settings ===
BUFFER_MAX_LEN = 50
MATCH_THRESHOLD = 0.15

# === HTTP Endpoint to add a new gesture ===
class GestureInput(BaseModel):
    label: str
    right_hand: List[List[float]]
    left_hand: List[List[float]]
    left_wrist: List[List[float]]
    right_wrist: List[List[float]]

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    # Maintain separate rolling buffers
    buffer_left_joints = deque(maxlen=BUFFER_MAX_LEN)
    buffer_right_joints = deque(maxlen=BUFFER_MAX_LEN)
    buffer_left_wrist = deque(maxlen=BUFFER_MAX_LEN)
    buffer_right_wrist = deque(maxlen=BUFFER_MAX_LEN)

    try:
        while True:
            message = await websocket.receive_text()
            data = json.loads(message)

            print(data)

            # Parse per-frame data
            left_joints_frame = joints_dict_to_tensor(data["left_hand"], JOINTS_LIST)
            right_joints_frame = joints_dict_to_tensor(data["right_hand"], JOINTS_LIST)
            left_wrist_frame = torch.tensor(data["left_wrist"], dtype=torch.float32)
            right_wrist_frame = torch.tensor(data["right_wrist"], dtype=torch.float32)


            # Add to rolling buffers
            buffer_left_joints.append(left_joints_frame)
            buffer_right_joints.append(right_joints_frame)
            buffer_left_wrist.append(left_wrist_frame)
            buffer_right_wrist.append(right_wrist_frame)

            # Start comparing when enough frames are buffered
            if len(buffer_left_joints) >= 5:
                seq_left_joints = torch.stack(list(buffer_left_joints), dim=0)
                seq_right_joints = torch.stack(list(buffer_right_joints), dim=0)
                seq_left_wrist = torch.stack(list(buffer_left_wrist), dim=0)
                seq_right_wrist = torch.stack(list(buffer_right_wrist), dim=0)

                # Compare to all gesture templates
                for gesture_key, template_list in gesture_templates.items():
                    for template in template_list:
                        dtw_dist = sequence_distance(
                            vqvae_model,
                            seq_left_joints,
                            seq_right_joints,
                            seq_left_wrist,
                            seq_right_wrist,
                            template["left_joints"],
                            template["right_joints"],
                            template["left_wrist"],
                            template["right_wrist"]
                        )

                        if dtw_dist < MATCH_THRESHOLD:
                            await websocket.send_json({
                                "match": True,
                                "label": gesture_key,
                                "dtw_distance": dtw_dist
                            })
                            # Reset buffers to avoid duplicate matches
                            buffer_left_joints.clear()
                            buffer_right_joints.clear()
                            buffer_left_wrist.clear()
                            buffer_right_wrist.clear()
                            break

    except WebSocketDisconnect:
        print("Client disconnected.")
        # Nothing else to do, the socket is already closed
    except Exception as e:
        print("WebSocket error:", e)
        # Only attempt to close if it's still open
        if websocket.client_state != "DISCONNECTED":
            await websocket.close()


@app.post("/add_gesture")
async def add_gesture(gesture: GestureInput):
    try:
        joints_seq = torch.tensor([r + l for r, l in zip(gesture.right_hand, gesture.left_hand)], dtype=torch.float32)
        # Combine left + right wrist per frame
        wrist_seq = torch.tensor([lw + rw for lw, rw in zip(gesture.left_wrist, gesture.right_wrist)], dtype=torch.float32)

        example = {"joints": joints_seq, "wrist": wrist_seq}

        if gesture.label not in gesture_templates:
            gesture_templates[gesture.label] = []
        gesture_templates[gesture.label].append(example)

        return {
            "status": "ok",
            "message": f"Gesture '{gesture.label}' added",
            "total_examples": len(gesture_templates[gesture.label])
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


# === GET endpoint to list current gestures ===
@app.get("/gestures")
async def get_gestures():
    return {label: len(examples) for label, examples in gesture_templates.items()}


if __name__ == "__main__":
    uvicorn.run("gesture_server:app", host="0.0.0.0", port=8000, reload=True)
