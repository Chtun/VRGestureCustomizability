import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
import yaml

# --- CONFIGURATION ---
FPS = 10  # Adjust this to match your recording frame rate
# ---------------------

try:
    config_file = Path(__file__).resolve().parents[1] / "server" / "config" / "config.yaml"
    cfg = yaml.safe_load(config_file.read_text())
    templates = cfg.get("gesture_template_paths", []) if isinstance(cfg, dict) else []
    data_folder = Path(cfg.get("paths", Path()).get("data_folder", Path())) if isinstance(cfg, dict) else Path()
    gesture_template_json = cfg.get("paths", {}).get("gesture_template_json", "") if isinstance(cfg, dict) else ""
    MATCH_THRESHOLD = cfg.get("gesture_settings", {}).get("MATCH_THRESHOLD", 1.5) if isinstance(cfg, dict) else 1.5
except Exception:
    templates = []
    # Simplified error for local testing
    print("Warning: Config not found, using empty templates.")

csv_names = [t.get("name", Path(t.get("path", "")).stem) for t in templates]
csv_paths = [data_folder / t.get("path") for t in templates]

output_folder = Path("..\\output")
output_file = output_folder / "hand_animation-Cast Fireball-multiple.html"
colors_left = ["blue", "green", "purple"]
colors_right = ["red", "orange", "pink"]
scale_factor = 5.0
left_prefix = "L_"
right_prefix = "R_"

all_left_frames = []
all_right_frames = []

for csv_path in csv_paths:
    if not csv_path.exists(): continue
    df = pd.read_csv(csv_path, skip_blank_lines=False)

    def extract_joints(row, prefix):
        # Check if any of the root position columns for this hand are NaN
        # If the root is missing, we consider the hand/row invalid
        if pd.isna(row[f"{prefix}Root_posX"]):
            return None

        joints = []
        root_x = row[f"{prefix}Root_posX"] * scale_factor
        root_y = row[f"{prefix}Root_posY"] * scale_factor
        root_z = row[f"{prefix}Root_posZ"] * scale_factor
        joints.append((root_x, root_y, root_z))

        for col in df.columns:
            if col.startswith(prefix) and col.endswith("_posX") and "Root" not in col:
                joint_name = col[len(prefix):-5]
                # If any specific joint in the hand is NaN, return None to signal "empty"
                if pd.isna(row[f"{prefix}{joint_name}_posX"]):
                    return None
                
                x = (row[f"{prefix}{joint_name}_posX"]) * scale_factor + root_x
                y = (row[f"{prefix}{joint_name}_posY"]) * scale_factor + root_y
                z = (row[f"{prefix}{joint_name}_posZ"]) * scale_factor + root_z
                joints.append((x, y, z))
        return joints

    current_file_left = []
    current_file_right = []

    for i in range(len(df)):
        left_data = extract_joints(df.iloc[i], left_prefix)
        right_data = extract_joints(df.iloc[i], right_prefix)

        # If either hand is missing data in this row, stop here
        if left_data is None or right_data is None:
            print(f"Empty data detected at row {i} in {csv_path.name}. Cutting file short.")
            break
        
        current_file_left.append(left_data)
        current_file_right.append(right_data)

    all_left_frames.append(current_file_left)
    all_right_frames.append(current_file_right)

if not all_left_frames:
    raise ValueError("No data found to visualize.")

max_frames = max(len(frames) for frames in all_left_frames)

def separate_xyz(joint_list):
    if len(joint_list) == 0: return [], [], []
    x, y, z = zip(*joint_list)
    return list(x), list(y), list(z)

trail_length = 10 
trail_opacity = 0.2

data = []
for f_idx in range(len(all_left_frames)):
    data.append(go.Scatter3d(x=[], y=[], z=[], mode='markers', marker=dict(color=colors_left[f_idx], size=4), name=f'L Current {f_idx+1}'))
    data.append(go.Scatter3d(x=[], y=[], z=[], mode='markers', marker=dict(color=colors_right[f_idx], size=4), name=f'R Current {f_idx+1}'))
    data.append(go.Scatter3d(x=[], y=[], z=[], mode='markers', marker=dict(color=colors_left[f_idx], size=2, opacity=trail_opacity), name=f'L Trail {f_idx+1}'))
    data.append(go.Scatter3d(x=[], y=[], z=[], mode='markers', marker=dict(color=colors_right[f_idx], size=2, opacity=trail_opacity), name=f'R Trail {f_idx+1}'))

fig = go.Figure(data=data)

# --- ANIMATION FRAMES ---
frames = []
for i in range(max_frames):
    frame_data = []
    # Calculate time based on frame index
    current_time = i / FPS 
    
    for f_idx, (left_f_list, right_f_list) in enumerate(zip(all_left_frames, all_right_frames)):
        curr_idx = min(i, len(left_f_list) - 1)
        lx, ly, lz = separate_xyz(left_f_list[curr_idx])
        rx, ry, rz = separate_xyz(right_f_list[curr_idx])
        
        frame_data.append(go.Scatter3d(x=lx, y=ly, z=lz, mode='markers', marker=dict(color=colors_left[f_idx], size=4)))
        frame_data.append(go.Scatter3d(x=rx, y=ry, z=rz, mode='markers', marker=dict(color=colors_right[f_idx], size=4)))

        future_idx_end = min(i + trail_length, len(left_f_list))
        l_trail_pts = [pt for f in left_f_list[i+1 : future_idx_end] for pt in f]
        r_trail_pts = [pt for f in right_f_list[i+1 : future_idx_end] for pt in f]
        
        tx, ty, tz = separate_xyz(l_trail_pts)
        rtx, rty, rtz = separate_xyz(r_trail_pts)

        frame_data.append(go.Scatter3d(x=tx, y=ty, z=tz, mode='markers', marker=dict(color=colors_left[f_idx], size=2, opacity=trail_opacity)))
        frame_data.append(go.Scatter3d(x=rtx, y=rty, z=rtz, mode='markers', marker=dict(color=colors_right[f_idx], size=2, opacity=trail_opacity)))

    # Add frame with layout update for the timestamp annotation
    frames.append(go.Frame(
        data=frame_data, 
        name=str(i),
        layout=go.Layout(annotations=[dict(
            text=f"Time: {current_time:.2f}s (Frame {i})",
            showarrow=False,
            xref="paper", yref="paper",
            x=0.05, y=0.95,
            font=dict(size=18, color="black"),
            bgcolor="white", opacity=0.8
        )])
    ))

fig.frames = frames

# --- LAYOUT ---
fig.update_layout(
    scene=dict(
        xaxis=dict(range=[-5, 5]),
        yaxis=dict(range=[-5, 5]),
        zaxis=dict(range=[-5, 5]),
        aspectmode='cube'
    ),
    # Add an initial annotation so it's visible before clicking 'Play'
    annotations=[dict(
        text=f"Time: 0.00s (Frame 0)",
        showarrow=False,
        xref="paper", yref="paper",
        x=0.05, y=0.95,
        font=dict(size=18, color="black"),
        bgcolor="white", opacity=0.8
    )],
    updatemenus=[dict(
        type="buttons",
        buttons=[dict(label="Play",
                      method="animate",
                      args=[None, {"frame": {"duration": 100, "redraw": True},
                                   "fromcurrent": True,
                                   "mode": "immediate"}])]
    )]
)

output_folder.mkdir(parents=True, exist_ok=True)
fig.write_html(output_file, include_plotlyjs='cdn')
print(f"Saved animation to {output_file}")