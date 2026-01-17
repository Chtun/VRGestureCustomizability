import pandas as pd
import plotly.graph_objects as go
from pathlib import Path

# List of file subnames
file_subnames = ["TEST-Cast Fireball 1", "TEST-Cast Fireball 2", "TEST-Cast Fireball 3"]

# Path to output folder
output_folder = Path("..\\output")
output_file = output_folder / "hand_animation-Cast Fireball-multiple.html"

# Colors for each file
colors_left = ["blue", "green", "purple"] # "green", "purple", "grey"
colors_right = ["red", "orange", "pink"] # "orange", "pink", "black"

# Scale factor
scale_factor = 5.0

# Prefixes
left_prefix = "L_"
right_prefix = "R_"

# Store frames for all files
all_left_frames = []
all_right_frames = []

# Load and process each CSV
for subname in file_subnames:
    csv_path = f"..\\server\\database\\raw_data\\live_recordings-{subname}.csv"
    df = pd.read_csv(csv_path)

    def extract_joints(row, prefix):
        joints = []
        root_x = row[f"{prefix}Root_posX"] * scale_factor
        root_y = row[f"{prefix}Root_posY"] * scale_factor
        root_z = row[f"{prefix}Root_posZ"] * scale_factor

        joints.append((root_x, root_y, root_z))

        for col in df.columns:
            if col.startswith(prefix) and col.endswith("_posX") and "Root" not in col:
                joint_name = col[len(prefix):-5]
                x = (row[f"{prefix}{joint_name}_posX"]) * scale_factor + root_x
                y = (row[f"{prefix}{joint_name}_posY"]) * scale_factor + root_y
                z = (row[f"{prefix}{joint_name}_posZ"]) * scale_factor + root_z
                joints.append((x, y, z))
        return joints

    left_frames = [extract_joints(df.iloc[i], left_prefix) for i in range(len(df))]
    right_frames = [extract_joints(df.iloc[i], right_prefix) for i in range(len(df))]

    all_left_frames.append(left_frames)
    all_right_frames.append(right_frames)

# Get max number of frames across all files
max_frames = max(len(frames) for frames in all_left_frames)

def separate_xyz(joint_list):
    if len(joint_list) == 0:
        return [], [], []
    x, y, z = zip(*joint_list)
    return list(x), list(y), list(z)

# Configuration for the "ghost" points
trail_length = 10  # How many future frames to show
trail_opacity = 0.2

# Initial data (must match the structure of frame_data)
data = []
for f_idx in range(len(all_left_frames)):
    # Current Points Traces (Indices 0, 1 for file 1; 4, 5 for file 2, etc.)
    data.append(go.Scatter3d(x=[], y=[], z=[], mode='markers', 
                             marker=dict(color=colors_left[f_idx], size=4), name=f'L Current {f_idx+1}'))
    data.append(go.Scatter3d(x=[], y=[], z=[], mode='markers', 
                             marker=dict(color=colors_right[f_idx], size=4), name=f'R Current {f_idx+1}'))
    
    # Future/Trail Traces (Indices 2, 3 for file 1; 6, 7 for file 2, etc.)
    data.append(go.Scatter3d(x=[], y=[], z=[], mode='markers', 
                             marker=dict(color=colors_left[f_idx], size=2, opacity=trail_opacity), name=f'L Trail {f_idx+1}'))
    data.append(go.Scatter3d(x=[], y=[], z=[], mode='markers', 
                             marker=dict(color=colors_right[f_idx], size=2, opacity=trail_opacity), name=f'R Trail {f_idx+1}'))

fig = go.Figure(data=data)

# Animation frames
frames = []
for i in range(max_frames):
    frame_data = []
    for f_idx, (left_frames, right_frames) in enumerate(zip(all_left_frames, all_right_frames)):
        # --- 1. Current Frame Points ---
        curr_idx = min(i, len(left_frames) - 1)
        lx, ly, lz = separate_xyz(left_frames[curr_idx])
        rx, ry, rz = separate_xyz(right_frames[curr_idx])
        
        frame_data.append(go.Scatter3d(x=lx, y=ly, z=lz, mode='markers',
                                       marker=dict(color=colors_left[f_idx], size=4)))
        frame_data.append(go.Scatter3d(x=rx, y=ry, z=rz, mode='markers',
                                       marker=dict(color=colors_right[f_idx], size=4)))

        # --- 2. Future Trail Points ---
        # Get next N frames, flatten them into a single list of coordinates
        future_idx_end = min(i + trail_length, len(left_frames))
        
        # Collect all points from i+1 to i+trail_length
        l_trail_pts = [pt for f in left_frames[i+1 : future_idx_end] for pt in f]
        r_trail_pts = [pt for f in right_frames[i+1 : future_idx_end] for pt in f]
        
        tx, ty, tz = separate_xyz(l_trail_pts)
        rtx, rty, rtz = separate_xyz(r_trail_pts)

        frame_data.append(go.Scatter3d(x=tx, y=ty, z=tz, mode='markers',
                                       marker=dict(color=colors_left[f_idx], size=2, opacity=trail_opacity)))
        frame_data.append(go.Scatter3d(x=rtx, y=rty, z=rtz, mode='markers',
                                       marker=dict(color=colors_right[f_idx], size=2, opacity=trail_opacity)))

    frames.append(go.Frame(data=frame_data, name=str(i)))

fig.frames = frames

# Layout
fig.update_layout(
    scene=dict(
        xaxis=dict(range=[-5, 5]),
        yaxis=dict(range=[-5, 5]),
        zaxis=dict(range=[-5, 5]),
        aspectmode='cube'
    ),
    updatemenus=[dict(
        type="buttons",
        buttons=[dict(label="Play",
                      method="animate",
                      args=[None, {"frame": {"duration": 100, "redraw": True},
                                   "fromcurrent": True,
                                   "mode": "immediate"}])]
    )]
)

# Save interactive HTML
fig.write_html(output_file, include_plotlyjs='cdn')
print(f"Saved animation to {output_file}")
