import pandas as pd
import plotly.graph_objects as go
from pathlib import Path

# List of file subnames
file_subnames = ["right_jab-Evan-1"] #"right_jab-1", "right_jab-2", "right_jab-3", ]

# Path to output folder
output_folder = Path("..\\output")
output_file = output_folder / "hand_animation-Evan.html"

# Colors for each file
colors_left = ["blue"] #, "green", "purple", "grey"]
colors_right = ["red"] #, "orange", "pink", "black"]

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
    csv_path = f"C:\\Users\\chtun\\AppData\\LocalLow\\DefaultCompany\\Unity_VR_Template\\hand_data-{subname}.csv"
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
                x = (row[f"{prefix}{joint_name}_posX"] + root_x) * scale_factor
                y = (row[f"{prefix}{joint_name}_posY"] + root_y) * scale_factor
                z = (row[f"{prefix}{joint_name}_posZ"] + root_z) * scale_factor
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

# Initial data (first frame of each file)
data = []
for f_idx, (left_frames, right_frames) in enumerate(zip(all_left_frames, all_right_frames)):
    lx, ly, lz = separate_xyz(left_frames[0])
    rx, ry, rz = separate_xyz(right_frames[0])
    data.append(go.Scatter3d(x=lx, y=ly, z=lz, mode='markers',
                             marker=dict(color=colors_left[f_idx], size=4),
                             name=f'Left Hand {f_idx+1}'))
    data.append(go.Scatter3d(x=rx, y=ry, z=rz, mode='markers',
                             marker=dict(color=colors_right[f_idx], size=4),
                             name=f'Right Hand {f_idx+1}'))

fig = go.Figure(data=data)

# Animation frames
frames = []
for i in range(max_frames):
    frame_data = []
    for f_idx, (left_frames, right_frames) in enumerate(zip(all_left_frames, all_right_frames)):
        # Use last frame if sequence is shorter
        lx, ly, lz = separate_xyz(left_frames[min(i, len(left_frames)-1)])
        rx, ry, rz = separate_xyz(right_frames[min(i, len(right_frames)-1)])
        frame_data.append(go.Scatter3d(x=lx, y=ly, z=lz, mode='markers',
                                       marker=dict(color=colors_left[f_idx], size=4)))
        frame_data.append(go.Scatter3d(x=rx, y=ry, z=rz, mode='markers',
                                       marker=dict(color=colors_right[f_idx], size=4)))
    frames.append(go.Frame(data=frame_data, name=str(i)))

fig.frames = frames

# Layout
fig.update_layout(
    scene=dict(
        xaxis=dict(range=[-5, 5]),
        yaxis=dict(range=[0, 10]),
        zaxis=dict(range=[-5, 5]),
        aspectmode='cube'
    ),
    updatemenus=[dict(
        type="buttons",
        buttons=[dict(label="Play",
                      method="animate",
                      args=[None, {"frame": {"duration": 50, "redraw": True},
                                   "fromcurrent": True,
                                   "mode": "immediate"}])]
    )]
)

# Save interactive HTML
fig.write_html(output_file, include_plotlyjs='cdn')
print(f"Saved animation to {output_file}")
