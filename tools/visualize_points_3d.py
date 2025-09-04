# pip install open3d
import os
import numpy as np
os.environ["LIBGL_ALWAYS_SOFTWARE"] = "1"  
import open3d as o3d
import plotly.graph_objects as go
import plotly.io as pio
import subprocess

pio.renderers.default = "browser"  

CLASS_NAMES = [
    'car','truck','trailer','bus','construction_vehicle',
    'bicycle','motorcycle','pedestrian','traffic_cone','barrier'
]

CLASS_COLORS = {
    "car": "#1f77b4",
    "truck": "#ff7f0e",
    "construction_vehicle": "#8c564b",
    "bus": "#2ca02c",
    "trailer": "#989d9e",
    "barrier": "#d62728",
    "motorcycle": "#9467bd",
    "bicycle": "#e377c2",
    "pedestrian": "#00f3fc",
    "traffic_cone": "#bcbd22",
}


def load_points(npz_path):
    if not os.path.exists(npz_path):
        raise FileNotFoundError(npz_path)
    z = np.load(npz_path)
    key = "points" if "points" in z.files else "arr_0"
    return z[key]

def make_class_colored_html(points,
                            out_html,
                            class_start=5,
                            num_classes=10,
                            class_names=CLASS_NAMES,
                            div_id="pcd_viewer",
                            label_threshold=0.5):
    """
    Colors points by one-hot classes at columns [class_start : class_start+num_classes).
    Unlabeled (all-zeros) are shown in grayscale by height (z).
    Double-click resets; click any point to recenter+zoom; Shift+Click for gentler zoom.
    """
    assert points.shape[1] >= 3, "points must have xyz in [:,0:3]"
    xyz = points[:, :3]

    # One-hot slice
    end = class_start + num_classes
    if points.shape[1] < end:
        raise ValueError(f"points shape {points.shape} lacks class columns [{class_start}:{end}).")

    onehot = points[:, class_start:end]
    # Pick the winning class and its score
    label_idx = np.argmax(onehot, axis=1)
    label_score = onehot[np.arange(onehot.shape[0]), label_idx]
    labeled_mask = label_score > label_threshold
    unlabeled_mask = ~labeled_mask

    # Build a figure with 1 trace per class (legend toggleable) + 1 unlabeled trace
    fig = go.Figure()

    # Add class traces
    for k, name in enumerate(class_names):
        m = labeled_mask & (label_idx == k)
        if not np.any(m):
            continue
        pts = xyz[m]
        fig.add_trace(go.Scatter3d(
            x=pts[:,0], y=pts[:,1], z=pts[:,2],
            mode='markers',
            name=name,
            marker=dict(size=2, color=CLASS_COLORS.get(name, "#000000"), opacity=0.9),
            legendgroup=name,
            showlegend=True
        ))

    # Unlabeled: grayscale by height
    if np.any(unlabeled_mask):
        pts = xyz[unlabeled_mask]
        fig.add_trace(go.Scatter3d(
            x=pts[:,0], y=pts[:,1], z=pts[:,2],
            mode='markers',
            name="unlabeled (height)",
            marker=dict(size=2, color=pts[:,2], colorscale="Gray", opacity=0.6, showscale=False),
            legendgroup="unlabeled",
            showlegend=True
        ))

    fig.update_layout(
        scene_aspectmode='data',
        scene_dragmode='orbit',
        legend=dict(orientation="h", yanchor="bottom", y=0.01, xanchor="left", x=0.01),
        height=800,
        margin=dict(l=0, r=0, t=30, b=0),
        title="nuScenes point cloud (class-colored; click to recenter)"
    )
    
    pio.write_html(fig, file=out_html, auto_open=False,
                   include_plotlyjs='cdn', full_html=True,
                   default_width='100%', default_height='100%',
                   div_id=div_id)
    print("[INFO] Saved to:", out_html)

    
    
if __name__ == '__main__':
    npz_directory = "/home/betty/CMU-intern/pillarnext/visualize_pointcloud/clustered"
    filename = "0fa505e5dd804d3b9f9f076d23b28d6d_fused_pts"
    out_html = os.path.join(npz_directory, f"{filename}.html")
    filepath = os.path.join(npz_directory, f"{filename}.npz")
    data = np.load(filepath)
    points = data["arr_0"]
    make_class_colored_html(points, out_html)