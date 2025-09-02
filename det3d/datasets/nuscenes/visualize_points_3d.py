# pip install open3d
import os
import numpy as np
os.environ["LIBGL_ALWAYS_SOFTWARE"] = "1"  # force Mesa software rasterizer
import open3d as o3d
import plotly.graph_objects as go
import plotly.io as pio
import subprocess

pio.renderers.default = "browser"   # or "vscode" if you use VS Code's Python plots

def load_points(npz_path):
    if not os.path.exists(npz_path):
        raise FileNotFoundError(npz_path)
    z = np.load(npz_path)
    key = "points" if "points" in z.files else "arr_0"
    return z[key]

def plotly_points_3d(points, out_html):
    xyz = points[:, :3]
    fig = go.Figure(go.Scatter3d(
        x=xyz[:,0], y=xyz[:,1], z=xyz[:,2],
        mode='markers', marker=dict(size=2, color=xyz[:,2], colorscale="Gray", opacity=0.9),
    ))
    fig.update_layout(scene_aspectmode='data', height=800)
    fig.write_html(out_html, auto_open=False)
    print(f"Saved HTML to {out_html}")
    # win_path = subprocess.check_output(["wslpath", "-w", out_html]).decode().strip()
    # subprocess.run(["powershell.exe", "-NoProfile", "-Command", f"Start-Process '{win_path}'"])
    # print("Opened in Windows browser:", win_path)
    # fig.show()  # opens your browser now

    
    
if __name__ == '__main__':
    out_html = "/home/betty/CMU-intern/pillarnext/visualize_pointcloud/view_cloud.html"
    npz_directory = "/home/betty/CMU-intern/pillarnext/visualize_pointcloud"
    filename = "0cd661df01aa40c3bb3a773ba86f753a_fused_pts.npz"
    filepath = os.path.join(npz_directory, filename)
    data = np.load(filepath)
    points = data["arr_0"]
    plotly_points_3d(points, out_html=out_html)