import matplotlib.pyplot as plt
import numpy as np


def parse_and_plot_points(filepath, n_points=None):
    """
    Parses an interleaved binary float32 file of format x1,x2...xn,y1,y2...yn
    and plots them efficiently.
    """
    # 1. Efficient Loading
    # Total floats = 2 * n_points. We read the entire file into a 1D float32 array.
    data = np.fromfile(filepath, dtype=np.float32)
    if data.size % 2 != 0:
        raise ValueError(f"{filepath}: {data.size} floats is odd; not a point array")

    points = data.reshape(-1, 2)
    if n_points is not None and len(points) != n_points:
        print(f"Warning: expected {n_points:,} points, file contains {len(points):,}")
    n_points = len(points)

    # Reshape or slice into X and Y arrays efficiently (no copying, just views)
    x = points[:, 0]
    y = points[:, 1]

    # 2. Optimized Plotting
    fig, ax = plt.subplots(figsize=(8, 8), dpi=300)  # High DPI for crisp rendering

    # Use plt.scatter with tiny markers (s=0.1 or s=1) and no edge colors
    # to prevent overlapping artifacts and heavy rendering.
    ax.scatter(
        x,
        y,
        s=0.5,  # Marker size (very small)
        c="black",  # Color
        alpha=0.6,  # Transparency to see density hotspots
        linewidths=0,  # Remove edge outlines for speed and cleaner look
    )

    # Grid and styling
    ax.set_aspect("equal", adjustable="box")  # Optional: keeps aspect ratio square
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_title(f"Dataset Visualization ({n_points:,} points)", fontsize=12)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    # Save to a high-resolution raster image (or vector if needed)
    plt.savefig("points_grid.png", dpi=300, bbox_inches="tight")
    print("Plot saved successfully as 'points_grid.png'.")


# Example usage:
# parse_and_plot_points('your_data.bin', n_points=250000)
parse_and_plot_points("dataset.bin")
