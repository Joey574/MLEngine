import sys

import matplotlib.pyplot as plt
import numpy as np

DPI = 2000


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

    fig, ax = plt.subplots(figsize=(8, 8), dpi=DPI)
    ax.scatter(
        x,
        y,
        s=0.01,
        c="black",
        alpha=0.6,
        linewidths=0,
    )

    ax.set_aspect("equal", adjustable="box")
    ax.grid(False)
    ax.set_title(f"Dataset Visualization ({n_points:,} points)", fontsize=8)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    plt.savefig("points.png", format="png", dpi=DPI, bbox_inches="tight")


# Example usage:
# parse_and_plot_points('your_data.bin')
parse_and_plot_points(sys.argv[1])
