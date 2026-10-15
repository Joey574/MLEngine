# WebGL2 Quadtree Cell Viewer

Interactive viewer for files containing **headerless, little-endian float64** records:
`x, y, width, height` (32 bytes per cell; `(x,y)` is bottom-left).

Requires Python 3.9+ with only the standard library, and a browser with WebGL2. No NumPy, OpenGL development packages, or external JS dependencies.

## Run

```bash
python3 viewer.py build /path/to/dump.bin
python3 viewer.py serve /path/to/dump.bin --port 8000
```

Open **http://127.0.0.1:8000/** in a browser.

The first command creates `/path/to/dump.bin.sqlite`, containing the original double-precision coordinates plus a spatial R-tree. This may take time and considerable disk space for multi-million-cell dumps, but memory usage during indexing is bounded. To rebuild, delete the `.sqlite` file first. The `.sqlite` must remain alongside the dump and cannot be reused if the source's size/modification time changes.

**Controls:** scroll to zoom at cursor, drag to pan, double-click to zoom, Shift+double-click to zoom out, F to fit, R to reload. Adjust the per-view cell cap in the top bar.

## How it works

- Python reads the dump in reusable buffers, ignoring malformed/nonpositive cells.
- SQLite stores world coordinates in float64 (SQLite REAL), alongside a **float32 R-tree** for fast approximate spatial lookup. The server rechecks intersections using the original float64 coordinates.
- An HTTP endpoint pages only cells intersecting the current viewport, sending raw float64 records back to the browser.
- The browser keeps visible records as float64 and projects them relative to the float64 camera center *before* transferring float32 normalized device coordinates to WebGL2. `gl.drawArraysInstanced(gl.LINES, 0, 8, n)` draws each cell as four outline segments.
- The browser caps the number of visible records to avoid freezes. When the cap is reached, **only a subset is shown**, not an overview/aggregation; zoom in to inspect the rest.

## Caveats

- R-tree indexing and SQLite indexes require additional disk space (potentially substantially more than the 32-byte-per-cell dump). Large, dense views can return huge numbers of cells: reduce the cap or zoom in.
- The SQLite R-tree bounds are float32, so *extreme* zoom in a region where many records collapse to the same float32 coordinates may result in slow queries. Intersection checking stays accurate at float64 precision; the limitation is lookup efficiency.
- Browser float64 arithmetic cannot distinguish coordinates beyond ~15–17 significant decimal digits. The viewer cannot recover details lost in the original doubles.
- `x + width` is rounded in float64, so cells smaller than the floating-point ULP at their world position cannot be represented faithfully.
- OpenGL `LINES` are usually one physical pixel wide. Lines remain crisp while zooming but are intentionally screen-space thin; overlapping cells may obscure one another.
- For best performance, serve on localhost. Binding to a LAN interface with `--host 0.0.0.0` exposes the unauthenticated API; do so only on a trusted network.
