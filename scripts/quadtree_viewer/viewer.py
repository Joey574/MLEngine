#!/usr/bin/env python3
"""Build a disk-backed spatial index and serve a WebGL2 quadtree-cell viewer.

Usage:
  python viewer.py build dump.bin
  python viewer.py serve dump.bin --port 8000

Binary input: repeated little-endian float64 (x, y, width, height).
"""

import argparse
import json
import math
import os
import sqlite3
import struct
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

CELL = struct.Struct("<dddd")
REPLY = struct.Struct("<QII")  # last inspected id, records, has_more
DEFAULT_BATCH = 100_000
MAX_BATCH = 500_000


def index_path(raw):
    return Path(str(raw) + ".sqlite")


def build(raw, db, batch_size=100_000):
    raw = Path(raw).resolve()
    if not raw.is_file():
        raise SystemExit(f"No such file: {raw}")
    if db.exists():
        raise SystemExit(f"Index already exists: {db}\nDelete it to rebuild.")
    source_stat = raw.stat()
    if source_stat.st_size < CELL.size:
        raise SystemExit("Input has no complete cells")
    if source_stat.st_size % CELL.size:
        print(f"Warning: ignoring {source_stat.st_size % CELL.size} trailing bytes")
    con = sqlite3.connect(db)
    try:
        con.executescript("""
            PRAGMA journal_mode=WAL;
            PRAGMA synchronous=NORMAL;
            CREATE TABLE cells (id INTEGER PRIMARY KEY, x REAL NOT NULL, y REAL NOT NULL,
                                w REAL NOT NULL, h REAL NOT NULL);
            CREATE VIRTUAL TABLE boxes USING rtree(id, xmin, xmax, ymin, ymax);
            CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        """)
        bounds = [math.inf, math.inf, -math.inf, -math.inf]
        count = 0
        invalid = 0
        data_rows, box_rows = [], []
        # Reused read buffer; no mmap, no file-sized allocation.
        buf = bytearray(batch_size * CELL.size)
        mv = memoryview(buf)
        with raw.open("rb", buffering=0) as f:
            try:
                os.posix_fadvise(f.fileno(), 0, 0, os.POSIX_FADV_SEQUENTIAL)
            except (AttributeError, OSError):
                pass
            remaining = source_stat.st_size // CELL.size
            while remaining:
                take = min(remaining, batch_size)
                needed = take * CELL.size
                received = 0
                while received < needed:
                    n = f.readinto(mv[received:needed])
                    if not n:
                        raise EOFError("Source file shrank during indexing")
                    received += n
                remaining -= take
                for x, y, w, h in CELL.iter_unpack(mv[:needed]):
                    if not all(map(math.isfinite, (x, y, w, h))) or w <= 0 or h <= 0:
                        invalid += 1
                        continue
                    x1, y1 = x + w, y + h
                    if not math.isfinite(x1) or not math.isfinite(y1):
                        invalid += 1
                        continue
                    count += 1
                    data_rows.append((count, x, y, w, h))
                    box_rows.append((count, x, x1, y, y1))
                    bounds[0] = min(bounds[0], x)
                    bounds[1] = min(bounds[1], y)
                    bounds[2] = max(bounds[2], x1)
                    bounds[3] = max(bounds[3], y1)
                with con:
                    con.executemany("INSERT INTO cells VALUES (?,?,?,?,?)", data_rows)
                    con.executemany("INSERT INTO boxes VALUES (?,?,?,?,?)", box_rows)
                data_rows.clear()
                box_rows.clear()
                if (count + invalid) % (batch_size * 50) < batch_size:
                    print(f"Indexed {count:,} cells ({invalid:,} skipped)", flush=True)
        if not count:
            raise ValueError("No valid cells")
        stat_after = raw.stat()
        if (stat_after.st_size, stat_after.st_mtime_ns) != (
            source_stat.st_size,
            source_stat.st_mtime_ns,
        ):
            raise RuntimeError("Source file changed while indexing")
        metadata = {
            "raw_size": source_stat.st_size,
            "raw_mtime_ns": source_stat.st_mtime_ns,
            "bounds": bounds,
            "count": count,
            "invalid": invalid,
        }
        with con:
            con.executemany(
                "INSERT INTO meta VALUES (?,?)",
                [(key, json.dumps(value)) for key, value in metadata.items()],
            )
        print(f"Finished: {count:,} cells; skipped {invalid:,}; bounds={bounds}")
        print(
            f"Index: {db} ({db.stat().st_size / (1024**2):.1f} MiB before WAL checkpoint)"
        )
        con.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    except BaseException:
        con.close()
        db.unlink(missing_ok=True)
        raise
    finally:
        con.close()


def load_meta(raw, db):
    if not db.exists():
        raise SystemExit(f'Index not found: {db}\nRun: python viewer.py build "{raw}"')
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        meta = {k: json.loads(v) for k, v in con.execute("SELECT key,value FROM meta")}
    finally:
        con.close()
    st = raw.stat()
    if (meta["raw_size"], meta["raw_mtime_ns"]) != (st.st_size, st.st_mtime_ns):
        raise SystemExit(
            "Binary file changed since indexing. Delete the .sqlite file and rebuild."
        )
    return meta


def serve(raw, db, host, port):
    meta = load_meta(raw, db)
    html = (Path(__file__).resolve().parent / "index.html").read_bytes()

    class Handler(BaseHTTPRequestHandler):
        def reply(self, status, content, mime):
            self.send_response(status)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(content)

        def do_GET(self):
            route = urlsplit(self.path)
            try:
                if route.path == "/":
                    return self.reply(200, html, "text/html; charset=utf-8")
                if route.path == "/meta":
                    return self.reply(
                        200, json.dumps(meta).encode(), "application/json"
                    )
                if route.path == "/cells":
                    p = parse_qs(route.query)

                    def one(key, default=None):
                        return p.get(key, [default])[0]

                    xmin = float(one("xmin"))
                    ymin = float(one("ymin"))
                    xmax = float(one("xmax"))
                    ymax = float(one("ymax"))
                    cursor = int(one("cursor", "0"))
                    limit = min(
                        max(int(one("limit", str(DEFAULT_BATCH))), 1), MAX_BATCH
                    )
                    if not all(map(math.isfinite, (xmin, ymin, xmax, ymax))):
                        raise ValueError("Nonfinite viewport")
                    if xmin >= xmax or ymin >= ymax or cursor < 0:
                        raise ValueError("Invalid viewport or cursor")
                    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
                    try:
                        # RTree uses outward-rounded float32 bounds, and cells uses the
                        # original float64 coordinates. Recheck exact intersections.
                        rows = con.execute(
                            """
                            SELECT cells.id, x, y, w, h FROM boxes
                            JOIN cells ON cells.id = boxes.id
                            WHERE boxes.xmin <= ? AND boxes.xmax >= ?
                              AND boxes.ymin <= ? AND boxes.ymax >= ?
                              AND boxes.id > ? ORDER BY boxes.id
                        """,
                            (xmax, xmin, ymax, ymin, cursor),
                        )
                        buf = bytearray(REPLY.size)
                        sent = 0
                        last = cursor
                        more = 0
                        for id_, x, y, w, h in rows:
                            last = id_
                            if x > xmax or x + w < xmin or y > ymax or y + h < ymin:
                                continue
                            if sent == limit:
                                more = 1
                                # Do not advance past an unsent valid cell.
                                last = id_ - 1
                                break
                            buf.extend(CELL.pack(x, y, w, h))
                            sent += 1
                        REPLY.pack_into(buf, 0, last, sent, more)
                    finally:
                        con.close()
                    return self.reply(200, buf, "application/octet-stream")
                self.reply(404, b"Not found", "text/plain")
            except (ValueError, TypeError, OverflowError) as e:
                self.reply(400, str(e).encode(), "text/plain")
            except (BrokenPipeError, ConnectionResetError):
                pass

    httpd = ThreadingHTTPServer((host, port), Handler)
    print(f"Open http://{host}:{port}/  ({meta['count']:,} cells)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    subs = ap.add_subparsers(dest="command", required=True)
    for cmd in ("build", "serve"):
        p = subs.add_parser(cmd)
        p.add_argument("dump", type=Path)
        p.add_argument("--index", type=Path, default=None)
        if cmd == "serve":
            p.add_argument("--host", default="127.0.0.1")
            p.add_argument("--port", type=int, default=8000)
    a = ap.parse_args()
    raw = a.dump.resolve()
    db = (a.index or index_path(raw)).resolve()
    if a.command == "build":
        build(raw, db)
    else:
        serve(raw, db, a.host, a.port)


if __name__ == "__main__":
    main()
