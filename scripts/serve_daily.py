"""Serve only the generated public directory on loopback, without auto-opening windows."""

from __future__ import annotations

import argparse
import functools
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8501)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1] / "_site"
    if not (root / "data.json").exists():
        raise SystemExit("Run scripts/daily_picks.py --saved-only first")
    handler = functools.partial(SimpleHTTPRequestHandler, directory=str(root))
    server = HTTPServer(("127.0.0.1", args.port), handler)
    print(f"Local: http://127.0.0.1:{args.port}/", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
