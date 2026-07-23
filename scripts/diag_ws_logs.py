"""Stream Lemonade backend logs over WebSocket for debugging.

Usage:
    python scripts/diag_ws_logs.py
    python scripts/diag_ws_logs.py --severity Warning,Error
    python scripts/diag_ws_logs.py --tag llama.cpp
    python scripts/diag_ws_logs.py --output /tmp/backend.log
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

import websockets

DEFAULT_HEALTH = "http://127.0.0.1:13305/v1/health"
DEFAULT_WS_PATH = "/logs/stream"


async def discover_ws_port(health_url: str) -> int:
    import urllib.request
    with urllib.request.urlopen(health_url, timeout=5) as resp:
        data = json.loads(resp.read().decode())
    port = data.get("websocket_port")
    if not port:
        raise RuntimeError("websocket_port not reported by /v1/health")
    return int(port)


async def stream_logs(ws_url: str, severity_filter: set[str] | None, tag_filter: str | None, output_path: str | None) -> None:
    async with websockets.connect(ws_url, max_size=50 * 1024 * 1024) as ws:
        await ws.send(json.dumps({"type": "logs.subscribe", "after_seq": None}))
        out_fh = open(output_path, "a", encoding="utf-8") if output_path else None
        try:
            async for raw in ws:
                msg = json.loads(raw)
                msg_type = msg.get("type")
                if msg_type == "logs.snapshot":
                    entries = msg.get("entries", [])
                    print(f"[snapshot] {len(entries)} entries")
                    for entry in entries:
                        await _maybe_print(entry, severity_filter, tag_filter, out_fh)
                elif msg_type == "logs.entry":
                    await _maybe_print(msg.get("entry", {}), severity_filter, tag_filter, out_fh)
                elif msg_type == "error":
                    print(f"[ws error] {msg}", file=sys.stderr)
        finally:
            if out_fh:
                out_fh.close()


async def _maybe_print(entry: dict, severity_filter: set[str] | None, tag_filter: str | None, out_fh) -> None:
    severity = entry.get("severity", "")
    tag = entry.get("tag", "")
    if severity_filter and severity not in severity_filter:
        return
    if tag_filter and tag_filter.lower() not in tag.lower():
        return
    line = entry.get("line", "")
    text = f"{entry.get('timestamp', '')} [{severity}] ({tag}) {line}"
    print(text, flush=True)
    if out_fh:
        out_fh.write(text + "\n")
        out_fh.flush()


def main() -> int:
    parser = argparse.ArgumentParser(description="Stream Lemonade backend logs over WebSocket")
    parser.add_argument("--health", default=DEFAULT_HEALTH, help="Health endpoint URL")
    parser.add_argument("--ws-path", default=DEFAULT_WS_PATH, help="WebSocket path")
    parser.add_argument("--severity", default=None, help="Comma-separated severity filter, e.g. Warning,Error")
    parser.add_argument("--tag", default=None, help="Tag substring filter, e.g. llama.cpp")
    parser.add_argument("--output", default=None, help="Append logs to file")
    args = parser.parse_args()

    severity_filter = {s.strip() for s in args.severity.split(",")} if args.severity else None
    try:
        port = asyncio.run(discover_ws_port(args.health))
    except Exception as exc:
        print(f"Failed to discover websocket port: {exc}", file=sys.stderr)
        return 1
    ws_url = f"ws://127.0.0.1:{port}{args.ws_path}"
    print(f"Connecting to {ws_url}")
    try:
        asyncio.run(stream_logs(ws_url, severity_filter, args.tag, args.output))
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
