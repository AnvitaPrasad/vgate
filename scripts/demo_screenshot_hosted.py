"""Regenerate paper/figures/demo_hosted.png: the hosted log view (the static page of scripts/build_space.sh --static) after one
click on "Use a published system's log": the button, the status line and the audit table, as three crops stacked in page order.

  scripts/build_space.sh --static && python scripts/demo_screenshot_hosted.py
Needs playwright with chromium and network access (the page loads Pyodide from a CDN).
"""
import functools
import http.server
import io
import pathlib
import threading

from PIL import Image
from playwright.sync_api import sync_playwright

ROOT = pathlib.Path(__file__).resolve().parents[1]
W, SCALE, GAP = 500, 3, 8

H = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(ROOT / "build/space_static"))
H.log_message = lambda *a, **k: None


class Server(http.server.ThreadingHTTPServer):
    allow_reuse_address = True


def main():
    srv = Server(("127.0.0.1", 8767), H); threading.Thread(target=srv.serve_forever, daemon=True).start()
    with sync_playwright() as p:
        b = p.chromium.launch(); pg = b.new_page(viewport=dict(width=W, height=1200), device_scale_factor=SCALE)
        pg.goto("http://127.0.0.1:8767/index.html", timeout=60000)
        pg.wait_for_function("document.getElementById('status').textContent.startsWith('Ready')", timeout=300000)
        pg.click("#b2")
        pg.wait_for_function("document.getElementById('status').textContent.startsWith('Done')", timeout=300000)
        parts = [Image.open(io.BytesIO(pg.locator(sel).screenshot())) for sel in ("#b2", "#status", "#audit table")]
        b.close()
    srv.shutdown()
    w = max(im.width for im in parts)
    fig = Image.new("RGB", (w, sum(im.height for im in parts) + GAP * SCALE * (len(parts) - 1)), "white")
    y = 0
    for im in parts:
        fig.paste(im, (0, y)); y += im.height + GAP * SCALE
    out = ROOT / "paper/figures/demo_hosted.png"; fig.save(out); print("wrote", out, fig.size)


if __name__ == "__main__":
    main()
