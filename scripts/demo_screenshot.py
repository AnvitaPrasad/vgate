"""Regenerate paper/figures/demo_inspector.png: launch the demo on held-out outputs, open the Inspector on one
query in a narrow viewport (so the text stays legible at column width) and capture, with Playwright, the
image + decision panel, the answers-and-cost table and the first evidence rows, stacked into one figure.

  python scripts/demo_screenshot.py [--ds evqa --qid EVQA_613] [--budgeted --budget 0.5]
Needs `pip install playwright && playwright install chromium`.
"""
import argparse
import pathlib
import subprocess
import sys
import time
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCALE = 3
BOXES_JS = """() => {
  const vis = e => { const r = e.getBoundingClientRect(); return r.width > 0 && r.height > 0; };
  const bb = e => { const r = e.getBoundingClientRect(); return [Math.round(r.x), Math.round(r.y + window.scrollY), Math.round(r.width), Math.round(r.height)]; };
  const blocks = [...document.querySelectorAll('.block')].filter(vis);
  const top = blocks.filter(e => e.querySelector('img') || /^(Q:|Decision:)/.test(e.innerText.trim())).map(bb);
  const tables = [...document.querySelectorAll('.table-wrap')].filter(vis);
  return {top: top, tables: tables.map(bb)};
}"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", default="evqa")
    ap.add_argument("--qid", default="EVQA_613")  # Google Landmarks image (CC BY): closed-book right, VGate searches but declines to read
    ap.add_argument("--budget", type=float, default=0.5)
    ap.add_argument("--budgeted", action="store_true", help="use the budget slider instead of the budget-free rule")
    ap.add_argument("--port", type=int, default=7877)
    ap.add_argument("--width", type=int, default=600)  # narrow viewport: UI text prints at about 6 pt
    ap.add_argument("--ev_rows", type=int, default=0, help="evidence rows kept in the figure (0: no evidence panel)")
    ap.add_argument("--out", default=str(ROOT / "paper/figures/demo_inspector.png"))
    a = ap.parse_args()
    import io

    from PIL import Image
    from playwright.sync_api import sync_playwright
    srv = subprocess.Popen([sys.executable, str(ROOT / "demo/app.py"), "--port", str(a.port), "--init_ds", a.ds,
                            "--init_qid", a.qid, "--init_budget", str(a.budget)] + ([] if a.budgeted else ["--init_auto"]), cwd=ROOT,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        url = f"http://127.0.0.1:{a.port}"
        for _ in range(120):
            try:
                urllib.request.urlopen(url, timeout=2)
                break
            except Exception:
                time.sleep(1)
        with sync_playwright() as p:
            b = p.chromium.launch()
            page = b.new_page(viewport=dict(width=a.width, height=1000), device_scale_factor=SCALE)
            page.goto(url)
            page.wait_for_selector("text=Decision:", timeout=120000)
            page.wait_for_timeout(2500)  # image and tables finish rendering
            boxes = page.evaluate(BOXES_JS)
            shot = Image.open(io.BytesIO(page.screenshot(full_page=True)))
            b.close()
        # Figure = [image | question + decision] over the answers-and-cost table over the first evidence rows.
        x0, x1 = min(b[0] for b in boxes["top"]) - 2, max(b[0] + b[2] for b in boxes["top"]) + 2
        y0, y1 = min(b[1] for b in boxes["top"]) - 2, max(b[1] + b[3] for b in boxes["top"]) + 2
        ans, ev = boxes["tables"][1], boxes["tables"][2]  # tables[0] is the signals table
        row = ans[3] / 4  # the answers table is a header + 3 rows (the dataframe is virtualised: no row boxes)
        ev = [ev[0], ev[1], ev[2], min(ev[3], round(row * (a.ev_rows + 1)))]
        parts = [shot.crop(tuple(SCALE * v for v in (x0, y0, x1, y1)))]
        tables = (ans, ev) if a.ev_rows > 0 else (ans,)  # --ev_rows 0: no evidence panel (keeps the text legible in print)
        parts += [shot.crop(tuple(SCALE * v for v in (t[0] - 2, t[1] - 2, t[0] + t[2] + 2, t[1] + t[3] + 2))) for t in tables]
        w, gap = max(im.width for im in parts), 8 * SCALE
        fig = Image.new("RGB", (w, sum(im.height for im in parts) + gap * (len(parts) - 1)), "white")
        y = 0
        for im in parts:
            fig.paste(im, (0, y))
            y += im.height + gap
        fig.save(a.out)
    finally:
        srv.terminate()
    print("wrote", a.out)


if __name__ == "__main__":
    main()
