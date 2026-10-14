"""Regenerate paper/figures/demo_audit.png: launch the demo, open "Audit your own log", load the sample log and capture
the audit table with Playwright.

  python scripts/demo_screenshot_audit.py
Needs `pip install playwright && playwright install chromium`.
"""
import argparse
import pathlib
import subprocess
import sys
import time
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[1]
BOX_JS = """() => {
  const t = [...document.querySelectorAll('.table-wrap')].filter(e => { const r = e.getBoundingClientRect();
    return r.width > 0 && r.height > 0 && e.innerText.includes('told the segment'); })[0];
  const r = t.getBoundingClientRect();
  return [Math.round(r.x), Math.round(r.y + window.scrollY), Math.round(r.width), Math.round(r.height)];
}"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=7878)
    ap.add_argument("--width", type=int, default=760)
    ap.add_argument("--out", default=str(ROOT / "paper/figures/demo_audit.png"))
    ap.add_argument("--published", action="store_true", help="load the published system's log (GPT-4o; Chen et al.'s gate) instead of the sample log")
    ap.add_argument("--log_only", action="store_true", help="start the app without the case-study runs (the log view alone), so that a "
                    "narrow window does not fold the tabs away; with a narrow --width the table's text prints larger")
    a = ap.parse_args()
    import io

    from PIL import Image
    from playwright.sync_api import sync_playwright
    srv = subprocess.Popen([sys.executable, str(ROOT / "demo/app.py"), "--port", str(a.port)] + (["--run", "build/no_such_run"] if a.log_only else []),
                           cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        url = f"http://127.0.0.1:{a.port}"
        for _ in range(180):
            try:
                urllib.request.urlopen(url, timeout=2)
                break
            except Exception:
                time.sleep(1)
        with sync_playwright() as p:
            b = p.chromium.launch()
            page = b.new_page(viewport=dict(width=a.width, height=1400), device_scale_factor=3)
            page.goto(url)
            if not a.log_only:
                page.get_by_role("tab", name="Audit your own log").click()
            page.get_by_role("button", name="Use a published system's log" if a.published else "Use the sample log").click()
            page.wait_for_selector("text=told the segment", state="attached", timeout=300000)
            page.wait_for_timeout(3000)
            box = page.evaluate(BOX_JS)  # the visible table that holds the audit (the dataframe keeps a hidden copy)
            shot = Image.open(io.BytesIO(page.screenshot(full_page=True)))
            shot.crop(tuple(3 * v for v in (box[0] - 2, box[1] - 2, box[0] + box[2] + 2, box[1] + box[3] + 2))).save(a.out)
            b.close()
    finally:
        srv.terminate()
    print("wrote", a.out)


if __name__ == "__main__":
    main()
