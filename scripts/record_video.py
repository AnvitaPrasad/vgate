"""Record the demo video: a captioned screen recording of the web application (no voice), under three minutes.

Launches the demo on the held-out outputs in --public mode (only redistributable images, with credit), drives it with
Playwright, shows the script of demo/VIDEO_SCRIPT.md as captions, and converts the recording to MP4 with ffmpeg.

  python scripts/record_video.py [--fast] [--out build/video/vgate_demo.mp4]
Needs the case study's outputs on disk (make score confirm extend), playwright with chromium, and ffmpeg.
--fast shortens every pause (to check the selectors).
"""
import argparse
import pathlib
import re
import shutil
import subprocess
import sys
import time
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[1]
W, H = 1280, 720
OVERLAY_JS = """() => {
  const s = document.createElement('style');
  s.textContent = `#vg-cap{position:fixed;left:0;right:0;bottom:0;z-index:99999;background:rgba(16,20,28,.92);color:#fff;
    font:500 21px/1.35 system-ui,sans-serif;padding:14px 28px;min-height:58px;display:flex;align-items:center}
    #vg-card{position:fixed;inset:0;z-index:99998;background:#10141c;color:#fff;display:flex;flex-direction:column;
    align-items:center;justify-content:center;text-align:center;font-family:system-ui,sans-serif;padding:0 90px}
    #vg-card h1{font-size:40px;margin:0 0 18px;line-height:1.2} #vg-card p{font-size:22px;margin:6px 0;color:#c9d3e3}`;
  document.head.appendChild(s);
  const c = document.createElement('div'); c.id = 'vg-cap'; document.body.appendChild(c);
  const k = document.createElement('div'); k.id = 'vg-card'; k.style.display = 'none'; document.body.appendChild(k);
}"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "build/video/vgate_demo.mp4"))
    ap.add_argument("--port", type=int, default=7879)
    ap.add_argument("--fast", action="store_true")
    a = ap.parse_args()
    from playwright.sync_api import sync_playwright
    out = pathlib.Path(a.out); tmp = out.parent / "raw"; shutil.rmtree(tmp, ignore_errors=True); tmp.mkdir(parents=True)
    srv = subprocess.Popen([sys.executable, str(ROOT / "demo/app.py"), "--port", str(a.port), "--init_ds", "evqa", "--init_qid", "EVQA_613",
                            "--init_auto", "--public"], cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    k = 0.15 if a.fast else 1.0
    try:
        url = f"http://127.0.0.1:{a.port}"
        for _ in range(180):
            try:
                urllib.request.urlopen(url, timeout=2); break
            except Exception:
                time.sleep(1)
        with sync_playwright() as p:
            b = p.chromium.launch()
            ctx = b.new_context(viewport=dict(width=W, height=H), record_video_dir=str(tmp), record_video_size=dict(width=W, height=H))
            t0 = time.time(); pg = ctx.new_page()
            pg.goto(url); pg.wait_for_selector("text=Decision:", timeout=180000); pg.wait_for_timeout(2500)
            pg.evaluate(OVERLAY_JS)
            marks = []  # (seconds into the final video, caption): printed at the end, to check frames against

            def cap(text, secs):
                if text:
                    marks.append((time.time() - t0, text[:60]))
                pg.evaluate("t => { document.getElementById('vg-cap').textContent = t; }", text)
                pg.wait_for_timeout(int(1000 * secs * k))
            card = lambda html: pg.evaluate("h => { const k = document.getElementById('vg-card'); k.innerHTML = h; k.style.display = h ? 'flex' : 'none'; }", html)
            tab = lambda name: (pg.get_by_role("tab", name=name, exact=True).click(), pg.wait_for_timeout(1200))
            scroll = lambda y: pg.evaluate("y => window.scrollTo({top: y, behavior: 'smooth'})", y)
            into = lambda text: pg.get_by_text(text, exact=False).first.evaluate(  # the element near the top of the window, clear of the caption bar
                "e => window.scrollTo({top: e.getBoundingClientRect().top + window.scrollY - 130, behavior: 'smooth'})")

            card("<h1>VGate: Auditing What Retrieval Gates Learn for Vision-Language Models</h1><p>WSDM 2027 demonstration</p>"
                 "<p>A screen recording of the web application; captions only, no sound.</p>")
            start = time.time() - t0  # everything before the title card is cut
            cap("", 6)
            card("")
            cap("Retrieved evidence helps a vision-language model on some questions and breaks correct answers on others. "
                "VGate fits retrieve-or-skip gates from logs, and audits what they have learned.", 9)
            cap("Inspector, a held-out question: \"In which city is this hotel located?\" Closed-book, the model is right. "
                "VGate decides twice: whether to search, then whether to read.", 9)
            scroll(330)
            cap("Here it searches, then declines to read, and keeps the correct answer that always-retrieve loses. "
                "Every pass is costed in tokens and GPU-seconds.", 9)
            scroll(0); tab("Harm auditor")
            cap("Harm auditor: every query where retrieval broke a correct answer, with the gate's decision.", 7)
            tab("Audit your own log")
            pg.get_by_role("button", name=re.compile("Use the sample log")).click()
            cap("Audit your own log. Upload a CSV with two outcomes per query, any signals, and a segment. "
                "Here: the case study's held-out log; the segment is the benchmark.", 9)
            cap("The app fits a benefit-trained and an error-trained gate on the log itself, then compares them with policies "
                "that only know the segment. (Fitting takes some seconds.)", 6)
            pg.get_by_text("segment classifier", exact=False).first.wait_for(timeout=300000)
            into("segment classifier"); pg.wait_for_timeout(800)
            cap("The gate reaches 45.7%. A classifier that only predicts the segment reaches 45.6, and being told the segment "
                "46.0: the gate has mostly learned which benchmark it is looking at.", 11)
            cap("Fitted on two segments and applied to the third, it falls far below. A decision column puts any logged gate "
                "through the same table; here, a published search-need prompt.", 10)
            into("should you gate?"); pg.wait_for_timeout(800)
            cap("The planner: how often retrieval helps and harms, the separation a gate needs to lead the better fixed policy "
                "by a point, and the noise floor of a threshold tuned on the same labels.", 10)
            scroll(0)
            cap("Next: the same view on a log we did not produce, loaded with one click.", 3)
            pg.get_by_role("button", name=re.compile("Use a published system's log")).click()
            pg.get_by_text("n=600", exact=False).first.wait_for(timeout=120000)
            pg.get_by_text("logged gate", exact=True).first.evaluate(
                "e => window.scrollTo({top: e.getBoundingClientRect().top + window.scrollY - 330, behavior: 'smooth'})"); pg.wait_for_timeout(800)
            cap("The same audit on another group's released log: GPT-4o's answers and their trained gate's decisions, six sources. "
                "By token accuracy, the logged gate is two points below the policy told the source.", 12)
            scroll(0); tab("Frontier")
            cap("Frontier: accuracy against cost for benefit- and error-trained gates. The budget slider lists the queries "
                "whose decision flips next.", 8)
            tab("Learn from feedback")
            pg.get_by_role("button", name="Run the stream").click()
            cap("Learn from feedback: a simulation that learns the gate from judgements of the shown answer alone. "
                "It ends within 0.8 points of the gate fitted on paired data.", 13)
            card("<h1>VGate</h1><p>Try the log view in your browser: hf.co/spaces/AnvitaPrasad/vgate</p>"
                 "<p>Code, frozen splits, outputs and preregistration log: github.com/AnvitaPrasad/vgate</p>")
            cap("", 7)
            total = time.time() - t0
            ctx.close(); raw = pathlib.Path(pg.video.path()); b.close()
        dur = total - start
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{start:.2f}", "-i", str(raw), "-t", f"{dur:.2f}", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                        "-crf", "23", "-movflags", "+faststart", str(out)], check=True)
        print(f"wrote {out}: {dur:.0f} s, {out.stat().st_size / 1e6:.1f} MB")
        for t, text in marks:
            print(f"  {t - start:6.1f}s  {text}")
    finally:
        srv.terminate()


if __name__ == "__main__":
    main()
