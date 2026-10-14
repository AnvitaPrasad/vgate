# Deploying the live demo (not done yet — needs the owner's account and go-ahead)

Research behind these choices: `docs/strengthening/R1_live_demo.md`; licence constraints: `docs/strengthening/R4_release.md`.

## Published 2026-10-11: the log view as a static page, https://huggingface.co/spaces/AnvitaPrasad/vgate
A Gradio Space on free hardware needs a paid plan on this account, and the Gradio-Lite runtime no longer installs
(5.45.0: no wheel for its `huggingface-hub` pin; 4.44.1 and 5.20.0 never finish loading). So the hosted page runs the log
view's own function (`demo/app.py: byo`) on Pyodide in the visitor's browser, with a plain HTML front end
(`demo/space_static/`). `scripts/build_space.sh --static` assembles `build/space_static/`; upload that folder to the Space
(SDK: static). Checked in a headless browser on the live URL: it boots in about 25 s, audits the published log in 10 s and
the full sample log in about 40 s; the last digit can differ from the Gradio app (older scikit-learn in Pyodide).
The same folder works on GitHub Pages or any static host.

## Option 0 — the log view on a free CPU Space (Gradio; needs a paid plan on this account)
The part of the demo a reviewer can try without a GPU: upload a log, or load the case study's sample log or the published
system's log, and get the frontiers, the audit table and the planner. No model, no benchmark text or images.

1. `scripts/build_space.sh` assembles `build/space_log/` (5.5 MB: `app.py`, `requirements.txt`, the Space card, `demo/app.py`,
   the two sample logs, `src/vgate/*.py`, the licence).
2. Create a Space (SDK: Gradio, hardware: CPU basic, free) on the owner's Hugging Face account and push that folder to it.
3. Check: the page loads, both sample-log buttons fill the audit table (told the segment 46.0 on the first, logged gate 27.3
   against 29.3 on the second), an uploaded CSV works.
Local test of the assembled folder: `cd build/space_log && python app.py` (needs gradio, numpy, pandas, scikit-learn, scipy,
matplotlib).

## Option A — Hugging Face ZeroGPU Space (first choice)
Requirements on the account (https://huggingface.co/docs/hub/spaces-zerogpu): a free personal account "in good
standing (verified email, account older than 30 days)" can host up to 2 ZeroGPU Spaces; otherwise PRO. Visitors get
2 GPU-minutes a day without an account and 5 with a free account, charged by the declared duration (30 s here).

1. Create a Space with the Gradio SDK and ZeroGPU hardware.
2. Copy into it: `demo/space/{app.py,requirements.txt,README.md}`, `src/vgate/` as `vgate/` (including `gates/`),
   `demo/app.py` as `demo_app.py`, and the index built by job J9 (`index/passage_emb.npy`, 176 MB, via Git LFS, and
   `index/passages.jsonl`).
3. **Check before sharing the link** (the hook path and fp16 on a different GPU are untested there): ask the ~50
   logged questions listed by `scripts/live_report.py` and compare `u_pre` with
   `experiments/results/frozen_gate_scores__Qwen3-VL-4B-Instruct.csv`; the signs must agree as on the T4.
4. Do not upload benchmark images. The live tab uses the visitor's own image; passages are Wikipedia text and need the
   CC BY-SA footer that `demo/app.py` already prints.

## Option B — the A2000 (or any CUDA machine) behind a tunnel
`python demo/app.py --live <index folder> --live_dtype bfloat16 --public` and expose the port with Tailscale Funnel.
The gates were fitted on fp16 features from a T4, so repeat the check in step 3 first.

## Option C — video only
Record the Live tab from any GPU session (for example a Kaggle notebook with `share=True`; the link lasts a week) and
keep the hosted demo to the replay tabs.
