"""Append one row per (model, dataset, condition) of a scored run to experiments/results/experiment_database.csv.

  python scripts/log_run.py --run experiments/runs/J0 --run_id J0 --platform colab-T4 --notes "env probe"
Idempotent per (run_id, model, dataset, condition): existing rows for the same key are not duplicated.
"""
import argparse
import csv
import datetime
import json
import pathlib
import subprocess

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
DB = ROOT / "experiments/results/experiment_database.csv"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--run_id", required=True)
    ap.add_argument("--platform", required=True)
    ap.add_argument("--split", default="dev")
    ap.add_argument("--status", default="RUN")
    ap.add_argument("--notes", default="")
    ap.add_argument("--git_sha", default="", help="commit the run's code matched (default: HEAD)")
    a = ap.parse_args()
    run = pathlib.Path(a.run)
    sha = a.git_sha or subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    split_sha = {l.split()[1]: l.split()[0][:12] for l in open(ROOT / "data/splits/SPLITS.sha256")}
    with open(DB) as f:
        header = next(csv.reader(f))
        existing = {(r["run_id"], r["model"], r["dataset"], r["condition"]) for r in csv.DictReader(open(DB))}
    rows = []
    for meta_p in sorted(run.glob("meta__*.json")):
        meta = json.loads(meta_p.read_text())
        _, ds, mname = meta_p.stem.split("__")
        scores = run / f"scores__{ds}__{mname}.csv"
        sc = pd.read_csv(scores, index_col="qid") if scores.exists() else None
        for f in sorted(run.glob(f"{ds}__{mname}__*.jsonl")):
            cond = f.stem.split("__")[-1]
            if (a.run_id, meta["model"], ds, cond) in existing:
                continue
            recs = [json.loads(l) for l in open(f)]
            acc = sc[f"c_{cond}"].mean() if sc is not None and f"c_{cond}" in sc else np.nan
            tok = np.mean([r.get("n_prompt", 0) + r.get("n_gen", 0) for r in recs])
            gpu = np.mean([r.get("gpu_s", 0) for r in recs])
            rows.append(dict(run_id=a.run_id, date=datetime.date.today().isoformat(), git_sha=sha,
                             build=meta.get("build", ""), sample_sha=split_sha.get(f"{ds}_{a.split}.txt", ""),
                             platform=a.platform, engine="hf-transformers", dtype=meta["dtype"], model=meta["model"],
                             dataset=ds, split=a.split, n=len(recs), condition=cond, k=meta["k"],
                             max_pixels=meta["max_area"], prompt_id=meta["prompt_id"],
                             metric="accuracy" if not np.isnan(acc) else "", value="" if np.isnan(acc) else round(acc, 4),
                             ci_lo="", ci_hi="", calls=len(recs) * (meta["k"] if cond.endswith("_sample") else 1),
                             tokens=round(tok, 1), latency=round(gpu, 3), vram="", runtime=round(gpu * len(recs), 1),
                             status=a.status, notes=a.notes))
    with open(DB, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header, lineterminator="\n")
        for r in rows:
            w.writerow(r)
    print(f"appended {len(rows)} rows to {DB.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
