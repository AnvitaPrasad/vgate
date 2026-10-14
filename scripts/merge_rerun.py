"""Replace failed generations of a run by their rerun (PREREG, note on v7d: J18 returned NaN for 72 questions in half
precision; J18b reran them one per batch in float32).

For every question of the rerun: its rows in <run>/<dataset>__<model>__{ne,ev}.jsonl and its hidden-state rows are replaced
by the rerun's. The run's original files are copied to <run>/failed_fp16/ first. A question whose rerun is still not finite
is left as it is and reported.

  python scripts/merge_rerun.py --run experiments/runs/J18 --fix experiments/runs/J18b --dataset cvqa --model Qwen2.5-VL-3B-Instruct
"""
import argparse
import glob
import json
import math
import pathlib
import shutil

import numpy as np


def finite(r):
    return all(isinstance(x, (int, float)) and math.isfinite(x) for x in r.get("tok_logprobs") or [float("nan")])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True); ap.add_argument("--fix", required=True)
    ap.add_argument("--dataset", required=True); ap.add_argument("--model", required=True)
    a = ap.parse_args()
    run, fix, stem = pathlib.Path(a.run), pathlib.Path(a.fix), f"{a.dataset}__{a.model}"
    keep = run / "failed_fp16"
    assert not keep.exists(), "already merged"
    keep.mkdir()
    for f in list(run.glob(f"{stem}__*.jsonl")) + list(run.glob(f"hidden_{stem}__*")):
        shutil.copy2(f, keep / f.name)
    new = {c: {r["qid"]: r for r in map(json.loads, open(fix / f"{stem}__{c}.jsonl"))} for c in ("ne", "ev")}
    ok = sorted(q for q in new["ne"] if q in new["ev"] and finite(new["ne"][q]) and finite(new["ev"][q]))
    print("rerun questions:", len(new["ne"]), "| finite in both passes:", len(ok))
    for c in ("ne", "ev"):
        p = run / f"{stem}__{c}.jsonl"
        rows = [json.loads(l) for l in open(p)]
        n = sum(r["qid"] in ok for r in rows)
        p.write_text("".join(json.dumps(new[c][r["qid"]] if r["qid"] in ok else r, ensure_ascii=False) + "\n" for r in rows))
        print(f"{p.name}: {n} rows replaced; still not finite: {sum(not finite(json.loads(l)) for l in open(p))}")
    fq = json.loads((fix / f"hidden_{stem}__qids0.json").read_text()); pos = {q: i for i, q in enumerate(fq)}
    for qf in sorted(run.glob(f"hidden_{stem}__qids*.json")):
        part = qf.stem.split("qids")[-1]; qids = json.loads(qf.read_text())
        for hf in sorted(glob.glob(str(run / f"hidden_{stem}__L*_part{part}.npy"))):
            layer = pathlib.Path(hf).name.split("__")[-1].split("_part")[0]
            H, F = np.load(hf), np.load(fix / f"hidden_{stem}__{layer}_part0.npy")
            idx = [i for i, q in enumerate(qids) if q in ok and q in pos]
            H[idx] = F[[pos[qids[i]] for i in idx]].astype(H.dtype)
            np.save(hf, H)
            print(f"{pathlib.Path(hf).name}: {len(idx)} rows replaced; rows still not finite: {int((~np.isfinite(H).all(1)).sum())}")


if __name__ == "__main__":
    main()
