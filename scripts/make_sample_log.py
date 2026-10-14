"""Write demo/sample_log.csv: the held-out outcomes and pre-read signals of the main model on the three benchmarks, in the
format of the demo's bring-your-own-log view (c_ne, c_ev, group, segment, cost_ne, cost_ev, the scalar signals, 64 principal
components of the closed-book hidden state (h0..h63; fitted on the development questions), and `decision`:
the read decisions of the search-need prompt of Chen et al. run with the same reader). No question text, answers or
images are included.

  python scripts/make_sample_log.py
"""
import pathlib
import sys

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
import analyze as A  # noqa: E402
import operating_point as OP  # noqa: E402
from vgate import uncertainty as U  # noqa: E402

M, RUNS = "Qwen3-VL-4B-Instruct", dict(evqa="J4", infoseek="J4", okvqa="J6")
# The 64 hidden-state components of the paper's gates: scaler and PCA fitted on the pooled development questions.
import lodo  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
_, Hdev = lodo.load(M, "dev")
import numpy as np  # noqa: E402
Hdev = np.vstack(list(Hdev.values()))
SC = StandardScaler().fit(Hdev)
PCS = PCA(n_components=64, random_state=0).fit(SC.transform(Hdev))
parts = []
for d, run in RUNS.items():
    s = A.load(ROOT / "experiments/runs" / run, ROOT / "data", d, M, "confirm")
    part = pd.DataFrame(dict(segment=d, group=s.group, c_ne=s.c_ne.astype(int), c_ev=s.c_ev.astype(int), cost_ne=s.tok_ne, cost_ev=s.tok_ev,
                             **{f: s[f].round(5) for f in OP.BASE}))
    h = U.load_hidden(ROOT / "experiments/runs" / run, d, M)
    part[[f"h{j}" for j in range(64)]] = PCS.transform(SC.transform(h[sorted(h)[len(h) // 2]].reindex(s.index).to_numpy())).round(3)
    g = U.read(ROOT / "experiments/runs/J16", d, M, "gate_chen")  # a published gate's decisions (PREREG v8), when it was run
    if g:
        part["decision"] = [int(g[q]["p_retrieve"] > 0.5) for q in s.index]
    parts.append(part)
out = pd.concat(parts, ignore_index=True)
out.to_csv(ROOT / "demo/sample_log.csv", index=False)
print(f"wrote demo/sample_log.csv: {len(out)} rows, columns {list(out.columns)}")
