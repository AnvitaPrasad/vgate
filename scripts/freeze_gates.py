"""Freeze the mixed-traffic gates (VGate-pre, VGate-full) of the paper into plain arrays for the live pipeline.

Fits exactly the objects of scripts/operating_point.py (`fit_benefit_gates` on the pooled development sets, no dataset
identity) and writes src/vgate/gates/<model>.npz. Self-checks before writing:
  1. the array implementation in vgate.live.Gates reproduces scikit-learn's scores on the pooled held-out sets;
  2. the read decisions give the macro accuracies stored in experiments/results/operating_point__<model>.json.

  python scripts/freeze_gates.py --model Qwen3-VL-4B-Instruct --runs evqa=J1:J4,infoseek=J1:J4,okvqa=J6:J6
"""
import argparse
import json
import pathlib
import sys

import numpy as np
import pandas as pd
import sklearn

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
import analyze as A  # noqa: E402
import operating_point as OP  # noqa: E402
from vgate import uncertainty as U  # noqa: E402
from vgate.live import Gates  # noqa: E402


def export(G, layer, n_train, out):
    """Write the fitted pieces of `operating_point.fit_benefit_gates` as plain arrays (read by vgate.live.Gates)."""
    arr = dict(base_names=np.array(OP.BASE), layer=np.array(layer), n_train=np.array(n_train),
               h_mean=G["sc"].mean_, h_scale=G["sc"].scale_, pca_mean=G["pca"].mean_, pca_components=G["pca"].components_)
    for name in ("pre", "full"):
        sc, lr = G[name][0], G[name][-1]
        arr.update({f"{name}_mean": sc.mean_, f"{name}_scale": sc.scale_, f"{name}_coef": lr.coef_,
                    f"{name}_intercept": lr.intercept_, f"{name}_classes": lr.classes_})
    out = pathlib.Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **arr)
    return Gates(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen3-VL-4B-Instruct")
    ap.add_argument("--runs", default="evqa=J1:J4,infoseek=J1:J4,okvqa=J6:J6")
    a = ap.parse_args()
    spec = {x.split("=")[0]: x.split("=")[1].split(":") for x in a.runs.split(",")}
    R, tr, te, Htr, Hte = ROOT / "experiments/runs", {}, {}, {}, {}
    for d, (r_tr, r_te) in spec.items():
        tr[d], te[d] = A.load(R / r_tr, ROOT / "data", d, a.model, "dev"), A.load(R / r_te, ROOT / "data", d, a.model, "confirm")
        H1, H2 = U.load_hidden(R / r_tr, d, a.model), U.load_hidden(R / r_te, d, a.model)
        L = sorted(H1)[len(H1) // 2]
        Htr[d], Hte[d] = H1[L].reindex(tr[d].index).to_numpy(), H2[L].reindex(te[d].index).to_numpy()
    ptr, pte = pd.concat(tr.values()), pd.concat(te.values())
    Hp_tr, Hp_te = np.vstack(list(Htr.values())), np.vstack(list(Hte.values()))
    G = OP.fit_benefit_gates(ptr, Hp_tr)
    out = ROOT / "src/vgate/gates" / f"{a.model}.npz"
    g = export(G, L, len(ptr), out)
    # --- self-checks
    P_te = G["pca"].transform(G["sc"].transform(Hp_te))
    X_te = np.nan_to_num(pte[OP.BASE].to_numpy())
    sk_pre, sk_full = OP.benefit_score(G["pre"], P_te), OP.benefit_score(G["full"], np.hstack([X_te, P_te]))
    np_pre, np_full = g.pre(Hp_te), g.full(pte[OP.BASE].to_numpy(), Hp_te)
    d_pre, d_full = float(np.abs(sk_pre - np_pre).max()), float(np.abs(sk_full - np_full).max())
    assert d_pre < 1e-4 and d_full < 1e-4, (d_pre, d_full)  # scikit-learn ran in float32 (hidden states are stored as float16)
    ref = json.load(open(ROOT / f"experiments/results/operating_point__{a.model}.json"))["mixed_traffic_gate"]
    offs, check = np.cumsum([0] + [len(te[d]) for d in spec]), {}
    for rule, s in (("benefit_rule_pre", np_pre), ("benefit_rule_full", np_full)):
        acc = [float(np.where(s[lo:hi] > 0, te[d].c_ev, te[d].c_ne).mean()) for d, lo, hi in zip(spec, offs[:-1], offs[1:])]
        check[rule] = float(np.mean(acc))
        assert abs(check[rule] - ref["macro"][rule]["acc"]) < 1e-9, (rule, check[rule], ref["macro"][rule]["acc"])
    meta = dict(model=a.model, runs=spec, layer=int(L), n_train=int(len(ptr)), base=OP.BASE, sklearn=sklearn.__version__,
                max_abs_diff_vs_sklearn=dict(pre=d_pre, full=d_full), heldout_macro_accuracy=check,
                note="mixed-traffic gates of scripts/operating_point.py (pooled dev sets, no dataset identity)")
    out.with_suffix(".json").write_text(json.dumps(meta, indent=1))
    # pooled-gate scores of the held-out queries, to compare with what the live pipeline recomputes
    pd.DataFrame(dict(u_pre=np_pre, u_full=np_full, a0=pte.ne_text.to_numpy(), aE=pte.ev_text.to_numpy(),
                      c_ne=pte.c_ne.to_numpy(), c_ev=pte.c_ev.to_numpy()), index=pte.index).to_csv(
        ROOT / f"experiments/results/frozen_gate_scores__{a.model}.csv")
    print(f"wrote {out} ({out.stat().st_size / 1e6:.2f} MB); numpy vs sklearn max |diff|: pre {d_pre:.1e}, full {d_full:.1e}; "
          f"held-out macro accuracy: {check}")


if __name__ == "__main__":
    main()
