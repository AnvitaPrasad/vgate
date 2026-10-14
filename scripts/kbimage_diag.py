"""EXPLORATORY diagnosis of the candidate-image pilot (PREREG v6 was NO-GO): where exactly is the loss?
Development split, E-VQA, five candidates per question, DINOv2-base. Nothing here is a registered result; the
"best threshold / best weight" numbers are in-sample upper bounds."""
import json
import pathlib
import sys
import urllib.parse

import numpy as np
import pandas as pd

ROOT = pathlib.Path(".").resolve()
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
import kbimage_pilot as KP  # noqa: E402
import loss_decomposition as LD  # noqa: E402
import passage_select as PS  # noqa: E402

IMG = pathlib.Path(sys.argv[1]); CACHE = IMG / "emb_dinov2.npz"
thumbs = json.loads((IMG / "thumbs.json").read_text())
kb_path = {t: IMG / "img" / (urllib.parse.quote(t, safe="")[:150] + ".jpg") for t, u in thumbs.items() if u}
kb_path = {t: p for t, p in kb_path.items() if p.exists()}
dev = set((ROOT / "data/splits/evqa_dev.txt").read_text().split())
raw = pd.read_parquet(ROOT / "data/raw/evqa_preflmr/data/test-00000-of-00001.parquet"); raw = raw[raw.question_id.isin(dev)].set_index("question_id")
recs = {r["qid"]: r for r in map(json.loads, open(ROOT / "data/processed/evqa.jsonl")) if r["qid"] in dev}
qimg = {}
import os
for dp, _, fs in os.walk(ROOT / "data/raw/evqa_images"):
    for f_ in fs:
        qimg[f_] = pathlib.Path(dp) / f_
c, q, _ = PS.candidates(ROOT / "experiments/runs/J10", ROOT / "experiments/runs/J1", ROOT / "data", "evqa", "Qwen3-VL-4B-Instruct", "dev")
qpaths = {i: qimg[pathlib.Path(recs[i]["image_file"]).name] for i in q.index}
if CACHE.exists():
    z_ = np.load(CACHE, allow_pickle=True); E = dict(zip(z_["k"].tolist(), z_["v"]))
else:
    paths = sorted(set(map(str, kb_path.values())) | set(map(str, qpaths.values())))
    E = {str(k): v for k, v in KP.embed("dinov2", [pathlib.Path(p) for p in paths]).items()}
    np.savez(CACHE, k=np.array(list(E)), v=np.stack(list(E.values())))
T = KP.TITLE
G = LD.gold_info("evqa")["rank"].reindex(q.index)            # rank of the gold passage among the five (0: not among them)
g = c.set_index(["qid", "k"]).c                                # correctness reading passage k alone
alw, nev = q.c_ev.to_numpy().astype(float), q.c_ne.to_numpy().astype(float)
zf = lambda x: (x - x.mean()) / (x.std() + 1e-9)
rows = []
for i in q.index:
    L20 = list(raw.at[i, "retrieved_passage"])[:20]; L = L20[:5]
    gold = set(map(str, raw.at[i, "pos_item_ids"])); gt = {T(x) for x in gold}
    t = [T(x["passage_id"]) for x in L]; sc = np.array([x["score"] for x in L], float)
    qv = E[str(qpaths[i])]
    sim = np.array([float(E[str(kb_path[x])] @ qv) if x in kb_path and str(kb_path[x]) in E else np.nan for x in t])
    sim = np.where(np.isnan(sim), np.nanmean(sim) if np.isfinite(np.nanmean(sim)) else 0.0, sim)
    t20 = [T(x["passage_id"]) for x in L20]
    rows.append(dict(qid=i, gold_rank=int(G[i]), n_gold_page_passages=sum(x in gt for x in t), gold_page_in5=bool(gt & set(t)),
                     gold_page_rank20=next((j + 1 for j, x in enumerate(t20) if x in gt), 0), gold_psg_rank20=next((j + 1 for j, x in enumerate(L20) if x["passage_id"] in gold), 0),
                     t=t, zs=zf(sc), zi=zf(sim), sim=sim, gt=gt, n_pages=len(set(t)), top_sim=float(sim.max()), margin=float(np.sort(sim)[-1] - np.sort(sim)[-2])))
D = pd.DataFrame(rows).set_index("qid")
n = len(D)
print(f"n = {n}; gold passage among the five: {100 * (D.gold_rank > 0).mean():.1f}%; gold PAGE among the five: {100 * D.gold_page_in5.mean():.1f}%")
print(f"gold page within PreFLMR's top-5 / 10 / 20 passages: " + " / ".join(f"{100 * ((D.gold_page_rank20 > 0) & (D.gold_page_rank20 <= k)).mean():.1f}%" for k in (5, 10, 20))
      + f"; gold passage within top-5 / 10 / 20: " + " / ".join(f"{100 * ((D.gold_psg_rank20 > 0) & (D.gold_psg_rank20 <= k)).mean():.1f}%" for k in (5, 10, 20)))
print(f"distinct pages among the five passages: mean {D.n_pages.mean():.2f}; passages of the gold page among the five (when present): mean {D.n_gold_page_passages[D.gold_page_in5].mean():.2f}")


def pick(score):
    return {i: int(np.argmax(s)) + 1 for i, s in score.items()}


def report(name, sel):
    page_ok = np.array([D.at[i, "t"][sel[i] - 1] in D.at[i, "gt"] for i in D.index]); psg_ok = np.array([sel[i] == D.at[i, "gold_rank"] for i in D.index])
    acc = np.array([g[(i, sel[i])] for i in D.index], float)
    a, b = D.gold_page_in5.to_numpy(), (D.gold_rank > 0).to_numpy()
    print(f"{name:42s} page right {100 * page_ok[a].mean():5.1f}% (of gold-page-in-5) | passage right {100 * psg_ok[b].mean():5.1f}% (of gold-passage-in-5) | "
          f"read alone {100 * acc.mean():5.1f} (always {100 * alw.mean():.1f})")
    return page_ok, psg_ok, acc


fused = {i: D.at[i, "zs"] + D.at[i, "zi"] for i in D.index}
p_r, s_r, a_r = report("retriever rank 1", {i: 1 for i in D.index})
p_i, s_i, a_i = report("image similarity (ties -> retriever)", pick({i: D.at[i, "zi"] + 1e-6 * D.at[i, "zs"] for i in D.index}))
p_f, s_f, a_f = report("z(retriever) + z(image)  [registered]", pick(fused))
# where does the registered pick lose? split of the 1,000 questions
sel = pick(fused); b = (D.gold_rank > 0).to_numpy(); a = D.gold_page_in5.to_numpy()
cases = {"gold passage picked": s_f & b, "gold page picked, wrong section": p_f & ~s_f & a, "gold page among five, another page picked": a & ~p_f,
         "gold page not among the five": ~a}
print("\nWhere the registered pick stands, and what reading it alone gives against reading all five:")
for k, m in cases.items():
    print(f"  {k:44s} {100 * m.mean():5.1f}% of questions | pick alone {100 * a_f[m].mean():5.1f} | all five {100 * alw[m].mean():5.1f} | closed-book {100 * nev[m].mean():5.1f}")
# upper bounds
orc_page = {i: (next(j + 1 for j, x in enumerate(D.at[i, "t"]) if x in D.at[i, "gt"]) if D.at[i, "gold_page_in5"] else 1) for i in D.index}
orc_psg = {i: (D.at[i, "gold_rank"] if D.at[i, "gold_rank"] > 0 else 1) for i in D.index}
print("\nUpper bounds (labels used; not achievable):")
report("oracle PAGE, first section by retriever", orc_page); report("oracle PASSAGE", orc_psg)
best_single = np.array([max(g[(i, k)] for k in range(1, 6)) for i in D.index], float)
print(f"best single passage of the five: {100 * best_single.mean():.1f}; best of (all five, closed-book): {100 * np.maximum(alw, nev).mean():.1f}")
# in-sample: fusion weight, and "use the pick only when the image match is confident, else read all five"
print("\nIn-sample sweeps (optimistic):")
for w in (0, 0.5, 1, 2, 4, 8):
    s_ = pick({i: D.at[i, "zs"] + w * D.at[i, "zi"] for i in D.index})
    ok = np.array([s_[i] == D.at[i, "gold_rank"] for i in D.index]); ac = np.array([g[(i, s_[i])] for i in D.index], float)
    print(f"  weight {w:>3}: passage right {100 * ok[b].mean():5.1f}% | read alone {100 * ac.mean():5.1f}")
for col in ("top_sim", "margin"):
    v = D[col].to_numpy(); best = max(((np.where(v >= th, a_f, alw).mean(), th, (v >= th).mean()) for th in np.quantile(v, np.linspace(0, 1, 41))), key=lambda x: x[0])
    print(f"  use the pick only when {col} >= {best[1]:.3f} ({100 * best[2]:.0f}% of questions), else all five: {100 * best[0]:.1f} (always {100 * alw.mean():.1f})")
# is the image match right when it is confident?
for lo, hi in ((0, .25), (.25, .5), (.5, .75), (.75, 1.01)):
    v = D.margin.to_numpy(); m = (v >= np.quantile(v, lo)) & (v <= np.quantile(v, min(hi, 1))) & a
    print(f"  margin quartile {lo:.2f}-{hi:.2f}: image pick is the gold page {100 * p_i[m].mean():5.1f}% (n={int(m.sum())})")
src = pd.Series([recs[i]["image_id"].split("_")[0] for i in D.index], index=D.index).to_numpy()
for s_ in sorted(set(src)):
    m = (src == s_) & a
    print(f"  {s_}: gold page first by retriever {100 * p_r[m].mean():.1f}%, by image {100 * p_i[m].mean():.1f}%, fused {100 * p_f[m].mean():.1f}% (n={int(m.sum())})")

# --- two variants that were NOT run, estimated from the logs (observational; a real test needs a new reader pass)
first, later = alw[(D.gold_rank == 1).to_numpy()].mean(), alw[(D.gold_rank > 1).to_numpy()].mean()
print(f"\nReading all five: right {100 * first:.1f}% when the gold passage is ranked first, {100 * later:.1f}% when it is at ranks 2-5 "
      f"(n = {int((D.gold_rank == 1).sum())} / {int((D.gold_rank > 1).sum())}).")
gain_flow, loss_flow = (s_f & ~s_r & b).mean(), (s_r & ~s_f & b).mean()
print(f"Reordering (picked passage first, all five kept): the gold passage would move to rank 1 for {100 * gain_flow:.1f}% of questions and leave it for "
      f"{100 * loss_flow:.1f}%; if rank alone drove the gap above, about {100 * (gain_flow - loss_flow) * (first - later):+.1f} points.")
out5 = ~a
in20 = ((D.gold_page_rank20 > 5)).to_numpy()
print(f"Re-ranking the top-20: the gold page is at ranks 6-20 for {100 * in20.mean():.1f}% of questions (it is never read today); when the gold page is among the "
      f"five read, all-five accuracy is {100 * alw[a].mean():.1f}, when it is not, {100 * alw[out5].mean():.1f}. Every 10 such questions moved into the five would be worth "
      f"about {10 * (alw[a].mean() - alw[out5].mean()):.1f} points if they then behaved like the others.")
have20 = np.mean([np.mean([T(x['passage_id']) in kb_path for x in list(raw.at[i, 'retrieved_passage'])[5:20]]) for i in D.index])
print(f"Share of rank 6-20 candidates with an image on disk: {100 * have20:.0f}%.")
