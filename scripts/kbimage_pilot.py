"""Pilot (PREREG v6, development data, no model pass): can the pictured entity be identified by matching the question's
image against the lead images of the candidate Wikipedia pages?

  python scripts/kbimage_pilot.py --images <folder written by the fetch script: thumbs.json + img/> [--encoder dinov2|clip]
      -> experiments/results/kbimage_pilot.{json,md}
"""
import argparse
import json
import pathlib
import sys
import urllib.parse

import numpy as np
import pandas as pd
import torch
from PIL import Image

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
import passage_select as PS  # noqa: E402
from vgate import gate  # noqa: E402

ENC = {"dinov2": "facebook/dinov2-base", "clip": "openai/clip-vit-base-patch16"}  # the cached laion ViT-B/16 is an open_clip file that transformers cannot load
TITLE = lambda pid: pid.removeprefix("WikiWeb_").rsplit("_", 1)[0]
N = 20


def embed(name, paths, bs=32):
    """L2-normalised image embeddings of a frozen encoder ({path: vector}); unreadable files are skipped."""
    from transformers import AutoImageProcessor, AutoModel, CLIPModel, CLIPProcessor
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    if name == "clip":
        proc, model = CLIPProcessor.from_pretrained(ENC[name]), CLIPModel.from_pretrained(ENC[name]).to(dev).eval()
        f = lambda px: model.visual_projection(model.vision_model(pixel_values=px).pooler_output)  # the projected image embedding
    else:
        proc, model = AutoImageProcessor.from_pretrained(ENC[name]), AutoModel.from_pretrained(ENC[name]).to(dev).eval()
        f = lambda px: model(pixel_values=px).pooler_output
    out = {}
    for i in range(0, len(paths), bs):
        ims, keep = [], []
        for p in paths[i:i + bs]:
            try:
                ims.append(Image.open(p).convert("RGB")); keep.append(p)
            except Exception:
                pass
        if not ims:
            continue
        with torch.inference_mode():
            v = f(proc(images=ims, return_tensors="pt")["pixel_values"].to(dev)).float().cpu().numpy()
        for p, x in zip(keep, v / np.linalg.norm(v, axis=1, keepdims=True)):
            out[p] = x
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--encoder", default="dinov2,clip")
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--n", type=int, default=N, help="candidate passages per question whose pages are matched (registered: 20)")
    a = ap.parse_args()
    img_dir = pathlib.Path(a.images)
    thumbs = json.loads((img_dir / "thumbs.json").read_text())
    kb_path = {t: img_dir / "img" / (urllib.parse.quote(t, safe="")[:150] + ".jpg") for t, u in thumbs.items() if u}
    kb_path = {t: p for t, p in kb_path.items() if p.exists()}
    dev = set((ROOT / "data/splits/evqa_dev.txt").read_text().split())
    raw = pd.read_parquet(ROOT / "data/raw/evqa_preflmr/data/test-00000-of-00001.parquet"); raw = raw[raw.question_id.isin(dev)].set_index("question_id")
    recs = {r["qid"]: r for r in map(json.loads, open(ROOT / "data/processed/evqa.jsonl")) if r["qid"] in dev}
    qimg = {}
    for dp, _, fs in __import__("os").walk(ROOT / "data/raw/evqa_images"):
        for f_ in fs:
            qimg[f_] = pathlib.Path(dp) / f_
    c, q, _ = PS.candidates(ROOT / "experiments/runs/J10", ROOT / "experiments/runs/J1", ROOT / "data", "evqa", "Qwen3-VL-4B-Instruct", "dev")
    rep = dict(n=len(q), titles=len(thumbs), titles_with_image=len(kb_path), candidates_per_question=a.n)
    z = lambda x: (x - x.mean()) / (x.std() + 1e-9)
    for enc in a.encoder.split(","):
        E_kb = embed(enc, sorted(set(kb_path.values()))); E_q = embed(enc, sorted({qimg[pathlib.Path(recs[i]["image_file"]).name] for i in q.index}))
        page_rows, pick, pick20, top_sim = [], {}, {}, {}
        for i in q.index:
            L = list(raw.at[i, "retrieved_passage"])[:a.n]; gold = set(map(str, raw.at[i, "pos_item_ids"])); gt = {TITLE(g) for g in gold}
            pids, sc = [x["passage_id"] for x in L], np.array([x["score"] for x in L], float)
            t = [TITLE(p) for p in pids]; qv = E_q[qimg[pathlib.Path(recs[i]["image_file"]).name]]
            sim = np.array([float(E_kb[kb_path[x]] @ qv) if x in kb_path and kb_path[x] in E_kb else np.nan for x in t])
            sim = np.where(np.isnan(sim), np.nanmean(sim) if np.isfinite(np.nanmean(sim)) else 0.0, sim)
            fused = z(sc) + z(sim); top_sim[i] = float(sim.max())
            first = {}  # page -> index of its best-ranked passage
            for j, x in enumerate(t):
                first.setdefault(x, j)
            pages = list(first)
            best = lambda s: pages[int(np.argmax([s[first[p]] if s is not sim else max(s[k] for k in range(len(t)) if t[k] == p) for p in pages]))]
            if gt & set(pages):
                page_rows.append(dict(retr=pages[0] in gt, img=best(sim) in gt, fused=best(fused) in gt, has_img=bool(gt & set(kb_path))))
            j5 = int(np.argmax(fused[:5])); pick[i] = j5 + 1                       # choice among the five passages that were read alone
            jb = first[best(fused)]; pick20[i] = jb + 1 if jb < 5 else pick[i]      # best page of the top-20 if its best passage is among the five
        P = pd.DataFrame(page_rows)
        g = c.set_index(["qid", "k"])
        G = __import__("loss_decomposition").gold_info("evqa")["rank"].reindex(q.index)
        acc = lambda sel: np.array([g.at[(i, int(sel[i])), "c"] for i in q.index], float)
        alw = q.c_ev.to_numpy().astype(float)
        boot = lambda v: [round(100 * x, 2) for x in gate.cluster_bootstrap(q.group.to_numpy(), lambda idx: float(v[idx].mean() - alw[idx].mean()), n_boot=a.n_boot, alpha=0.2)]
        in5 = G > 0
        src = pd.Series([recs[i]["image_id"].split("_")[0] for i in q.index], index=q.index)  # landmark photos come from Wikimedia Commons
        by_src = {s_: dict(n=int((src == s_).sum()), near_duplicate=round(100 * float(np.mean([top_sim[i] > 0.95 for i in q.index[src == s_]])), 1),
                           passage_pick_is_gold=round(100 * float(np.mean([pick[i] == G[i] for i in q.index[in5 & (src == s_)]])), 1),
                           retriever_top1_is_gold=round(100 * float(np.mean([G[i] == 1 for i in q.index[in5 & (src == s_)]])), 1),
                           pick_minus_always=round(100 * float((acc(pick) - alw)[(src == s_).to_numpy()].mean()), 2)) for s_ in sorted(set(src))}
        rep[enc] = dict(gold_page_in_top20=round(100 * len(P) / len(q), 1), gold_page_has_image=round(100 * float(P.has_img.mean()), 1),
                        page_top1_retriever=round(100 * float(P.retr.mean()), 1), page_top1_image=round(100 * float(P.img.mean()), 1), page_top1_fused=round(100 * float(P.fused.mean()), 1),
                        passage_pick_is_gold=round(100 * float(np.mean([pick[i] == G[i] for i in q.index[in5]])), 1),
                        acc_always=round(100 * float(alw.mean()), 2), acc_pick=round(100 * float(acc(pick).mean()), 2), pick_minus_always=boot(acc(pick)),
                        acc_pick_top20=round(100 * float(acc(pick20).mean()), 2), pick_top20_minus_always=boot(acc(pick20)), by_source=by_src)
    best_enc = max((e for e in a.encoder.split(",")), key=lambda e: rep[e]["passage_pick_is_gold"])
    rep["go"] = bool(rep[best_enc]["passage_pick_is_gold"] >= 80 and rep[best_enc]["pick_minus_always"][0] >= 3.0 and rep[best_enc]["pick_minus_always"][1] > 0)
    rep["best_encoder"] = best_enc
    L = [f"# Entity identification from candidate page images (PREREG v6 pilot; E-VQA dev, n={rep['n']}; 80% intervals)", "",
         f"Candidate titles {rep['titles']}, with a lead image {rep['titles_with_image']}.", "", "| | " + " | ".join(a.encoder.split(",")) + " |", "|---|" + "---|" * len(a.encoder.split(","))]
    for k, name in (("gold_page_in_top20", "gold page among the candidates (%)"), ("gold_page_has_image", "… of which the gold page has a lead image (%)"),
                    ("page_top1_retriever", "gold page first: retriever order (%)"), ("page_top1_image", "gold page first: image similarity (%)"),
                    ("page_top1_fused", "gold page first: z(retriever) + z(image) (%)"), ("passage_pick_is_gold", "fused pick among the five is the gold passage (%)"),
                    ("acc_always", "accuracy, always-retrieve"), ("acc_pick", "accuracy, fused pick read alone"), ("pick_minus_always", "pick − always"),
                    ("acc_pick_top20", "accuracy, best page of the candidates when its passage is among the five"), ("pick_top20_minus_always", "… − always")):
        L.append(f"| {name} | " + " | ".join(str(rep[e][k]) for e in a.encoder.split(",")) + " |")
    for e in a.encoder.split(","):
        L += ["", f"By image source ({e}): " + "; ".join(f"{s_} (n={v['n']}): near-duplicate of a candidate's lead image {v['near_duplicate']}%, gold pick {v['passage_pick_is_gold']}% "
                                                        f"(retriever {v['retriever_top1_is_gold']}%), pick − always {v['pick_minus_always']:+.2f}" for s_, v in rep[e]["by_source"].items()) + "."]
    L += ["", f"**Go / no-go (PREREG v6): {'GO' if rep['go'] else 'NO-GO'}** (encoder {best_enc}; needs ≥ 80% gold picks and ≥ +3.0 over always-retrieve with the interval above 0)."]
    out = ROOT / "experiments/results/kbimage_pilot"
    out.with_suffix(".json").write_text(json.dumps(rep, indent=1)); out.with_suffix(".md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
