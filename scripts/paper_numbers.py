"""Generate paper/tables/numbers.tex: every number quoted in the paper text, as LaTeX macros, computed from run
artifacts. Nothing is typed by hand; re-run after any analysis change.

  python scripts/paper_numbers.py
"""
import json
import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT / "src"))
import analyze as A  # noqa: E402
from vgate import gate  # noqa: E402

M1, M2 = "Qwen3-VL-4B-Instruct", "Qwen2.5-VL-3B-Instruct"
R = {"J4": ROOT / "experiments/runs/J4", "J2": ROOT / "experiments/runs/J2", "J1": ROOT / "experiments/runs/J1"}
mac = {}


def put(name, value, fmt="{:.1f}"):
    mac[name] = fmt.format(value) if not isinstance(value, str) else value


def pct(x):
    return 100 * x


c1 = json.load(open(R["J4"] / f"confirm__{M1}.json"))
c1s = json.load(open(R["J4"] / f"confirm__{M1}__strict.json"))
c2 = json.load(open(R["J2"] / f"confirm__{M2}.json"))
c3 = json.load(open(ROOT / "experiments/runs/J5/confirm__InternVL3_5-4B-HF.json"))
for tag, d in [("E", "evqa"), ("I", "infoseek")]:
    q = c1[d]["quadrants"]
    put(f"nTest{tag}", f"{c1[d]['n_test']:,}")
    put(f"help{tag}", pct(q["help"][0])); put(f"harm{tag}", pct(q["harm"][0]))
    put(f"never{tag}", pct(q["acc_never"][0])); put(f"always{tag}", pct(q["acc_always"][0]))
    put(f"oracleGate{tag}", pct(q["acc_oracle_gate"][0]))
    put(f"neverTok{tag}", pct(c1[d]["never_rel_tokens"]), "{:.0f}")
    P = c1[d]["policies"]
    put(f"readFact{tag}", pct(P["factorised_post"]["read_rate_at_always_acc"]), "{:.0f}")
    put(f"readBen{tag}", pct(P["benefit_post"]["read_rate_at_always_acc"]), "{:.0f}")
    put(f"searchPre{tag}", pct(P["probe_benefit"]["read_rate_at_always_acc"]), "{:.0f}")   # pre-search: read = search
    put(f"searchEnt{tag}", pct(P["signal_ent_mean"]["read_rate_at_always_acc"]), "{:.0f}")
    put(f"gainBen{tag}", pct(P["benefit_post"]["gain_recovered_at_50"]), "{:.0f}")
    put(f"gainPre{tag}", pct(P["probe_benefit"]["gain_recovered_at_50"]), "{:.0f}")
    put(f"gainErr{tag}", pct(P["error_post"]["gain_recovered_at_50"]), "{:.0f}")
    P2 = c2[d]["policies"]
    put(f"searchPreMtwo{tag}", pct(P2["probe_benefit"]["read_rate_at_always_acc"]), "{:.0f}")
    put(f"primary{tag}", pct(c1["per_dataset_dnauc_95ci"][d]["benefit_post - error_post"][0]), "{:+.2f}")
    df = A.load(R["J4"], ROOT / "data", d, M1, "confirm")
    put(f"fixedGivenWrong{tag}", pct(((df.c_ne == 0) & (df.c_ev == 1)).sum() / (df.c_ne == 0).sum()), "{:.0f}")
    put(f"brokenGivenRight{tag}", pct(((df.c_ne == 1) & (df.c_ev == 0)).sum() / (df.c_ne == 1).sum()), "{:.0f}")
import re as _re
sys.path.insert(0, str(ROOT / "demo"))
from app import failure_mode  # noqa: E402  (same classifier the demo's harm auditor uses)
recs_all, modes = {}, []
for d in ("evqa", "infoseek"):
    recs = {r["qid"]: r for r in map(json.loads, open(ROOT / f"data/processed/{d}.jsonl"))}
    df = A.load(R["J4"], ROOT / "data", d, M1, "confirm")
    for q, row in df[df.u == -1].iterrows():
        modes.append(failure_mode(row.get("ev_text", ""), recs[q]["evidence"]))
    put(f"neTok{d[0].upper()}", round(df.tok_ne.mean(), -1), "{:.0f}"); put(f"evTok{d[0].upper()}", round(df.tok_ev.mean(), -1), "{:.0f}")
mc = pd.Series(modes).value_counts()
put("harmN", len(modes), "{:d}")
put("harmCapture", 100 * mc.get("distractor-entity capture", 0) / len(modes), "{:.0f}")
put("harmAbstain", 100 * mc.get("evidence-induced abstention", 0) / len(modes), "{:.0f}")
gv = [float(mac[k]) for k in ("gainBenE", "gainBenI", "gainPreE", "gainPreI")]
ge = [float(mac[k]) for k in ("gainErrE", "gainErrI")]
put("gainVgLo", min(gv), "{:.0f}"); put("gainVgHi", max(gv), "{:.0f}")
put("gainErrLo", min(ge), "{:.0f}"); put("gainErrHi", max(ge), "{:.0f}")
toks = [min(v["rel_token_cost_at_always_acc"] for k, v in c1[d]["policies"].items() if k != "oracle" and
            v["rel_token_cost_at_always_acc"] == v["rel_token_cost_at_always_acc"]) for d in ("evqa", "infoseek")]
put("minTokCost", min(toks), "{:.2f}")


def ci(rep, key, name, scale=100):
    p, lo, hi = rep["pooled_dnauc_95ci"][key]
    put(name, scale * p, "{:+.2f}"); put(name + "Lo", scale * lo, "{:+.2f}"); put(name + "Hi", scale * hi, "{:+.2f}")


ci(c1, "benefit_post - error_post", "primaryMone"); ci(c2, "benefit_post - error_post", "primaryMtwo")
ci(c1s, "benefit_post - error_post", "primaryStrict")
ci(c1, "benefit_pre - error_pre", "preMone"); ci(c2, "benefit_pre - error_pre", "preMtwo")
ci(c1, "benefit_post - factorised_post", "vsFactMone"); ci(c2, "benefit_post - factorised_post", "vsFactMtwo")
ci(c1, "benefit_post - error_pre", "vsErrPreMone")
ci(c1, "factorised_post - error_post", "factVsErrMone")
ci(c1, "probe_benefit - probe_error", "probeMone"); ci(c2, "probe_benefit - probe_error", "probeMtwo")
ci(c3, "benefit_post - error_post", "primaryMthree"); ci(c3, "probe_benefit - probe_error", "probeMthree")
ci(c3, "benefit_pre - error_pre", "preMthree")
ci(c1, "benefit_post - signal_qpp_nqc", "vsNqcMone"); ci(c1, "signal_qpp_nqc - error_post", "nqcVsErrMone")
for tagm, rep in [("Mone", c1), ("Mtwo", c2)]:
    pp, plo, phi = rep["pooled_pauc70_95ci_exploratory"]["benefit_post - error_post"]
    put(f"paucPrim{tagm}", 100 * pp, "{:+.2f}"); put(f"paucPrim{tagm}Lo", 100 * plo, "{:+.2f}"); put(f"paucPrim{tagm}Hi", 100 * phi, "{:+.2f}")
fp, flo, fhi = c1["pooled_dnauc_95ci"]["benefit_post - factorised_post"]  # factorised lead = negated, bounds swapped
put("factLeadMone", -100 * fp, "{:.2f}"); put("factLeadMoneLo", -100 * fhi, "{:.2f}"); put("factLeadMoneHi", -100 * flo, "{:.2f}")
err_gain = np.mean([c1[d]["policies"]["error_post"]["nauc"] - c1[d]["policies"]["error_pre"]["nauc"] for d in ("evqa", "infoseek")])
put("errAddQpp", 100 * err_gain, "{:+.2f}")

sig = pd.read_csv(ROOT / "experiments/results/signal_auroc_confirm.csv")
unc, qpp = sig[sig.signal.isin(["ent_mean", "lp_min", "seq_lp"])], sig[sig.signal.str.startswith("qpp")]
put("uncErrLo", unc.err.min(), "{:.2f}"); put("uncErrHi", unc.err.max(), "{:.2f}")
put("uncHelpLo", unc.help.min(), "{:.2f}"); put("uncHelpHi", unc.help.max(), "{:.2f}")
put("qppErrHi", qpp.err.max(), "{:.2f}"); put("qppHelpHi", qpp.help.max(), "{:.2f}")

for tag, ds in [("E", "evqa"), ("I", "infoseek")]:
    a = pd.read_csv(ROOT / f"experiments/audits/harm_audit_{ds}_J1.csv")
    put(f"auditReal{tag}", int((a.verdict == "real").sum()), "{:d}"); put(f"auditN{tag}", len(a), "{:d}")
for tag, ds in [("E", "evqa"), ("I", "infoseek")]:
    h = pd.read_csv(ROOT / f"experiments/audits/help_audit_{ds}_J1.csv")
    put(f"helpAuditReal{tag}", int((h.verdict == "real").sum()), "{:d}"); put(f"helpAuditN{tag}", len(h), "{:d}")
v2 = {d: pd.read_csv(R["J2"] / f"scores__{d}__{M2}.csv", index_col="qid").c_ne.mean() for d in ("evqa", "infoseek")}
put("reproE", pct(v2["evqa"])); put("reproI", pct(v2["infoseek"]))

c2s = json.load(open(R["J1"].parent / "J3" / "c2_summary.json"))
for tag, d in [("E", "evqa"), ("I", "infoseek")]:
    put(f"thinkFix{tag}", 100 * c2s[d]["think_acc_on_ne_wrong"], "{:.0f}")
    put(f"evFixDev{tag}", 100 * c2s[d]["ev_acc_on_ne_wrong"], "{:.0f}")
    put(f"thinkBreak{tag}", 100 * c2s[d]["think_breaks_correct_ne"], "{:.1f}")
th = {d: [json.loads(l) for l in open(R["J1"].parent / "J3" / f"{d}__Qwen3-VL-4B-Thinking__think.jsonl")] for d in ("evqa", "infoseek")}
ev = {d: [json.loads(l) for l in open(R["J4"] / f"{d}__Qwen3-VL-4B-Instruct__ev.jsonl")] for d in ("evqa", "infoseek")}
ratio = [np.mean([r["gpu_s"] for r in th[d]]) / np.mean([r["gpu_s"] for r in ev[d]]) for d in ("evqa", "infoseek")]
put("thinkCostLo", min(ratio), "{:.0f}"); put("thinkCostHi", max(ratio), "{:.0f}")

bgr = json.load(open(R["J1"].parent / "J3" / "c2_breaks_given_right.json"))
for tag, d in [("E", "evqa"), ("I", "infoseek")]:
    put(f"thinkBreakRight{tag}", 100 * bgr[d]["think_breaks_given_right"], "{:.0f}")
    put(f"evBreakRightDev{tag}", 100 * bgr[d]["ev_breaks_given_right"], "{:.0f}")
cas = json.load(open(R["J4"] / f"cascade__{M1}.json"))
for tag, d in [("E", "evqa"), ("I", "infoseek")]:
    for b, bt in [("0.9", "Hi"), ("0.5", "Lo")]:
        c = cas.get(f"{d}_{b}_full", cas[f"{d}_{b}"])  # the demo's cascade: VGate-pre -> VGate-full
        put(f"cas{bt}Read{tag}", 100 * c["reads"], "{:.0f}"); put(f"cas{bt}Acc{tag}", 100 * c["acc"])
        put(f"cas{bt}Tok{tag}", c["rel_tokens"], "{:.2f}"); put(f"cas{bt}Gap{tag}", 100 * (c["acc_always"] - c["acc"]))
for tag, d in [("E", "evqa"), ("I", "infoseek")]:
    P = c1[d]["policies"]
    put(f"gainProbeErr{tag}", pct(P["probe_error"]["gain_recovered_at_50"]), "{:.0f}")
    put(f"readEnt{tag}", pct(P["signal_ent_mean"]["read_rate_at_always_acc"]), "{:.0f}")
    put(f"readNqc{tag}", pct(P["signal_qpp_nqc"]["read_rate_at_always_acc"]), "{:.0f}")
    put(f"readErrPost{tag}", pct(P["error_post"]["read_rate_at_always_acc"]), "{:.0f}")
put("qppErrLo", qpp.err.min(), "{:.2f}")
put("maxTokSave", 100 * max(0.0, 1 - float(mac["minTokCost"])), "{:.0f}")
for tagm, rep in [("Mone", c1), ("Mtwo", c2)]:
    pp, plo, phi = rep["pooled_pauc70_95ci_exploratory"]["benefit_pre - error_pre"]
    put(f"paucPre{tagm}", 100 * pp, "{:+.2f}"); put(f"paucPre{tagm}Lo", 100 * plo, "{:+.2f}"); put(f"paucPre{tagm}Hi", 100 * phi, "{:+.2f}")

# --- PREREG v2b extensions (scripts/extend.py): VGate-full, four-outcome probe, VGate-select
EXT = {"Mone": R["J4"] / f"extend__{M1}.json", "Mtwo": R["J2"] / f"extend__{M2}.json",
       "Mthree": ROOT / "experiments/runs/J5/extend__InternVL3_5-4B-HF.json",
       "Mfour": ROOT / "experiments/runs/J7/extend__Qwen3-VL-8B-Instruct.json"}
c4p = ROOT / "experiments/runs/J7/confirm__Qwen3-VL-8B-Instruct.json"  # model-4 replication (PREREG v2c)
if c4p.exists():
    c4 = json.load(open(c4p))
    ci(c4, "benefit_post - error_post", "primaryMfour"); ci(c4, "probe_benefit - probe_error", "probeMfour")
    ci(c4, "benefit_pre - error_pre", "preMfour")
    for tag, d in [("E", "evqa"), ("I", "infoseek")]:
        put(f"neverMfour{tag}", pct(c4[d]["quadrants"]["acc_never"][0])); put(f"alwaysMfour{tag}", pct(c4[d]["quadrants"]["acc_always"][0]))
ext = {k: json.load(open(v)) for k, v in EXT.items() if v.exists()}
sel = []
for tagm, e in ext.items():
    ci(e, "vgate_full - benefit_post", f"fullVsPost{tagm}"); ci(e, "vgate_full - factorised_post", f"fullVsFact{tagm}")
    ci(e, "vgate_full - probe_benefit", f"fullVsPre{tagm}"); ci(e, "vgate_full - error_post", f"fullVsErr{tagm}")
    ci(e, "probe_benefit - probe4", f"probeVsFour{tagm}")
    for tag, d in [("E", "evqa"), ("I", "infoseek")]:
        pt, lo, hi = e[d]["select_minus_always_95ci"]
        sel.append((pt, lo, hi))
        put(f"sel{tagm}{tag}", 100 * pt, "{:+.2f}"); put(f"sel{tagm}{tag}Lo", 100 * lo, "{:+.2f}"); put(f"sel{tagm}{tag}Hi", 100 * hi, "{:+.2f}")
if ext:
    put("probeVsFourMax", 100 * max(abs(e["pooled_dnauc_95ci"]["probe_benefit - probe4"][0]) for e in ext.values()), "{:.2f}")
    put("selLo", 100 * min(x[0] for x in sel), "{:+.1f}"); put("selHi", 100 * max(x[0] for x in sel), "{:+.1f}")
    put("selSig", sum(x[1] > 0 for x in sel), "{:d}"); put("selN", len(sel), "{:d}")
    head = [e[d]["acc_oracle_select"] - e[d]["acc_always"] for e in ext.values() for d in ("evqa", "infoseek")]
    put("selHeadLo", 100 * min(head), "{:.0f}"); put("selHeadHi", 100 * max(head), "{:.0f}")
    for tag, d in [("E", "evqa"), ("I", "infoseek")]:
        P = ext["Mone"][d]["policies"]["vgate_full"]
        put(f"readFull{tag}", pct(P["read_rate_at_always_acc"]), "{:.0f}")
        put(f"gainFull{tag}", pct(P["gain_recovered_at_50"]), "{:.0f}")
    toks_full = min(ext["Mone"][d]["policies"]["vgate_full"]["rel_token_cost_at_always_acc"] for d in ("evqa", "infoseek"))
    put("maxTokSave", 100 * max(0.0, 1 - min(float(mac["minTokCost"]), toks_full)), "{:.0f}")  # now includes VGate-full

le = pd.read_csv(ROOT / "experiments/results/label_efficiency.csv")
for tag, d in [("E", "evqa"), ("I", "infoseek")]:
    x = le[(le.dataset == d)]
    full = x[(x.gate == "benefit_post") & (x.n == x.n.max())].nauc.iloc[0]
    put(f"leGap{tag}", 100 * (full - x[(x.gate == "benefit_post") & (x.n == 200)].nauc.iloc[0]), "{:.1f}")
    put(f"leVsErr{tag}", 100 * (x[(x.gate == "benefit_post") & (x.n == 200)].nauc.iloc[0] - x[x.gate == "error_post"].nauc.iloc[0]), "{:+.1f}")

# --- OK-VQA (PREREG v2, run J6) and operating points / mixed traffic (PREREG v2e, exploratory)
J6 = ROOT / "experiments/runs/J6"
if (J6 / f"confirm__{M1}.json").exists():
    o1, o2 = json.load(open(J6 / f"confirm__{M1}.json")), json.load(open(J6 / f"confirm__{M2}.json"))
    x1, x2 = json.load(open(J6 / f"extend__{M1}.json")), json.load(open(J6 / f"extend__{M2}.json"))
    q = o1["okvqa"]["quadrants"]
    put("nTestO", f"{o1['okvqa']['n_test']:,}")
    put("helpO", pct(q["help"][0])); put("harmO", pct(q["harm"][0])); put("neverO", pct(q["acc_never"][0]))
    put("alwaysO", pct(q["acc_always"][0])); put("oracleGateO", pct(q["acc_oracle_gate"][0]))
    put("oracleHeadO", pct(q["acc_oracle_gate"][0] - max(q["acc_always"][0], q["acc_never"][0])))
    put("dropO", pct(q["acc_never"][0] - q["acc_always"][0]), "{:.0f}")
    q2 = o2["okvqa"]["quadrants"]
    put("neverOMtwo", pct(q2["acc_never"][0])); put("alwaysOMtwo", pct(q2["acc_always"][0]))
    ci(o1, "benefit_post - error_post", "primaryOMone"); ci(o2, "benefit_post - error_post", "primaryOMtwo")
    ci(o1, "probe_benefit - probe_error", "probeOMone"); ci(o2, "probe_benefit - probe_error", "probeOMtwo")
    ci(o1, "benefit_pre - error_pre", "preOMone"); ci(o2, "benefit_pre - error_pre", "preOMtwo")
    ci(x1, "vgate_full - benefit_post", "fullVsPostOMone"); ci(x2, "vgate_full - benefit_post", "fullVsPostOMtwo")
    ci(x1, "vgate_full - error_post", "fullVsErrOMone"); ci(x1, "probe_benefit - probe4", "probeVsFourOMone")
    put("hokReadO", pct(x1["okvqa"]["hok"]["read_rate"]), "{:.0f}")
    ci(o1, "benefit_post - signal_qpp_nqc", "vsNqcOMone"); ci(o1, "benefit_post - factorised_post", "vsFactOMone")
    so = pd.read_csv(J6 / f"scores__okvqa__{M1}.csv", index_col="qid"); so.index = so.index.astype(str)
    so = so.loc[[q for q in (ROOT / "data/splits/okvqa_confirm.txt").read_text().split() if q in so.index]]
    put("softNeverO", pct(so.soft_ne.mean())); put("softAlwaysO", pct(so.soft_ev.mean()))
    put("softDropO", pct(so.soft_ne.mean() - so.soft_ev.mean()), "{:.0f}")
    put("gainEdrop", pct(c1["evqa"]["quadrants"]["acc_always"][0] - c1["evqa"]["quadrants"]["acc_never"][0]), "{:.0f}")
    put("gainIdrop", pct(c1["infoseek"]["quadrants"]["acc_always"][0] - c1["infoseek"]["quadrants"]["acc_never"][0]), "{:.0f}")
    RES = ROOT / "experiments/results"
    op1, op2 = json.load(open(RES / f"operating_point__{M1}.json")), json.load(open(RES / f"operating_point__{M2}.json"))
    opr = json.load(open(RES / f"operating_point__{M1}__relaxed.json"))
    rel = opr["mixed_traffic_gate"]
    put("neverOrel", pct(rel["okvqa"]["never"]["acc"])); put("alwaysOrel", pct(rel["okvqa"]["always"]["acc"]))
    put("dropOrel", pct(rel["okvqa"]["never"]["acc"] - rel["okvqa"]["always"]["acc"]), "{:.0f}")

    def opci(rep, key, name):
        p_, lo, hi = rep["macro_diff_95ci"][key]
        put(name, 100 * p_, "{:+.1f}"); put(name + "Lo", 100 * lo, "{:+.1f}"); put(name + "Hi", 100 * hi, "{:+.1f}")
    for tagm, op in [("Mone", op1), ("Mtwo", op2), ("Rel", opr)]:
        mx, pd_ = op["mixed_traffic_gate"], op["per_dataset_gates"]
        for k, nm in [("never", "Never"), ("always", "Always"), ("error_rule_full", "Err"), ("benefit_rule_full", "Full"),
                      ("benefit_rule_pre", "Pre"), ("oracle", "Oracle")]:
            put(f"mix{nm}{tagm}", pct(mx["macro"][k]["acc"])); put(f"mix{nm}Read{tagm}", pct(mx["macro"][k]["read_rate"]), "{:.0f}")
            for tg, d in [("E", "evqa"), ("I", "infoseek"), ("O", "okvqa")]:
                put(f"mix{nm}{tg}{tagm}", pct(mx[d][k]["acc"])); put(f"mix{nm}Read{tg}{tagm}", pct(mx[d][k]["read_rate"]), "{:.0f}")
        opci(mx, "benefit_rule_full - always", f"mixFullVsAlways{tagm}"); opci(mx, "benefit_rule_full - never", f"mixFullVsNever{tagm}")
        opci(mx, "benefit_rule_full - error_rule_full", f"mixFullVsErr{tagm}"); opci(mx, "benefit_full_tuned - error_full_tuned", f"mixTunedVsErr{tagm}")
        opci(mx, "benefit_rule_pre - always", f"mixPreVsAlways{tagm}")
        opci(pd_, "benefit_rule_post - error_rule", f"perPostVsErr{tagm}"); opci(pd_, "benefit_rule_post - always", f"perPostVsAlways{tagm}")
        put(f"perPost{tagm}", pct(pd_["macro"]["benefit_rule_post"]["acc"])); put(f"perErr{tagm}", pct(pd_["macro"]["error_rule"]["acc"]))
        put(f"perErrReadO{tagm}", pct(pd_["okvqa"]["error_rule"]["read_rate"]), "{:.0f}")
        put(f"perPostReadO{tagm}", pct(pd_["okvqa"]["benefit_rule_post"]["read_rate"]), "{:.0f}")
        put(f"perErrO{tagm}", pct(pd_["okvqa"]["error_rule"]["acc"]))
    put("mixSearchSave", 100 - float(mac["mixPreReadMone"]), "{:.0f}")
    for tagm, op in [("Mone", op1), ("Mtwo", op2), ("Rel", opr)]:
        mx = op["mixed_traffic_gate"]
        for k, nm in [("fixed_by_dataset", "Fixed"), ("router", "Router"), ("benefit_rule_post", "Post"), ("cascade", "Cas")]:
            put(f"mix{nm}{tagm}", pct(mx["macro"][k]["acc"])); put(f"mix{nm}Read{tagm}", pct(mx["macro"][k]["read_rate"]), "{:.0f}")
        put(f"mixCasSearch{tagm}", pct(mx["macro"]["cascade"]["search_rate"]), "{:.0f}")
        put(f"routerAcc{tagm}", pct(mx["router_dataset_accuracy"]), "{:.0f}")
        for key, nm in [("benefit_rule_full - router", "mixFullVsRouter"), ("benefit_rule_full - fixed_by_dataset", "mixFullVsFixed")]:
            p_, lo, hi = mx["macro_diff_95ci"][key]  # negative: report the gap as a positive number of points
            put(nm + tagm, -100 * p_, "{:.1f}"); put(nm + tagm + "Lo", -100 * hi, "{:.1f}"); put(nm + tagm + "Hi", -100 * lo, "{:.1f}")
        for tg, d in [("E", "evqa"), ("I", "infoseek"), ("O", "okvqa")]:
            put(f"mixMargin{tg}{tagm}", 100 * mx["dataset_diff_95ci"][d][0], "{:+.1f}")
        pd_ = op["per_dataset_gates"]
        put(f"perErrTuned{tagm}", pct(pd_["macro"]["error_tuned"]["acc"])); put(f"perFullTuned{tagm}", pct(pd_["macro"]["benefit_full_tuned"]["acc"]))
    # Table: one gate for mixed traffic, per model (macro accuracy over the three benchmarks), with the benchmark-identity
    # controls, and the same gates on a benchmark they were not fitted on (scripts/lodo.py; PREREG v4a / v4c)
    M3, M4 = "InternVL3_5-4B-HF", "Qwen3-VL-8B-Instruct"
    OPS = [(n, RES / f"operating_point__{m}.json", m) for n, m in (("Q3-4B", M1), ("Q2.5-3B", M2), ("IVL-4B", M3), ("Q3-8B", M4))]
    OPS = [(n, json.load(open(f))["mixed_traffic_gate"], m) for n, f, m in OPS if f.exists()]
    LODO = {}
    for f in (RES / "lodo__confirm.json", RES / "lodo__confirm__v4c.json"):
        if f.exists():
            LODO.update(json.load(open(f))["results"])
    EOL, nc = " \\\\", len(OPS)
    row = lambda lab, k: f"{lab} & " + " & ".join(f"{pct(mx['macro'][k]['acc']):.1f}" for _, mx, _ in OPS) + EOL
    lrow = lambda lab, k: f"{lab} & " + " & ".join(f"{LODO[m]['macro'][k]:.1f}" if m in LODO else "--" for _, _, m in OPS) + EOL
    tl = ["\\begin{tabular}{@{}l" + "r" * nc + "@{}}", "\\toprule", "Policy & " + " & ".join(n for n, _, _ in OPS) + EOL, "\\midrule",
          row("Never retrieve", "never"), row("Always retrieve", "always"), "\\midrule",
          f"\\multicolumn{{{nc + 1}}}{{@{{}}l}}{{\\emph{{Benchmark identity: that benchmark's better fixed policy}}}}" + EOL,
          row("Told the benchmark", "fixed_by_dataset"), row("Benchmark classifier", "router"), "\\midrule",
          f"\\multicolumn{{{nc + 1}}}{{@{{}}l}}{{\\emph{{One gate for the mixed traffic, no benchmark identity}}}}" + EOL,
          row("Error gate (same features)", "error_rule_full"), row("\\textbf{\\sys{}-full}", "benefit_rule_full"),
          row("Oracle gate", "oracle"), "\\midrule",
          f"\\multicolumn{{{nc + 1}}}{{@{{}}l}}{{\\emph{{Gates fitted on two benchmarks, tested on the third}}}}" + EOL,
          lrow("Error gate (same features)", "error_full"), lrow("\\sys{}-full", "benefit_full"), "\\bottomrule", "\\end{tabular}"]
    (ROOT / "paper/tables/mixed.tex").write_text("\n".join(tl) + "\n")
    for tagm, m in (("Mone", M1), ("Mtwo", M2), ("Mthree", M3), ("Mfour", M4)):  # unseen-benchmark macros
        if m not in LODO:
            continue
        L_ = LODO[m]
        for nm, k in (("lodoErr", "error_full"), ("lodoFull", "benefit_full"), ("lodoAlways", "always"), ("lodoFixed", "best_fixed")):
            put(nm + tagm, L_["macro"][k])
        pt, lo, hi = L_["macro_diff_95ci"]["error_full - benefit_full"]
        put("lodoErrVsFull" + tagm, pt); put("lodoErrVsFull" + tagm + "Lo", lo); put("lodoErrVsFull" + tagm + "Hi", hi)
        put("lodoBestVsFixed" + tagm, -max(L_["macro_diff_95ci"][f"{g} - best_fixed"][0] for g in ("nearest_regime", "error_post", "error_full", "benefit_post", "benefit_pre", "benefit_full")))
        put("lodoReadOMin" + tagm, min(L_["okvqa"][g]["read"] for g in ("nearest_regime", "error_post", "error_full", "benefit_post", "benefit_pre", "benefit_full")), "{:.0f}")
    put("nMixModels", len(OPS), "{:d}")
    rng_ = lambda vals, fmt="{:.1f}": (fmt.format(min(vals)), fmt.format(max(vals)))
    dci = lambda mx, k: [100 * v for v in mx["macro_diff_95ci"][k]]
    for nm, k, sign in (("mixGain", "benefit_rule_full - always", 1), ("mixVsErr", "benefit_rule_full - error_rule_full", 1),
                        ("mixRouterLead", "benefit_rule_full - router", -1), ("mixFixedLead", "benefit_rule_full - fixed_by_dataset", -1)):
        lo, hi = rng_([sign * dci(mx, k)[0] for _, mx, _ in OPS]); put(nm + "Min", lo); put(nm + "Max", hi)
    heads = [100 * (mx[d]["oracle"]["acc"] - max(mx[d]["never"]["acc"], mx[d]["always"]["acc"])) for _, mx, _ in OPS for d in ("evqa", "infoseek", "okvqa")]
    put("oracleHeadMax", max(heads)); put("oracleHeadMin", min(heads))
    mx1 = OPS[0][1]  # main model: where the mixed-traffic gate stands on each benchmark
    for tg, d in (("E", "evqa"), ("I", "infoseek"), ("O", "okvqa")):
        put(f"mixFullVsBest{tg}", 100 * (mx1[d]["benefit_rule_full"]["acc"] - max(mx1[d]["never"]["acc"], mx1[d]["always"]["acc"])), "{:+.1f}")
    lod = [LODO[m] for _, _, m in OPS if m in LODO]
    lo, hi = rng_([L_["macro_diff_95ci"]["error_full - benefit_full"][0] for L_ in lod]); put("lodoErrVsFullMin", lo); put("lodoErrVsFullMax", hi)
    gates_ = ("nearest_regime", "error_post", "error_full", "benefit_post", "benefit_pre", "benefit_full")
    put("lodoBestVsFixedMin", min(-max(L_["macro_diff_95ci"][f"{g} - best_fixed"][0] for g in gates_) for L_ in lod))
    put("lodoBestVsFixedMax", max(-max(L_["macro_diff_95ci"][f"{g} - best_fixed"][0] for g in gates_) for L_ in lod))
    put("lodoErrVsAlwaysMax", max(abs(L_["macro"]["error_full"] - L_["macro"]["always"]) for L_ in lod))
    lo, hi = rng_([L_["evqa"]["benefit_full"]["read"] for L_ in lod], "{:.0f}"); put("lodoEvqaBenReadMin", lo); put("lodoEvqaBenReadMax", hi)
    lo, hi = rng_([L_["evqa"]["always"]["acc"] - L_["evqa"]["benefit_full"]["acc"] for L_ in lod]); put("lodoEvqaBenLossMin", lo); put("lodoEvqaBenLossMax", hi)
    for tg, ds_ in (("O", ("okvqa",)), ("EI", ("evqa", "infoseek"))):  # where the margin over the equal-feature error gate comes from
        lo, hi = rng_([100 * mx["dataset_diff_95ci"][d][0] for _, mx, _ in OPS for d in ds_], "{:+.1f}"); put(f"mixMargin{tg}Min", lo); put(f"mixMargin{tg}Max", hi)
    for tagm, run_, m in (("Mthree", "J11", M3), ("Mfour", "J12", M4)):  # OK-VQA, models 3-4 (PREREG v4c: reported)
        f_ = ROOT / f"experiments/runs/{run_}/extend__{m}.json"
        if f_.exists():
            e_ = json.load(open(f_))
            ci(e_, "vgate_full - benefit_post", f"fullVsPostO{tagm}")
            c_ = json.load(open(ROOT / f"experiments/runs/{run_}/confirm__{m}.json"))
            ci(c_, "benefit_post - error_post", f"primaryO{tagm}"); ci(c_, "probe_benefit - probe_error", f"probeO{tagm}")
            pt, lo, hi = e_["okvqa"]["select_minus_never_95ci"]
            put(f"selNeverO{tagm}", 100 * pt, "{:+.1f}"); put(f"selNeverO{tagm}Lo", 100 * lo, "{:.1f}"); put(f"selNeverO{tagm}Hi", 100 * hi, "{:.1f}")
    put("nLodoModels", len(lod), "{:d}")
    put("lodoInfoBenLead", min(L_["infoseek"]["benefit_full"]["acc"] - L_["infoseek"]["error_full"]["acc"] for L_ in lod))  # the exception to L2
    put("lodoEvqaBenLoss", LODO[M1]["evqa"]["always"]["acc"] - LODO[M1]["evqa"]["benefit_full"]["acc"]); put("lodoEvqaBenRead", LODO[M1]["evqa"]["benefit_full"]["read"], "{:.0f}")
    skip_ = ("never", "always", "oracle", "fixed_by_dataset", "router")  # best read/skip gate fitted per benchmark vs its better fixed policy
    PDG = [json.load(open(RES / f"operating_point__{m}.json"))["per_dataset_gates"] for _, _, m in OPS]
    put("perBestVsFixedMax", 100 * max(B[d][k]["acc"] - max(B[d]["never"]["acc"], B[d]["always"]["acc"]) for B in PDG
                                        for d in ("evqa", "infoseek", "okvqa") for k in B[d] if k not in skip_), "{:.2f}")
    put("lodoReadOMin", min(float(mac[k]) for k in mac if k.startswith("lodoReadOMinM")), "{:.0f}")
    for i, (n_, mx, m) in enumerate(OPS):
        tagm = ("Mone", "Mtwo", "Mthree", "Mfour")[[M1, M2, M3, M4].index(m)]
        if tagm in ("Mthree", "Mfour"):  # models 1-2 are covered above
            for k, nm in [("never", "Never"), ("always", "Always"), ("error_rule_full", "Err"), ("benefit_rule_full", "Full"), ("fixed_by_dataset", "Fixed"), ("router", "Router"), ("oracle", "Oracle")]:
                put(f"mix{nm}{tagm}", pct(mx["macro"][k]["acc"]))
            opci(mx, "benefit_rule_full - always", f"mixFullVsAlways{tagm}"); opci(mx, "benefit_rule_full - error_rule_full", f"mixFullVsErr{tagm}")

# --- Table: replication of the benefit-vs-error effect (equal features), pooled dnAUC with 95% CI
REP = [("Qwen3-VL-4B", c1, R["J4"].parent / "J6" / f"confirm__{M1}.json"), ("Qwen2.5-VL-3B", c2, R["J4"].parent / "J6" / f"confirm__{M2}.json"),
       ("InternVL3.5-4B", c3, ROOT / "experiments/runs/J11/confirm__InternVL3_5-4B-HF.json"),  # OK-VQA for models 3-4: PREREG v4c
       ("Qwen3-VL-8B", ROOT / "experiments/runs/J7/confirm__Qwen3-VL-8B-Instruct.json", ROOT / "experiments/runs/J12/confirm__Qwen3-VL-8B-Instruct.json")]


def repcell(rep, key):
    if rep is None or key not in rep.get("pooled_dnauc_95ci", {}):
        return "--"
    p_, lo, hi = rep["pooled_dnauc_95ci"][key]
    f = lambda v: f"{100 * v:.2f}".replace("-", "$-$")
    return f"{100 * p_:+.2f}".replace("-", "$-$").replace("+", "$+$") + f"\\,{{\\scriptsize[{f(lo)}, {f(hi)}]}}"


rl = ["\\begin{tabular}{@{}lcccc@{}}", "\\toprule",
      " & \\multicolumn{2}{c}{E-VQA + InfoSeek} & \\multicolumn{2}{c}{OK-VQA}" + " \\\\",
      "Model & read gate & probe & read gate & probe" + " \\\\", "\\midrule"]
for name, a_, b_ in REP:
    a_ = json.load(open(a_)) if isinstance(a_, pathlib.Path) and a_.exists() else a_ if isinstance(a_, dict) else None
    b_ = json.load(open(b_)) if isinstance(b_, pathlib.Path) and b_.exists() else None
    if a_ is None:
        continue
    cells = [repcell(x, k) for x in (a_, b_) for k in ("benefit_post - error_post", "probe_benefit - probe_error")]
    if name in ("Qwen3-VL-4B", "Qwen2.5-VL-3B"):  # probe on E-VQA + InfoSeek: scored in a second pass, not preregistered
        cells[1] += "$^{\\dagger}$"
    rl.append(f"{name} & " + " & ".join(cells) + " \\\\")
rl += ["\\bottomrule", "\\end{tabular}"]
(ROOT / "paper/tables/replication.tex").write_text("\n".join(rl) + "\n")

# OK-VQA retrieval quality in the metric the literature headlines (answer string in a top-5 passage); analysis only.
okp = ROOT / "data/processed/okvqa.jsonl"
if okp.exists():
    from vgate.evaluation import preprocess_answer
    keep = set((ROOT / "data/splits/okvqa_confirm.txt").read_text().split())

    def _hit(r, k=5):
        txt = " ".join(e["title"] + " " + e["text"] for e in r["evidence"][:k]).lower()
        return any(a and _re.search(r"\b" + _re.escape(a) + r"\b", txt) for a in {preprocess_answer(x) for x in r["answers"]})
    ok = [r for r in map(json.loads, open(okp)) if r["qid"] in keep]
    put("okPseudoRecall", 100 * np.mean([_hit(r) for r in ok]), "{:.0f}")
    put("okIdRecall", 100 * np.mean([r["lab_gold_in_top5"] for r in ok]), "{:.0f}")
# Literature numbers quoted in the text (not ours): MMKB-RAG, arXiv 2504.10074v3, Table 6 "Performance comparison on the
# M2KR dataset", OK-VQA column: Qwen2-VL-Instruct 60.45; Qwen2-VL-Instruct w/ PreFLMR 46.99 (checked 2026-10-04).
put("litOkNoRag", 60.45); put("litOkRag", 46.99)
lc = ROOT / "experiments/results/live_check.json"  # live-pipeline system check and refit (PREREG v2f / v2g, scripts/live_report.py)
if lc.exists():
    lv = json.load(open(lc)); rf = lv["refit_on_live_logs"]
    dsl = [d for d in ("okvqa", "evqa", "infoseek") if d in lv]
    put("liveN", sum(lv[d]["n"] for d in dsl), "{:d}")
    put("liveAgree", 100 * min(lv[d]["closed_book_answer_agreement"] for d in dsl), "{:.0f}")
    put("liveSign", 100 * min(lv[d]["u_pre_sign_agreement"] for d in dsl), "{:.0f}")
    put("liveSec", float(np.mean([sum(lv[d]["seconds"][k]["median"] for k in ("closed_book", "caption", "retrieval", "evidence")) for d in dsl])), "{:.1f}")
    for key, nm in [("transferred - never", "liveTransVsNever"), ("refit - transferred", "liveRefitVsTrans"),
                    ("refit - always", "liveRefitVsAlways"), ("refit - never", "liveRefitVsNever")]:
        p_, lo, hi = rf["macro_diff_95ci"][key]
        put(nm, 100 * p_, "{:+.1f}"); put(nm + "Lo", 100 * lo, "{:+.1f}"); put(nm + "Hi", 100 * hi, "{:+.1f}")
    tl = rf["macro_diff_95ci"]["transferred - never"]  # reported as a loss: sign flipped, bounds swapped
    put("liveTransLoss", -100 * tl[0], "{:.1f}"); put("liveTransLossLo", -100 * tl[2], "{:.1f}"); put("liveTransLossHi", -100 * tl[1], "{:.1f}")
    put("liveRefitGain", 100 * rf["macro_diff_95ci"]["refit - transferred"][0], "{:.1f}")
    put("liveRefitVsAlwaysAbs", 100 * rf["macro_diff_95ci"]["refit - always"][0], "{:.1f}")
mp = ROOT / "experiments/results/multiplicity.csv"  # Holm over the preregistered family (scripts/multiplicity.py)
if mp.exists():
    hm = pd.read_csv(mp)
    put("holmN", len(hm), "{:d}"); put("holmSig", int(hm.significant_after_holm.sum()), "{:d}")
    put("holmUnadj", int(hm.unadjusted_ci_excludes_0.sum()), "{:d}")
if "select_hidden" in ext.get("Mone", {}).get("evqa", {}):  # PREREG v2d (J8): selection with both passes' hidden states
    for tag, d in (("E", "evqa"), ("I", "infoseek")):
        pt, lo, hi = ext["Mone"][d]["select_hidden"]["selector_hf"]["minus_always_95ci"]
        put(f"selHf{tag}", 100 * pt, "{:+.1f}"); put(f"selHf{tag}Lo", 100 * lo, "{:+.1f}"); put(f"selHf{tag}Hi", 100 * hi, "{:+.1f}")
ps = ROOT / f"experiments/results/passage_select__{M1}__dev.json"  # PREREG v3 pilot (development data; NO-GO)
if ps.exists():
    PS = json.load(open(ps))
    for tag, d in (("E", "evqa"), ("I", "infoseek")):
        put(f"pilotPick{tag}", 100 * PS[d]["minus_always"]["pick_benefit"][0], "{:.1f}")
    put("pilotTokRatio", float(np.mean([PS[d]["tokens"]["pick_benefit"] / PS[d]["tokens"]["always"] for d in ("evqa", "infoseek")])), "{:.1f}")
    for tag, d in (("E", "evqa"), ("I", "infoseek"), ("O", "okvqa")):  # best choice among the six candidate answers vs the better fixed policy
        put(f"pilotHead{tag}", 100 * (PS[d]["acc"]["oracle6"][0] - max(PS[d]["acc"]["always"][0], PS[d]["acc"]["never"][0])), "{:.0f}")
def _share(rep, d, k):  # share of the distance from a random gate to the oracle gate, in nAUC
    P = rep[d]["policies"]
    return 100 * (P[k]["nauc"] - P["random"]["nauc"]) / (P["oracle"]["nauc"] - P["random"]["nauc"])


for nm, k in (("shareBen", "benefit_post"), ("shareErr", "error_post"), ("shareProbeBen", "probe_benefit"), ("shareProbeErr", "probe_error")):
    put(nm, float(np.mean([_share(c1, d, k) for d in ("evqa", "infoseek")])), "{:.0f}")
for tag, d in (("E", "evqa"), ("I", "infoseek")):
    q = c1[d]["quadrants"]
    put(f"oracleHead{tag}", pct(q["acc_oracle_gate"][0] - max(q["acc_always"][0], q["acc_never"][0])))
cp_, cr_ = ROOT / "experiments/results/critic_pilot.json", ROOT / "experiments/results/critic_pilot__reader_check.json"  # PREREG v5 pilot (dev; NO-GO)
pc_ = ROOT / f"experiments/results/passage_select__{M1}__dev__critic.json"
if cp_.exists() and cr_.exists() and pc_.exists():
    CP_, CR_, PC_ = json.load(open(cp_))["results"], json.load(open(cr_))["results"], json.load(open(pc_))
    put("pickGoldRetr", CP_["evqa"]["top1_is_gold"]); put("pickGoldCritic", CP_["evqa"]["argmax_is_gold"]); put("pickGoldReader", CR_["evqa"]["argmax_is_gold"])
    put("criticSpec", CP_["evqa"]["specificity_at_thr"], "{:.0f}"); put("criticRecall", CP_["evqa"]["recall_at_thr"], "{:.0f}")
    for tag, d in (("E", "evqa"), ("I", "infoseek")):
        put(f"criticSel{tag}", 100 * PC_[d]["minus_always"]["pick_benefit"][0], "{:.1f}")
ld = ROOT / "experiments/results/loss_decomposition__confirm.json"  # where the value of retrieval is lost (PREREG v3a)
if ld.exists():
    L = json.load(open(ld))["results"]
    M3, M4 = "InternVL3_5-4B-HF", "Qwen3-VL-8B-Instruct"

    def put3(name, v, fmt="{:.1f}"):
        put(name, v[0], fmt); put(name + "Lo", v[1], fmt); put(name + "Hi", v[2], fmt)
    for tag, d in (("E", "evqa"), ("I", "infoseek")):
        r = L[M1][d]
        put3(f"ldIdeal{tag}", r["ideal_gate_minus_always"], "{:+.1f}"); put3(f"ldGap{tag}", r["gold_alone_minus_all_five"])
        put(f"ldAllFive{tag}", r["all_five_given_gold"][0]); put(f"ldGoldAlone{tag}", r["gold_alone_given_gold"][0])
        put(f"ldSelector{tag}", r["ideal_selector"][0]); put(f"ldGap{tag}Mtwo", L[M2][d]["gold_alone_minus_all_five"][0])
    ideal = [L[m][d]["ideal_gate_minus_always"][0] for m in (M1, M2, M3, M4) for d in ("evqa", "infoseek")]
    put("ldIdealMin", min(ideal), "{:+.1f}"); put("ldIdealMax", max(ideal), "{:+.1f}")
    e = L[M1]["evqa"]
    put("ldRankOne", e["rank1"][0]); put("ldRankRest", e["rank25"][0]); put3("ldRankGap", e["rank1_minus_rank25"])
    gaps = [L[m]["evqa"]["rank1_minus_rank25"][0] for m in (M1, M2, M3, M4)]
    put("ldRankGapMin", min(gaps)); put("ldRankGapMax", max(gaps))
    put("ldGoldAloneRest", e["gold_alone_rank25"][0]); put3("ldGapRest", e["gold_alone_minus_all_five_rank25"])
    put3("ldGapSame", e["gold_alone_minus_all_five_untruncated"]); put("ldGapSameMtwo", L[M2]["evqa"]["gold_alone_minus_all_five_untruncated"][0])
    put("ldNUntrunc", f"{e['n_untruncated']:,}"); put("ldGoldAloneOne", e["gold_alone_rank1"][0])
    put("ldCbWrongGap", e["rank1_minus_rank25__cb_wrong"][0]); put("ldCbRightGap", e["rank1_minus_rank25__cb_right"][0])
    put("ldUntrunc", e["untruncated"][0]); put("ldTrunc", e["truncated"][0])
    i = L[M1]["infoseek"]
    put("ldChunk", i["ans_in_chunks_given_gold"]); put("ldChunkWith", i["all_five_ans_in_chunks"][0]); put("ldChunkWithout", i["all_five_ans_missing"][0])
og = ROOT / "experiments/results/online_gate__confirm.json"  # gate learned from one-sided feedback (PREREG v3b)
if og.exists():
    O = json.load(open(og))
    put("onN", f"{O['reference']['n']:,}"); put("onOrders", O["settings"]["orders"], "{:d}"); put("onEps", 100 * O["settings"]["eps"], "{:.0f}")
    for nm, key in (("onOneSided", "one_sided"), ("onBoth", "both_outcomes"), ("onFrozen", "frozen_dev_gate"), ("onAlways", "always"), ("onNever", "never")):
        put(nm, O["results"][key]["last"][0]); put(nm + "All", O["results"][key]["all"][0])
    o2 = ROOT / f"experiments/results/online_gate__confirm__{M2}.json"  # registered replication (v3b-rep)
    if o2.exists():
        O2 = json.load(open(o2))["results"]
        put("onOneSidedMtwo", O2["one_sided"]["last"][0]); put("onFrozenMtwo", O2["frozen_dev_gate"]["last"][0]); put("onAlwaysMtwo", O2["always"]["last"][0])
    gaps = [O["results"]["frozen_dev_gate"]["last"][0] - O["results"]["one_sided"]["last"][0]]  # learner vs dev-fitted gate, every model
    for tagm, m in (("Mtwo", M2), ("Mthree", "InternVL3_5-4B-HF"), ("Mfour", "Qwen3-VL-8B-Instruct")):
        f = ROOT / f"experiments/results/online_gate__confirm__{m}.json"
        if f.exists():
            Om = json.load(open(f))["results"]
            gaps.append(Om["frozen_dev_gate"]["last"][0] - Om["one_sided"]["last"][0])
            put("onOneSided" + tagm, Om["one_sided"]["last"][0]); put("onFrozen" + tagm, Om["frozen_dev_gate"]["last"][0]); put("onAlways" + tagm, Om["always"]["last"][0])
    put("onGapMax", max(gaps)); put("nOnModels", len(gaps), "{:d}")
    put("onFixed", O["reference"]["fixed_by_dataset"])
    if O.get("live"):
        put("onLiveHalf", O["live"]["one_sided_last_half"][0]); put("onLiveTrans", O["live"]["transferred_gate"]); put("onLiveNever", O["live"]["never"])
        put("onLiveAll", O["live"]["one_sided"][0]); put("onLiveN", O["live"]["n"], "{:d}")
# What the demo's log view shows on the shipped sample log (demo/sample_log.csv; gates cross-validated on the log itself).
sl = ROOT / "demo/sample_log.csv"
if sl.exists():
    SL = pd.read_csv(sl)
    feats_ = [c for c in SL.select_dtypes("number") if c not in ("c_ne", "c_ev", "group", "segment", "cost_ne", "cost_ev", "decision")]
    AT = gate.audit(SL[feats_].to_numpy(), SL.c_ne, SL.c_ev, SL["group"].to_numpy(), SL["segment"].to_numpy(),
                    decision=SL["decision"].to_numpy() if "decision" in SL else None, n_boot=300).set_index("policy")
    for pol_, nm_ in (("benefit gate", "appGate"), ("segment predicted", "appClf"), ("fixed per segment (needs the label)", "appTold"), ("error gate", "appErr"), ("benefit gate per segment (needs the label)", "appPerSeg"),
                      ("logged gate (decision column)", "appLogged"), ("logged gate: its majority decision per segment", "appLoggedMajority")):
        if pol_ in AT.index:
            put(nm_, AT.accuracy[pol_]); put(nm_ + "VsFixed", f"{AT.vs_fixed[pol_]:+.1f}"); put(nm_ + "VsFixedLo", f"{AT.lo[pol_]:+.1f}"); put(nm_ + "VsFixedHi", f"{AT.hi[pol_]:+.1f}")
    put("appSignals", len(feats_), "{:d}"); put("appRows", f"{len(SL):,}")
# What a gate can earn (scripts/gate_bound.py; exploratory, PREREG v9 for the prospective tests): the ceiling for any concave
# ROC curve and the estimate for a binormal one. Macro prefix bd.
_gb = ROOT / "experiments/results/gate_bound.json"
if _gb.exists():
    GB = json.load(open(_gb))
    put("bdN", GB["n_points"], "{:d}"); put("bdMae", GB["mean_abs_error"], "{:.2f}"); put("bdMaxErr", GB["max_abs_error"], "{:.2f}")
    put("bdNeedMin", GB["needed_min"], "{:.2f}"); put("bdNeedMax", GB["needed_max"], "{:.2f}"); put("bdBestAuc", GB["max_auroc"], "{:.2f}")
    put("bdMaxObs", GB["max_observed"], "{:.2f}"); put("bdWithinHalf", GB["within_half_point"], "{:.0f}"); put("bdCells", len(GB["cells"]), "{:d}")
    put("bdAbove", GB["n_above_estimate"], "{:d}"); put("bdMaxAbove", GB["max_above_estimate"], "{:.2f}"); put("bdAboveCeil", GB["n_above_ceiling"], "{:d}")
    put("bdFloorMin", GB["floor_min"], "{:.2f}"); put("bdFloorMax", GB["floor_max"], "{:.2f}"); put("bdBestAboveFloor", GB["cells_best_above_floor"], "{:d}")
    put("bdCeilBestMin", GB["ceiling_at_best_min"], "{:.1f}"); put("bdCeilBestMax", GB["ceiling_at_best_max"], "{:.1f}")
    put("bdCeilShareMed", GB["share_of_ceiling_median"], "{:.0f}"); put("bdCeilShareMax", GB["share_of_ceiling_max"], "{:.0f}")
    put("bdEstMean", GB["estimate_mean"], "{:.2f}"); put("bdObsMean", GB["observed_mean"], "{:.2f}"); put("bdCeilMean", GB["ceiling_mean"], "{:.1f}")
    put("bdSplitMae", GB["split_mean_abs_error"], "{:.2f}"); put("bdSplitMean", f"{GB['split_mean']:+.2f}"); put("bdSplitMax", GB["split_max"], "{:.2f}")
    if "cvqa" in GB:
        c_ = GB["cvqa"]; put("bdCvNeed", c_["auroc_needed_for_1pt"], "{:.2f}"); put("bdCvFloor", c_["auroc_floor_for_1pt"], "{:.2f}")
        for k_, nm in (("benefit_full", "Full"), ("benefit_post", "Post"), ("after_reading", "After")):
            v_ = c_["scores"][k_]
            put(f"bdCvAuc{nm}", v_["auroc"], "{:.2f}"); put(f"bdCvEst{nm}", f"{v_['estimate']:+.1f}"); put(f"bdCvObs{nm}", f"{v_['observed']:+.1f}"); put(f"bdCvCeil{nm}", f"{v_['ceiling']:+.1f}")
    if "published" in GB:
        _pa = [v["auroc"] for v in GB["published"].values()]
        put("bdPubAucMin", min(_pa), "{:.2f}"); put("bdPubAucMax", max(_pa), "{:.2f}"); put("bdPubN", len(_pa), "{:d}")
        put("bdPubBelowFloor", sum(max(v["auroc"], 1 - v["auroc"]) < v["floor"] for v in GB["published"].values()), "{:d}")  # with the sign that suits the score
# The planner table of the log view on the sample log (gate.plan; segment rows: a gate fitted on that segment of the log, out
# of fold; last row: one gate on all of it), plus CVQA (scripts/tail_gate_diag.py --cvqa: the same fit on CVQA's held-out log).
if sl.exists():
    _u = (SL.c_ev - SL.c_ne).to_numpy().astype(int)
    PL = gate.plan(SL.c_ne, SL.c_ev, segment=SL["segment"].to_numpy(), score=gate.oof_scores(SL[feats_].to_numpy(), _u, SL["group"].to_numpy(), kind="benefit"),
                   score_segment=gate.oof_scores_by_segment(SL[feats_].to_numpy(), _u, SL["group"].to_numpy(), SL["segment"].to_numpy()))
    nice = {"evqa": "E-VQA", "infoseek": "InfoSeek", "okvqa": "OK-VQA", "all queries pooled": "All three pooled"}
    line = lambda name, h_, m_, fl, nd, au, ce, es, ld: f"{name} & {h_:.1f} & {m_:.1f} & {fl:.2f} & {nd:.2f} & {au:.2f} & {ce:.1f} & {es:.1f} & " + f"{ld:+.1f}".replace("+", "$+$").replace("-", "$-$") + " \\\\"
    body = []
    for r_ in PL.itertuples():
        body.append(line(nice[r_.segment], r_.help, r_.harm, r_.auroc_floor, r_.auroc_needed, r_.auroc, r_.ceiling, r_.estimate, r_.lead))
        tag_ = {"evqa": "E", "infoseek": "I", "okvqa": "O", "all queries pooled": "Pool"}[r_.segment]
        put(f"plFloor{tag_}", r_.auroc_floor, "{:.2f}"); put(f"plNeed{tag_}", r_.auroc_needed, "{:.2f}"); put(f"plAuc{tag_}", r_.auroc, "{:.2f}")
        put(f"plCeil{tag_}", r_.ceiling, "{:.1f}"); put(f"plEst{tag_}", r_.estimate, "{:.1f}"); put(f"plLead{tag_}", f"{r_.lead:+.1f}")
    _cv = ROOT / "experiments/results/tail_gate_diag__cvqa.json"
    if _cv.exists():
        c_ = json.load(open(_cv))["cvqa"]; v_ = c_["rows"]["logistic, image folds"]
        body.insert(3, line("CVQA", c_["help"], c_["harm"], c_["auroc_floor"], c_["auroc_needed"], v_["auroc"], v_["ceiling"], v_["estimate"], v_["lead"]))
        put("plAucCv", v_["auroc"], "{:.2f}"); put("plEstCv", v_["estimate"], "{:.1f}"); put("plLeadCv", f"{v_['lead']:+.1f}")
        put("plLeadCvLo", f"{v_['lo']:+.1f}"); put("plLeadCvHi", f"{v_['hi']:+.1f}")
    (ROOT / "paper/tables/plan.tex").write_text("\\begin{tabular}{@{}lrrrrrrrr@{}}\n\\toprule\n & \\multicolumn{2}{c}{Retrieval} & \\multicolumn{3}{c}{AUROC} & \\multicolumn{3}{c}{Lead (pts)} \\\\\n"
                                              "Traffic & helps & harms & floor & needed & gate & ceiling & est. & gate \\\\\n\\midrule\n" + "\n".join(body[:-1]) + "\n\\midrule\n" + body[-1] + "\n\\bottomrule\n\\end{tabular}\n")
# PREREG v10 and its diagnosis (scripts/tail_gate_check.py, tail_gate_diag.py; exploratory): a gate fitted on one benchmark's own
# held-out log, with image folds, with folds that keep question strings together, and the policy that is told the string. Prefix tg.
_tg = {m_: ROOT / ("experiments/results/tail_gate_diag" + ("" if m_ == "Qwen3-VL-4B-Instruct" else f"__{m_}") + ".json") for m_ in
       ("Qwen3-VL-4B-Instruct", "Qwen2.5-VL-3B-Instruct", "InternVL3_5-4B-HF", "Qwen3-VL-8B-Instruct")}
if all(f_.exists() for f_ in _tg.values()):
    TG = {m_: json.load(open(f_)) for m_, f_ in _tg.items()}
    ci = lambda v: (f"{v['lead']:+.1f}", f"{v['lo']:+.1f}", f"{v['hi']:+.1f}")
    main_ = TG["Qwen3-VL-4B-Instruct"]["infoseek"]
    for key_, nm in (("logistic, image folds", "Log"), ("boosted, image folds", "Bst"), ("logistic, string folds", "LogStr"), ("boosted, string folds", "BstStr"),
                     ("told the string (image folds)", "Told"), ("boosted, image folds, 1,000 labels", "BstK"), ("logistic, image folds, 1,000 labels", "LogK")):
        a_, b_, c__ = ci(main_["rows"][key_]); put(f"tgI{nm}", a_); put(f"tgI{nm}Lo", b_); put(f"tgI{nm}Hi", c__)
    put("tgIBstEst", f"{main_['rows']['boosted, image folds']['estimate']:+.1f}")
    put("tgIBstAuc", main_["rows"]["boosted, image folds"]["auroc"], "{:.2f}"); put("tgILogAuc", main_["rows"]["logistic, image folds"]["auroc"], "{:.2f}")
    put("tgIBstStrAuc", main_["rows"]["boosted, string folds"]["auroc"], "{:.2f}"); put("tgIStrings", main_["question_strings"], "{:d}")
    put("tgIShareTen", main_["share_in_strings_of_10_or_more"], "{:.0f}")
    d_ = main_["boosted_minus_told_string"]; put("tgIBstVsTold", f"{d_['diff']:+.1f}"); put("tgIBstVsToldLo", f"{d_['lo']:+.1f}"); put("tgIBstVsToldHi", f"{d_['hi']:+.1f}")
    # every reader x benchmark x model with image folds: how many lead by one point with an interval above zero, and where
    cells_ = [(m_, d__, k_, v_) for m_, R_ in TG.items() for d__, c__ in R_.items() for k_, v_ in c__["rows"].items() if k_ in ("logistic, image folds", "boosted, image folds")]
    win = [(m_, d__, k_) for m_, d__, k_, v_ in cells_ if v_["lead"] >= 1.0 and v_["lo"] > 0]
    put("tgCells", len(cells_), "{:d}"); put("tgWins", len(win), "{:d}"); put("tgWinsInfoseek", sum(d__ == "infoseek" for _, d__, _ in win), "{:d}")
    put("tgWinReaders", len({m_ for m_, _, _ in win}), "{:d}")
    put("tgRestMax", f"{max(v_['lead'] for m_, d__, k_, v_ in cells_ if (m_, d__, k_) not in win):+.1f}")
    put("tgPos", sum(v_["lo"] > 0 for _, _, _, v_ in cells_), "{:d}")
    other = [v_["lead"] for m_, d__, k_, v_ in cells_ if d__ != "infoseek"]
    put("tgOtherMax", f"{max(other):+.1f}")
    isk = [v_["lead"] for m_, d__, k_, v_ in cells_ if d__ == "infoseek" and k_.startswith("boosted")]
    put("tgIBstMin", f"{min(isk):+.1f}"); put("tgIBstMax", f"{max(isk):+.1f}")
    drop = [R_["infoseek"]["rows"]["boosted, image folds"]["auroc"] - R_["infoseek"]["rows"]["boosted, string folds"]["auroc"] for R_ in TG.values()]
    put("tgIAucDropMin", min(drop), "{:.2f}"); put("tgIAucDropMax", max(drop), "{:.2f}")
    strf = [R_["infoseek"]["rows"][k_]["lead"] for R_ in TG.values() for k_ in ("logistic, string folds", "boosted, string folds")]
    put("tgIStrMin", f"{min(strf):+.1f}"); put("tgIStrMax", f"{max(strf):+.1f}")
    if all("string_folds_repeated" in R_["infoseek"] for R_ in TG.values()):  # ten random assignments of the question strings to folds
        rp = [v_ for R_ in TG.values() for v_ in R_["infoseek"]["string_folds_repeated"].values()]
        put("tgIRepN", rp[0]["repeats"], "{:d}"); put("tgIRepMeanMin", f"{min(v_['lead_mean'] for v_ in rp):+.1f}"); put("tgIRepMeanMax", f"{max(v_['lead_mean'] for v_ in rp):+.1f}")
        put("tgIRepMax", f"{max(v_['lead_max'] for v_ in rp):+.1f}"); put("tgIRepMin", f"{min(v_['lead_min'] for v_ in rp):+.1f}")
        dr = [R_["infoseek"]["rows"]["boosted, image folds"]["auroc"] - R_["infoseek"]["string_folds_repeated"]["boosted"]["auroc_mean"] for R_ in TG.values()]
        put("tgIRepAucDropMin", min(dr), "{:.2f}"); put("tgIRepAucDropMax", max(dr), "{:.2f}")
# PREREG v10b (scripts/tail_gate_dev.py; exploratory, not blind): the same two model classes fitted on development data. Prefix tgd;
# tgAll counts every fit of v10 and v10b (four readers x three benchmarks and the main reader on CVQA, two models, two ways of fitting).
_td = ROOT / "experiments/results/tail_gate_dev.json"
_tc = ROOT / "experiments/results/tail_gate_diag__cvqa.json"
if _td.exists() and _tc.exists() and all(f_.exists() for f_ in _tg.values()):
    TD = json.load(open(_td)); m1_ = "Qwen3-VL-4B-Instruct"
    c3 = lambda nm, v: (put(nm, f"{v['lead']:+.1f}"), put(nm + "Lo", f"{v['lo']:+.1f}"), put(nm + "Hi", f"{v['hi']:+.1f}"))
    di = TD[m1_]["infoseek"]
    c3("tgdIBst", di["models"]["boosted"]); c3("tgdILog", di["models"]["logistic"]); c3("tgdISeen", di["models"]["boosted"]["seen"]); c3("tgdIUnseen", di["models"]["boosted"]["unseen"])
    put("tgdIShareSeen", di["share_seen_strings"], "{:.0f}"); put("tgdIBstAuc", di["models"]["boosted"]["auroc"], "{:.2f}")
    fits = [(("dev", m_, d_, n_), v_) for m_, R_ in TD.items() for d_, c_ in R_.items() for n_, v_ in c_["models"].items()]
    fits += [(("log", m_, d_, k_.split(",")[0]), v_) for m_, R_ in list(TG.items()) + [(m1_, json.load(open(_tc)))] for d_, c_ in R_.items() for k_, v_ in c_["rows"].items()
             if k_ in ("logistic, image folds", "boosted, image folds")]
    won = [k_ for k_, v_ in fits if v_["lead"] >= 1.0 and v_["lo"] > 0]
    assert all(k_[1] == m1_ and k_[2] == "infoseek" for k_ in won), won
    put("tgAllFits", len(fits), "{:d}"); put("tgAllWins", len(won), "{:d}"); put("tgAllRestMax", f"{max(v_['lead'] for k_, v_ in fits if k_ not in won):+.1f}")
    put("tgAllPos", sum(v_["lo"] > 0 for _, v_ in fits), "{:d}")
    # the planner's estimate against these fits' leads (no threshold is chosen on the held-out labels here): correlation, and whether
    # the fits that lead by a point are exactly those with the highest estimates
    le_, es_ = np.array([v_["lead"] for _, v_ in fits]), np.array([v_["estimate"] for _, v_ in fits])
    top_ = [fits[i][0] for i in np.argsort(-es_, kind="stable")[:len(won)]]
    assert set(top_) == set(won) and np.sort(es_)[-len(won)] > np.sort(es_)[-len(won) - 1], (top_, won)
    put("tgAllCorr", float(np.corrcoef(le_, es_)[0, 1]), "{:.2f}"); put("tgAllEstRestMax", float(np.sort(es_)[-len(won) - 1]), "{:.1f}")
    put("tgAllEstWonMin", float(np.sort(es_)[-len(won)]), "{:.1f}"); put("tgAllAbove", int((le_ > es_).sum()), "{:d}")
    pos_ = np.array([v_["lo"] > 0 for _, v_ in fits])  # among the fits whose interval is above zero: how often the estimate is below the lead
    put("tgAllPosBelow", int((es_[pos_] < le_[pos_]).sum()), "{:d}")
# The planner on counts another paper prints for GPT-4o on InfoSeek (scripts/published_rates.py; arXiv:2405.18740). Prefix pb.
_pb = ROOT / "experiments/results/published_rates.json"
if _pb.exists():
    PB = json.load(open(_pb)); ag_ = list(PB["agents"].values())
    put("pbN", f"{PB['n']:,}"); put("pbHelp", PB["help"], "{:.1f}"); put("pbHarm", PB["harm"], "{:.1f}"); put("pbFloor", PB["auroc_floor"], "{:.2f}"); put("pbNeed", PB["auroc_needed"], "{:.2f}")
    put("pbAucMin", min(v_["auroc_of_the_decision"] for v_ in ag_), "{:.2f}"); put("pbAucMax", max(v_["auroc_of_the_decision"] for v_ in ag_), "{:.2f}")
    put("pbLeadMin", f"{-max(v_['lead_from_identity'] for v_ in ag_):.1f}"); put("pbLeadMax", f"{-min(v_['lead_from_identity'] for v_ in ag_):.1f}")  # points BELOW always-retrieve
    put("pbAgents", len(ag_), "{:d}"); put("pbIdErr", PB["identity_max_abs_error"], "{:.2f}")
# PREREG v7d (preregistered): CVQA on a second reader (J18). Prefix cvT. PREREG v9 forward tests of the planner (B1-B4). Prefix fw.
_t2 = "Qwen2.5-VL-3B-Instruct"
_ca2, _cs2 = ROOT / f"experiments/results/cvqa_audit__confirm__{_t2}.json", ROOT / f"experiments/results/cvqa_select__confirm__{_t2}.json"
if _ca2.exists() and _cs2.exists():
    CA2, CS2 = json.load(open(_ca2)), json.load(open(_cs2))
    put("cvTHelp", CA2["help"], "{:.1f}"); put("cvTHarm", CA2["harm"], "{:.1f}"); put("cvTN", f"{CA2['n']:,}")
    for k_, nm, src_ in (("full_minus_best_fixed", "cvTFullVsFixed", CA2), ("select_minus_never", "cvTSelVsFixed", CS2), ("select_minus_vgate_full", "cvTSelVsFull", CS2)):
        put(nm, f"{src_[k_][0]:+.2f}"); put(nm + "Lo", f"{src_[k_][1]:+.2f}"); put(nm + "Hi", f"{src_[k_][2]:+.2f}")
    assert CA2["full_minus_best_fixed"][0] < 1.0 and CS2["select_minus_never"][0] < 1.0 and CS2["no_option_but_more_confident"] == 0.0  # as the paragraph says
_fw = ROOT / "experiments/results/gate_bound__prospective.json"
if _fw.exists():
    FW = json.load(open(_fw))
    if isinstance(FW.get("B1"), dict) and isinstance(FW.get("B2_B3"), dict):
        b1_, b2_ = list(FW["B1"].values()), list(FW["B2_B3"]["scores"].values())
        rule_ = [v_["B1_at_most_estimate_plus_0p3"] for v_ in b1_] + [v_["B2_rule_at_most_estimate_plus_0p3"] for v_ in b2_]
        best_ = [v_["B2_best_lead_within_0p5_of_estimate"] for v_ in b2_]
        ceil_ = [v_["B4_best_lead_at_most_ceiling"] for v_ in b1_ + b2_]
        assert [v_["B3_correct"] for v_ in b2_] == best_  # the paper says "for N of M scores the estimate was within half a point ... and the needed AUROC told ..."
        put("fwRuleHeld", sum(rule_), "{:d}"); put("fwRuleN", len(rule_), "{:d}"); put("fwBestHeld", sum(best_), "{:d}"); put("fwBestN", len(best_), "{:d}")
        put("fwCeilHeld", sum(ceil_), "{:d}"); put("fwCeilN", len(ceil_), "{:d}")
# PREREG v13 (preregistered): a larger reader, Qwen3-VL-32B in 4-bit, on a frozen sample of held-out questions (scripts/big_reader.py). Prefix br.
_br = ROOT / "experiments/results/big_reader.json"
if _br.exists():
    BR = json.load(open(_br)); au_ = BR["audit"]
    assert BR["Q1"] and BR["Q2"] and BR["Q3"] and BR["Q4"]  # the sentence is worded for this outcome
    put("brN", f"{BR['n']:,}")
    for k_, nm in (("benefit gate", "Gate"), ("segment predicted", "Cls"), ("fixed per segment (needs the label)", "Told"), ("always retrieve", "Always")):
        put("br" + nm, au_[k_]["accuracy"], "{:.1f}")
    for k_, nm in (("benefit gate", "GateVsTold"), ("segment predicted", "ClsVsTold")):
        v_ = au_[k_]["vs_told"]; put("br" + nm, f"{-v_[0]:.1f}"); put("br" + nm + "Lo", f"{-v_[2]:.1f}"); put("br" + nm + "Hi", f"{-v_[1]:.1f}")  # points BELOW told
# PREREG v12 (exploratory): the planner's noise floor (scripts/null_lead_check.py). Prefix nl.
_nl = ROOT / "experiments/results/null_lead.json"
if _nl.exists():
    NL = json.load(open(_nl)); lg_ = {r_["log"]: r_ for r_ in NL["logs"]}
    put("nlCaseMax", NL["summary"]["null_lead_with_sign_max"], "{:.2f}")  # the larger of the two (sign chosen too), over the twelve cells
    put("nlCaseCeil", float(np.ceil(100 * NL["summary"]["null_lead_with_sign_max"]) / 100), "{:.2f}")  # a ceiling, for "below ..." (the with-sign maximum rounded up)
    pub_ = lg_["published log (GPT-4o, six sources pooled)"]; put("nlPub", pub_["null_lead"], "{:.1f}"); put("nlPubSign", pub_["null_lead_with_sign"], "{:.1f}")
    put("nlRef", lg_["reference: helps 10%, harms 10%, 1,000 queries"]["null_lead"], "{:.1f}")
# PREREG v11 (registered): the audit on Chen et al.'s released logs (scripts/chen_logs_audit.py). Prefix ck. Differences are
# printed as points BELOW told-the-source.
_ck = ROOT / "experiments/results/chen_logs_audit.json"
if _ck.exists():
    CK = json.load(open(_ck)); cr_ = CK["readers"]; g4_ = cr_["gpt4-o"]["policies"]
    hd_ = [c_["policies"]["hard"]["vs_told"] for c_ in cr_.values()]; sf_ = [c_["policies"]["soft"]["vs_told"] for c_ in cr_.values()]
    assert all(v_[0] < 0 for v_ in hd_ + sf_)  # no gate of theirs is ahead of told-the-source for any reader
    put("ckN", CK["n"], "{:d}"); put("ckSources", len(CK["sources"]), "{:d}"); put("ckReaders", len(cr_), "{:d}")
    put("ckHardMin", f"{min(-v_[0] for v_ in hd_):.1f}"); put("ckHardMax", f"{max(-v_[0] for v_ in hd_):.1f}")
    put("ckSoftMin", f"{min(-v_[0] for v_ in sf_):.1f}"); put("ckSoftMax", f"{max(-v_[0] for v_ in sf_):.1f}")
    v_ = g4_["hard"]["vs_told"]; put("ckGptHard", f"{-v_[0]:.1f}"); put("ckGptHardLo", f"{-v_[2]:.1f}"); put("ckGptHardHi", f"{-v_[1]:.1f}")
    put("ckHardBelow", sum(v_[2] < 0 for v_ in hd_), "{:d}"); put("ckSoftBelow", sum(v_[2] < 0 for v_ in sf_), "{:d}")
    for nm, k_ in (("Never", "never"), ("Always", "always"), ("HardAcc", "hard"), ("SoftAcc", "soft"), ("Told", "told the source"), ("Oracle", "oracle")):
        put("ckGpt" + nm, g4_[k_]["accuracy"], "{:.1f}")
    put("ckHardRate", g4_["hard"]["search_rate"], "{:.0f}"); put("ckSoftRate", g4_["soft"]["search_rate"], "{:.0f}")
    tr_ = [c_["policies"]["told the source"]["search_rate"] for c_ in cr_.values()]; put("ckToldRateMin", min(tr_), "{:.0f}"); put("ckToldRateMax", max(tr_), "{:.0f}")
    put("ckExpHeld", sum(bool(CK[k_]) for k_ in ("K1", "K2", "K3", "K4")), "{:d}"); put("ckExpN", 4, "{:d}")
    put("ckRepro", max(c_.get("reproduction_max_abs_diff", 0) for c_ in cr_.values()), "{:.2f}")
    put("ckGptHelp", cr_["gpt4-o"]["planner"]["help"], "{:.1f}"); put("ckGptHarm", cr_["gpt4-o"]["planner"]["harm"], "{:.1f}")
    put("ckAheadMax", max(c_["told_200_assignments"]["hard_ahead_share"] for c_ in cr_.values()), "{:.1f}")  # % of 200 other fold assignments
    assert all(c_["told_200_assignments"]["soft_ahead_share"] == 0 for c_ in cr_.values())
# PREREG v6b / v6c (pilot on E-VQA's development split): image re-ranking of the top-20 and the readers on the re-ranked evidence. Prefix rr.
_rr, _rk = ROOT / "experiments/results/kbimage_reader_pass.json", ROOT / "experiments/results/kbimage_rerank.json"
if _rr.exists() and _rk.exists():
    RRP, RK = json.load(open(_rr)), json.load(open(_rk))
    for m_, nm in (("Qwen3-VL-4B-Instruct", "Main"), ("Qwen2.5-VL-3B-Instruct", "Second")):
        v_ = RRP[m_]
        put(f"rr{nm}Diff", f"{v_['diff'][0]:+.1f}"); put(f"rr{nm}Lo", f"{v_['diff'][1]:+.1f}"); put(f"rr{nm}Hi", f"{v_['diff'][2]:+.1f}")
        put(f"rr{nm}Before", v_["always_retriever"], "{:.1f}"); put(f"rr{nm}After", v_["always_reranked"], "{:.1f}")
        put(f"rr{nm}HarmBefore", v_["harm"]["retriever"], "{:.1f}"); put(f"rr{nm}HarmAfter", v_["harm"]["reranked"], "{:.1f}")
        put(f"rr{nm}NeedBefore", v_["planner"]["retriever"]["needed"], "{:.2f}"); put(f"rr{nm}NeedAfter", v_["planner"]["reranked"]["needed"], "{:.2f}")
    put("rrPageBefore", RK["page_in5"]["retriever"], "{:.1f}"); put("rrPageAfter", RK["page_in5"]["reranked"], "{:.1f}"); put("rrN", f"{RRP['n']:,}")
# PREREG v6d / v6f (preregistered, E-VQA held-out split): the same re-ranking, and the planner's forecast for gates on it. Prefix rrc / rg.
_rc, _rg = ROOT / "experiments/results/kbimage_reader_pass__confirm.json", ROOT / "experiments/results/rerank_gate_pilot__confirm.json"
if _rc.exists() and _rg.exists():
    RC, RG = json.load(open(_rc)), json.load(open(_rg))
    assert RC["E1"] and RC["E2"] and RC["E3"] and RG["F1"] and not RG["F1_unrounded"] and RG["F2"] and RG["F3"]  # the paragraph is worded for this outcome
    for m_, nm in (("Qwen3-VL-4B-Instruct", "Main"), ("Qwen2.5-VL-3B-Instruct", "Second")):
        v_ = RC[m_]
        assert v_["diff"][1] > 0
        put(f"rrc{nm}Diff", f"{v_['diff'][0]:+.1f}"); put(f"rrc{nm}Lo", f"{v_['diff'][1]:+.1f}"); put(f"rrc{nm}Hi", f"{v_['diff'][2]:+.1f}")
        put(f"rrc{nm}Before", v_["always_retriever"], "{:.1f}"); put(f"rrc{nm}After", v_["always_reranked"], "{:.1f}")
        put(f"rrc{nm}HarmBefore", v_["harm"]["retriever"], "{:.1f}"); put(f"rrc{nm}HarmAfter", v_["harm"]["reranked"], "{:.1f}")
    put("rrcGoldBefore", RC["gold_passage_among_five"]["retriever"], "{:.1f}"); put("rrcGoldAfter", RC["gold_passage_among_five"]["reranked"], "{:.1f}"); put("rrcN", f"{RC['n']:,}")
    bc_ = [RG[m_][k_] for m_ in ("Qwen3-VL-4B-Instruct", "Qwen2.5-VL-3B-Instruct") for k_ in ("B", "C")]
    put("rgFits", sum(len(c_["fits"]) for c_ in bc_) // 2, "{:d}")  # per reader
    put("rgMaxLead", f"{max(v_['lead'] for c_ in bc_ for v_ in c_['fits'].values()):+.2f}")
    rare_ = [min(c_["help"], c_["harm"]) for c_ in bc_]; put("rgRareMin", min(rare_), "{:.1f}"); put("rgRareMax", max(rare_), "{:.1f}")
# Published gates through the audit (PREREG v8): scripts/external_gates.py. Macro prefix xg<Gate>; dev-only gates get the suffix Dev.
sg = lambda v: f"{v:+.1f}"
for split_, suf in (("confirm", ""), ("dev", "Dev")):
    xf = ROOT / f"experiments/results/external_gates__{split_}.json"
    if not xf.exists():
        continue
    XG = json.load(open(xf))["gates"]
    held = []
    for gname, tag in (("chen", "Chen"), ("urag", "Urag"), ("reflectiva", "Refl"), ("reag_critic", "Critic")):
        o = XG.get(gname)
        if not o or "status" in o:
            continue
        for d_, t_ in (("evqa", "E"), ("infoseek", "I"), ("okvqa", "O")):
            bb = o["by_benchmark"][d_]
            put(f"xg{tag}Read{t_}{suf}", bb["read"], "{:.0f}"); put(f"xg{tag}VsFixed{t_}{suf}", sg(bb["gate_minus_fixed"][0]))
            put(f"xg{tag}Auc{t_}{suf}", bb["auroc_help_vs_harm"], "{:.2f}")
        mx = o["mixed"]
        put(f"xg{tag}Mixed{suf}", mx["gate"]); put(f"xg{tag}Majority{suf}", mx["majority"]); put(f"xg{tag}Same{suf}", mx["same_as_majority"], "{:.0f}")
        for key, nm in (("gate_minus_told_the_benchmark", "VsTold"), ("gate_minus_always", "VsAlways"), ("gate_minus_its_benchmark_majority", "VsMajority"),
                        ("gate_minus_random_at_its_read_rates", "VsRandom")):
            put(f"xg{tag}{nm}{suf}", sg(mx[key][0])); put(f"xg{tag}{nm}{suf}Lo", sg(mx[key][1])); put(f"xg{tag}{nm}{suf}Hi", sg(mx[key][2]))
        put(f"xg{tag}VsToldAbs{suf}", f"{abs(mx['gate_minus_told_the_benchmark'][0]):.1f}")  # for "N points from the policy told the benchmark"
        put(f"xg{tag}VsToldLoAbs{suf}", f"{abs(mx['gate_minus_told_the_benchmark'][1]):.1f}"); put(f"xg{tag}VsToldHiAbs{suf}", f"{abs(mx['gate_minus_told_the_benchmark'][2]):.1f}")
        put(f"xg{tag}MaxVsFixed{suf}", sg(max(o["by_benchmark"][d_]["gate_minus_fixed"][0] for d_ in ("evqa", "infoseek", "okvqa"))))
        held.append(all(o["expectations"].values()))
    put(f"xgN{suf}", len(held), "{:d}"); put(f"xgHeld{suf}", sum(held), "{:d}")
    put(f"xgTold{suf}", next(iter(o_ for o_ in XG.values() if "mixed" in o_), {"mixed": {"fixed": float("nan")}})["mixed"]["fixed"])
# Registered expectations of PREREG v8 (two prompts on the held-out split, the critic on the development split) and the largest
# lead of any released gate over a benchmark's better fixed policy.
_xc, _xd = ROOT / "experiments/results/external_gates__confirm.json", ROOT / "experiments/results/external_gates__dev.json"
if _xc.exists() and _xd.exists():
    _gc, _gd = json.load(open(_xc))["gates"], json.load(open(_xd))["gates"]
    _reg = [_gc[g_] for g_ in ("chen", "urag", "reflectiva") if "expectations" in _gc.get(g_, {})] + [_gd["reag_critic"]]
    put("xgExpN", sum(len(o_["expectations"]) for o_ in _reg), "{:d}"); put("xgExpHeld", sum(sum(o_["expectations"].values()) for o_ in _reg), "{:d}")
    put("xgGates", len(_reg), "{:d}")
    _held = [_gc[g_] for g_ in ("chen", "urag", "reflectiva") if "expectations" in _gc.get(g_, {})]  # held-out split only
    put("xgExpNConf", sum(len(o_["expectations"]) for o_ in _held), "{:d}"); put("xgExpHeldConf", sum(sum(o_["expectations"].values()) for o_ in _held), "{:d}")
    put("xgMaxVsFixedConf", f"{max(o_['by_benchmark'][d_]['gate_minus_fixed'][0] for o_ in _held for d_ in ('evqa', 'infoseek', 'okvqa')):+.2f}")
    for d_, t_ in (("evqa", "E"), ("infoseek", "I"), ("okvqa", "O")):
        put(f"xgChenVsRandom{t_}", sg(_gc["chen"]["by_benchmark"][d_]["gate_minus_random"][0]))
    put("xgMaxVsFixedAll", sg(max(o_["by_benchmark"][d_]["gate_minus_fixed"][0] for o_ in _reg for d_ in ("evqa", "infoseek", "okvqa"))))
    put("xgMinToTold", f"{min(-o_['mixed']['gate_minus_told_the_benchmark'][0] for o_ in _reg):.1f}"); put("xgMaxToTold", f"{max(-o_['mixed']['gate_minus_told_the_benchmark'][0] for o_ in _reg):.1f}")
# CVQA, a fourth benchmark (PREREG v7): scripts/cvqa_audit.py. Macro prefix cv; development split: cvDev.
for split_, suf in (("confirm", ""), ("dev", "Dev"), ("confirm__fallback", "Fb")):  # Fb: registered sensitivity labels, held-out
    cf = ROOT / f"experiments/results/cvqa_audit__{split_}.json"
    if not cf.exists():
        continue
    CV = json.load(open(cf))
    put(f"cv{suf}N", f"{CV['n']:,}"); put(f"cv{suf}Help", CV["help"]); put(f"cv{suf}Harm", CV["harm"]); put(f"cv{suf}OracleHead", CV["oracle_headroom"])
    for k_, nm in (("never", "Never"), ("always", "Always"), ("best_fixed", "Fixed"), ("benefit_full", "Full"), ("error_full", "Err"), ("told_category", "Cat"),
                   ("category_classifier", "CatClf"), ("told_subset", "Sub"), ("oracle", "Oracle")):
        put(f"cv{suf}{nm}", CV["rules"][k_]["acc"]); put(f"cv{suf}{nm}Read", CV["rules"][k_]["read"], "{:.0f}")
    put(f"cv{suf}PostVsNever", sg(CV["rules"]["benefit_post"]["acc"] - CV["rules"]["never"]["acc"]))
    for k_, nm in (("full_minus_best_fixed", "FullVsFixed"), ("full_minus_error_full", "FullVsErr"), ("full_minus_told_category", "FullVsCat"),
                   ("full_minus_told_subset", "FullVsSub"), ("nauc_full_minus_error_full", "NaucVsErr"), ("full_minus_always", "FullVsAlways"), ("full_minus_never", "FullVsNever")):
        put(f"cv{suf}{nm}", sg(CV[k_][0])); put(f"cv{suf}{nm}Lo", sg(CV[k_][1])); put(f"cv{suf}{nm}Hi", sg(CV[k_][2]))
# CVQA, answer selection after reading (PREREG v7c): scripts/cvqa_select.py. Macro prefix cvSel.
_cs = ROOT / "experiments/results/cvqa_select__confirm.json"
if _cs.exists():
    CS = json.load(open(_cs))
    put("cvSel", CS["acc"]["select"]); put("cvSelUses", CS["uses_evidence_answer"], "{:.0f}"); put("cvSelTokens", CS["tokens_relative_to_always"], "{:.2f}")
    put("cvSelHelpKept", CS["help_kept"], "{:.0f}"); put("cvSelHarmAvoided", CS["harm_avoided"], "{:.0f}")
    for k_, nm in (("select_minus_never", "cvSelVsFixed"), ("select_minus_always", "cvSelVsAlways"), ("select_minus_vgate_full", "cvSelVsFull"),
                   ("select_minus_vgate_post", "cvSelVsPost"), ("prob_only_minus_never", "cvSelProbOnly")):
        put(nm, sg(CS[k_][0])); put(nm + "Lo", sg(CS[k_][1])); put(nm + "Hi", sg(CS[k_][2]))
    _so = ROOT / "experiments/results/select_open_ended__dev.json"  # the same rule on the open-ended benchmarks, development split, four readers
    if _so.exists():
        SO = [c_["rule_minus_better_fixed"] for m_ in json.load(open(_so)).values() for c_ in m_.values()]
        put("selOpenN", len(SO), "{:d}"); put("selOpenBelow", sum(c_[2] < 0 for c_ in SO), "{:d}"); put("selOpenAbove", sum(c_[1] > 0 for c_ in SO), "{:d}"); put("selOpenTrail", sum(c_[0] < 0 for c_ in SO), "{:d}")
    _csf = ROOT / "experiments/results/cvqa_select__confirm__fallback.json"  # the same rule against gates that get the same fallback
    if _csf.exists():
        CF = json.load(open(_csf))
        for k_, nm in (("select_minus_vgate_full", "cvFbSelVsFull"), ("select_minus_vgate_post", "cvFbSelVsPost")):
            put(nm, sg(CF[k_][0])); put(nm + "Lo", sg(CF[k_][1])); put(nm + "Hi", sg(CF[k_][2]))
for k in ("primaryMone", "probeMone", "mixFullVsAlwaysMone", "mixFullVsErrMone", "mixFullVsNeverMone"):  # unsigned, for running text
    if k in mac:
        put(k + "Abs", mac[k].lstrip("+"))

lines = ["% AUTO-GENERATED by scripts/paper_numbers.py from experiments/runs/{J1,J2,J4}, experiments/audits,",
         "% experiments/results/signal_auroc_confirm.csv. Do not edit by hand."]
tex = lambda v: ("$-$" + v[1:]) if isinstance(v, str) and v.startswith("-") else v  # typographic minus
lines += [f"\\newcommand{{\\num{k}}}{{{tex(v)}}}" for k, v in sorted(mac.items())]
(ROOT / "paper/tables/numbers.tex").write_text("\n".join(lines) + "\n")
print("\n".join(f"{k} = {v}" for k, v in sorted(mac.items())))
