"""PREREG v11, step 1: score the released answers of Chen et al. (arXiv:2502.18023) with their own accuracy function.

Input: a copy of github.com/Chord-Chen-30/VLLM-KnowledgeBoundary (Apache-2.0) with the Git-LFS files of
evaluation/eval_data/{mix,life}/ fetched. Their evaluation code is imported and run unchanged (VQA, VQAEval.evaluate, the
decision parser of score_with_search_decision_all.py at their default threshold 4.0); only the judge-model score, which
calls a paid API, is replaced by a constant and not used. Needs jieba and nltk (with the punkt_tab resource), which are not
dependencies of this repository: pass --deps <folder they were installed into with pip --target>.

  python scripts/chen_logs_score.py --repo <copy of their repository> [--deps <folder>]
      -> experiments/results/chen_logs__questions.csv   (one row per reader, set and question; no question text)
"""
import argparse
import contextlib
import io
import json
import os
import pathlib
import sys
import types

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
READERS = ["gpt4-o", "qwen-vl-max", "qwen-vl-2", "qwen-vl-chat", "deepseek-vl-chat"]
GATES = dict(mix=dict(hard="mix_1129_hard.jsonl", soft="mix_1129_soft.jsonl", human="mix_1129_human.jsonl", prompt="mix_prompt_KB.jsonl"),
             life=dict(hard="life_oss_cz_1022_hard_KB_model.jsonl", soft="life_oss_cz_1022_soft_KB_model.jsonl", human="life_oss_cz_1022_human.jsonl",
                       prompt="life_oss_prompt_KB.jsonl"))
# Mix: six sources of 100 questions each, in this order (from the image folders of the decision files).
SOURCES = ["life", "in-house 2", "dyn-vqa zh", "dyn-vqa en", "nocaps", "visual7w"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--deps")
    ap.add_argument("--judge_inputs", help="PREREG v11b: also write what their code would send to the judge model (processed reply, reference answers) "
                                           "for every Mix answer to <folder>/judge_inputs.jsonl; the folder is uploaded as a private dataset, not committed")
    a = ap.parse_args()
    repo = pathlib.Path(a.repo).resolve()
    if a.deps:
        sys.path.insert(0, a.deps); os.environ.setdefault("NLTK_DATA", str(pathlib.Path(a.deps) / "nltk_data"))
    for name in ("dashscope", "dotenv", "torch", "tqdm"):  # imported by their modules, not needed for the accuracy
        if name not in sys.modules:
            try:
                __import__(name)
            except ImportError:
                sys.modules[name] = types.ModuleType(name)
    sys.modules["dotenv"].load_dotenv = getattr(sys.modules["dotenv"], "load_dotenv", lambda *x, **k: None)
    sys.modules["dotenv"].find_dotenv = getattr(sys.modules["dotenv"], "find_dotenv", lambda *x, **k: "")
    sys.modules["tqdm"].tqdm = getattr(sys.modules["tqdm"], "tqdm", lambda x, **k: x)
    import nltk
    nltk.download("punkt_tab", download_dir=os.environ.get("NLTK_DATA"), quiet=True)
    sys.path.insert(0, str(repo / "evaluation/src"))
    work = ROOT / "experiments/results/_chen_tmp"; work.mkdir(parents=True, exist_ok=True)
    cwd = os.getcwd(); os.chdir(work)  # their code writes annotation files relative to the working directory
    import vqa_eval as VE
    from utils import extract_questions, generate_and_reorder_annotation_file, get_filename_without_extension
    from vqa import VQA
    sent = []  # (processed reply, reference answers) of every call their code makes to the judge, in order

    def no_judge(pred, answers, *x, **k):  # the judge-model score (paid API) is not computed here
        sent.append((pred, list(map(str, answers))))
        return 0.0
    VE.qwen_evaluate = no_judge
    src = (repo / "evaluation/src/score_with_search_decision_all.py").read_text()
    ns = {}
    exec(compile(src.split('if __name__ == "__main__":')[0], "score_with_search_decision_all.py", "exec"), ns)  # their functions, not their main
    args = types.SimpleNamespace(rag_threshold=4.0)
    os.makedirs("anno_storage", exist_ok=True); os.makedirs("q_storage", exist_ok=True)

    def acc(res_file, anno_from):
        name = get_filename_without_extension(str(anno_from))
        with contextlib.redirect_stdout(io.StringIO()):
            annotation = generate_and_reorder_annotation_file(str(anno_from), f"anno_storage/{name}.json")
            question = extract_questions(str(anno_from), f"q_storage/{name}.json")
            vqa = VQA(annotation, question)
            out = VE.VQAEval(vqa, vqa.loadRes(resFile=str(res_file), quesFile=question), n=2).evaluate()
        return out["acc"]
    rows, judge = [], []
    for s, gates in GATES.items():
        D = repo / "evaluation/eval_data" / s
        with contextlib.redirect_stdout(io.StringIO()) as warn:
            dec = {g: ns["get_search_decision"](str(D / f), args) for g, f in gates.items()}
        n_warn = warn.getvalue().count("Warning")
        soft_raw = [json.loads(l)["search_decision"] for l in open(D / gates["soft"])]
        for r in READERS:
            rag, norag = D / f"{r}_rag_{s}.jsonl", D / f"{r}_norag_{s}.jsonl"
            if not rag.exists():
                continue
            n0 = len(sent)
            a_rag, a_no = acc(rag, rag), acc(norag, rag)  # as their script: annotations and questions come from the search file
            assert len(a_rag) == len(a_no) == len(dec["hard"]), (s, r, len(a_rag), len(a_no))
            assert len(sent) - n0 == 2 * len(a_rag)  # one judge call per answer, search answers first
            if s == "mix":
                judge += [dict(reader=r, cond=c, row=i, pred=sent[n0 + k * len(a_rag) + i][0], answers=sent[n0 + k * len(a_rag) + i][1])
                          for k, c in enumerate(("rag", "norag")) for i in range(len(a_rag))]
            for i in range(len(a_rag)):
                rows.append(dict(set=s, reader=r, row=i, source=SOURCES[i // 100] if s == "mix" else "life", acc_norag=round(float(a_no[i]), 6),
                                 acc_rag=round(float(a_rag[i]), 6), **{g: int(dec[g][i]) for g in gates}, soft_score=soft_raw[i]))
        print(s, "decision-parser warnings:", n_warn, "| soft score types:", {type(x).__name__ for x in soft_raw}, flush=True)
    os.chdir(cwd)
    d = pd.DataFrame(rows)
    d.to_csv(ROOT / "experiments/results/chen_logs__questions.csv", index=False, lineterminator="\n")
    print(d.groupby(["set", "reader"]).size().to_string())
    if a.judge_inputs:
        out = pathlib.Path(a.judge_inputs); out.mkdir(parents=True, exist_ok=True)
        (out / "judge_inputs.jsonl").write_text("".join(json.dumps(j, ensure_ascii=False) + "\n" for j in judge))
        print("judge inputs:", len(judge), "->", out / "judge_inputs.jsonl")


if __name__ == "__main__":
    main()
