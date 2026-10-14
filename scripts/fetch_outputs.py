"""Download the released per-query model outputs into experiments/runs/ (about 1.2 GB; hidden states are most of it).

  python scripts/fetch_outputs.py [--no_hidden]    then: make score confirm extend external cvqa cvqa2 results

Source: https://huggingface.co/datasets/AnvitaPrasad/vgate-outputs (question ids and model outputs only; no benchmark
content). --no_hidden skips the hidden states (the gates that use them cannot be refitted then).
"""
import argparse
import pathlib

from huggingface_hub import snapshot_download

ROOT = pathlib.Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no_hidden", action="store_true")
    a = ap.parse_args()
    out = snapshot_download("AnvitaPrasad/vgate-outputs", repo_type="dataset", local_dir=ROOT / "experiments/runs",
                            ignore_patterns=["README.md", ".gitattributes"] + (["*/hidden*"] if a.no_hidden else []))
    print("outputs in", out)
