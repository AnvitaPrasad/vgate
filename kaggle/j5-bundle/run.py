"""CPU-only: bundle J5's needed outputs (ne/ev JSONL, meta, hidden-state qids + layer 27) into one tar.gz."""
import glob, os, subprocess
src = os.path.dirname(glob.glob("/kaggle/input/**/J5/evqa__InternVL3_5-4B-HF__ne.jsonl", recursive=True)[0])
keep = [f for f in os.listdir(src) if f.endswith((".jsonl", ".json")) or "__L27_" in f]
print(src, keep, flush=True)
subprocess.run(["tar", "czf", "/kaggle/working/j5_bundle.tgz", "-C", src] + keep, check=True)
print(os.path.getsize("/kaggle/working/j5_bundle.tgz"), "bytes")
