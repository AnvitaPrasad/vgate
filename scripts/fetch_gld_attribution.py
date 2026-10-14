"""Image credits for the Google Landmarks v2 images used by E-VQA (the only benchmark images here that may be shown
publicly: "All images in the train set have CC-BY licenses", https://github.com/cvdfoundation/google-landmark).

Streams the official `train_attribution.csv` (about 1 GB, not stored) and keeps the rows of our images:
  python scripts/fetch_gld_attribution.py   ->  data/gld_attribution_evqa.csv  (id, url, author, license, title)
"""
import csv
import io
import json
import pathlib
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[1]
URL = "https://s3.amazonaws.com/google-landmark/metadata/train_attribution.csv"


def main():
    want = set()
    for line in open(ROOT / "data/processed/evqa.jsonl"):
        f = json.loads(line)["image_file"]
        if "google-landmark/train/" in f:
            want.add(pathlib.Path(f).stem)
    out, found = ROOT / "data/gld_attribution_evqa.csv", 0
    with urllib.request.urlopen(URL, timeout=120) as resp, open(out, "w", newline="") as fo:
        reader = csv.reader(io.TextIOWrapper(resp, encoding="utf-8", newline=""))
        w = csv.writer(fo)
        w.writerow(next(reader))  # header: id,url,author,license,title
        for row in reader:
            if row and row[0] in want:
                w.writerow(row)
                found += 1
                if found == len(want):
                    break
    print(f"{found} of {len(want)} E-VQA landmark images found -> {out}")


if __name__ == "__main__":
    main()
