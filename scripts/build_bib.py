"""Build paper/references.bib from arXiv metadata (title/authors/year from the arXiv API, never typed by hand).

Venue: taken from research/literature/literature_database.csv only when that row is `verified` and the venue is
a real venue (not UNKNOWN / preprint / "from memory"); otherwise the entry is cited as an arXiv preprint.
  python scripts/build_bib.py 2306.09224 2302.11713 ...   (or no args: uses CITE below)
"""
import pathlib
import re
import sys
import time
import urllib.request
import xml.etree.ElementTree as ET

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
CITE = """2306.09224 2302.11713 2402.08327 2501.12835
2607.24010 2604.07985 2411.16863 2502.18023 2504.10074 2510.22694 2603.00511
2609.12437 1906.00067 2606.16494 2504.20734 2601.07206 2511.22715
2406.05967 2512.05959
2511.09803 2605.06350 2610.01535 2405.18740""".split()  # CVQA and M4-RAG (PREREG v7); last line: prior work on what a gate or router
# can earn (docs/strengthening/R14_gate_bound_novelty.md; abstract pages and the cited passages re-read on 2026-10-07)
NS = {"a": "http://www.w3.org/2005/Atom"}
# Non-arXiv entries; fields copied from the Crossref record of the DOI (checked 2026-09-30; Provost and Fawcett: 2026-10-06;
# Hauff et al.: 2026-10-07, the volume is the ECIR 2010 proceedings).
# DOIs (left out of the 4-page review version): Shtok et al. 10.1145/2180868.2180873; Provost and Fawcett 10.1023/A:1007601015854;
# Hauff et al. 10.1007/978-3-642-12275-0_20.
# Entries with a proceedings venue that are printed without their arXiv id in the review version (space); the id stays in
# the unprinted field `arxivid`. M4-RAG (CVPR 2026), InfoSeek (EMNLP 2023), Dado et al. (CIKM 2026).
VENUE_ONLY = {"2512.05959", "2302.11713", "2604.07985"}
EXTRA = r"""@article{shtok2012predicting,
  title = {{Predicting Query Performance by Query-Drift Estimation}},
  author = {Anna Shtok and Oren Kurland and David Carmel and Fiana Raiber and Gad Markovits},
  journal = {ACM Transactions on Information Systems},
  volume = {30},
  number = {2},
  pages = {1--35},
  year = {2012}
}

@article{provost2001robust,
  title = {{Robust Classification for Imprecise Environments}},
  author = {Foster Provost and Tom Fawcett},
  journal = {Machine Learning},
  volume = {42},
  number = {3},
  pages = {203--231},
  year = {2001}
}

@inproceedings{hauff2010query,
  title = {{Query Performance Prediction: Evaluation Contrasted with Effectiveness}},
  author = {Claudia Hauff and Leif Azzopardi and Djoerd Hiemstra and Franciska de Jong},
  booktitle = {ECIR 2010},
  pages = {204--216},
  year = {2010}
}"""


def fetch(ids):
    out = {}
    for i in range(0, len(ids), 20):
        url = "http://export.arxiv.org/api/query?max_results=50&id_list=" + ",".join(ids[i:i + 20])
        root = ET.fromstring(urllib.request.urlopen(url, timeout=60).read())
        for e in root.findall("a:entry", NS):
            aid = re.sub(r"v\d+$", "", e.find("a:id", NS).text.rsplit("/", 1)[-1])
            out[aid] = dict(title=" ".join(e.find("a:title", NS).text.split()),
                            authors=[a.find("a:name", NS).text for a in e.findall("a:author", NS)],
                            year=e.find("a:published", NS).text[:4])
        time.sleep(3)  # arXiv API etiquette
    return out


def venue_for(db, aid):
    row = db[db.arxiv.astype(str).str.contains(aid, regex=False)]
    if row.empty or row.iloc[0].confidence != "verified":
        return None
    v = str(row.iloc[0].venue)
    if re.search(r"UNKNOWN|preprint|memory|arXiv", v, re.I):
        return None
    return re.sub(r"\s*\(.*\)", "", v).strip()


def key(meta, aid):
    last = re.sub(r"[^A-Za-z]", "", meta["authors"][0].split()[-1]).lower()
    word = next((w for w in re.findall(r"[A-Za-z]+", meta["title"]) if len(w) > 3), "x").lower()
    return f"{last}{meta['year']}{word}"


MAX_AUTHORS = 3  # longer author lists are printed as the first three + "et al." (4-page limit)


def cap_authors(names):
    names = [" ".join(w.capitalize() if w.islower() and w not in ("de", "van", "von", "der", "others") else w for w in n.split()) for n in names]
    return names if len(names) <= MAX_AUTHORS else names[:3] + ["others"]


def recap():
    """Apply cap_authors to the existing references.bib without calling the arXiv API."""
    p = ROOT / "paper/references.bib"
    fix = lambda m: "  author = {" + " and ".join(cap_authors(m.group(1).split(" and "))) + "},"
    p.write_text(re.sub(r"^  author = \{(.*)\},$", fix, p.read_text(), flags=re.M))


def prune():
    """Keep only the entries whose arXiv id is in CITE (plus the non-arXiv EXTRA), without calling the arXiv API.
    Ids in CITE that have no entry yet are reported; a normal run fetches them."""
    p = ROOT / "paper/references.bib"
    head, *ents = re.split(r"\n\n(?=@)", p.read_text().strip())
    eid = lambda e: (re.search(r"(?:eprint|arxivid) = \{(.*?)\}", e) or [None, None])[1]
    keep = [e for e in ents if eid(e) is None or eid(e) in CITE]
    p.write_text(head + "\n\n" + "\n\n".join(keep) + "\n")
    print(f"{len(keep)} entries kept; not yet fetched: {sorted(set(CITE) - {eid(e) for e in keep})}")


def main():
    if sys.argv[1:] == ["--recap"]:
        return recap()
    if sys.argv[1:] == ["--prune"]:
        return prune()
    ids = sys.argv[1:] or CITE
    db = pd.read_csv(ROOT / "research/literature/literature_database.csv")
    meta = fetch(ids)
    missing = sorted(set(ids) - set(meta))
    entries, keys = [], {}
    for aid in ids:
        if aid not in meta:
            continue
        m, v = meta[aid], venue_for(db, aid)
        esc = lambda x: x.replace("&", "\\&") if x else x
        m = dict(m, title=esc(m["title"])); v = esc(v)
        k = key(m, aid)
        keys[aid] = k
        vy = re.search(r"(19|20)\d\d", v or "")  # an entry with a proceedings venue is dated by the venue (the key keeps the arXiv year)
        m = dict(m, year=vy.group(0)) if vy else m
        authors = " and ".join(cap_authors(m["authors"]))
        title = "{" + m["title"] + "}"
        if v and aid in VENUE_ONLY:
            entries.append(f"@inproceedings{{{k},\n  title = {{{title}}},\n  author = {{{authors}}},\n"
                           f"  booktitle = {{{v}}},\n  year = {{{m['year']}}},\n  arxivid = {{{aid}}}\n}}")
        elif v:
            entries.append(f"@inproceedings{{{k},\n  title = {{{title}}},\n  author = {{{authors}}},\n"
                           f"  booktitle = {{{v}}},\n  year = {{{m['year']}}},\n  eprint = {{{aid}}},\n"
                           f"  archivePrefix = {{arXiv}}\n}}")
        else:
            entries.append(f"@misc{{{k},\n  title = {{{title}}},\n  author = {{{authors}}},\n  year = {{{m['year']}}},\n"
                           f"  eprint = {{{aid}}},\n  archivePrefix = {{arXiv}}\n}}")
    head = "% Generated by scripts/build_bib.py from the arXiv API; venues only from verified literature-DB rows.\n\n"
    (ROOT / "paper/references.bib").write_text(head + "\n\n".join(entries + [EXTRA]) + "\n")
    (ROOT / "paper/cite_keys.tsv").write_text("".join(f"{a}\t{k}\n" for a, k in keys.items()))
    recap()  # the cap also applies to the entries of EXTRA
    print(f"{len(entries)} entries written; missing from arXiv API: {missing}")


if __name__ == "__main__":
    main()
