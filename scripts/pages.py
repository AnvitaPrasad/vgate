"""Page and word count of the built paper: python scripts/pages.py  (the limit is 4 pages including references)."""
import pathlib
from pypdf import PdfReader
ROOT = pathlib.Path(__file__).resolve().parents[1]
pages = [p.extract_text() for p in PdfReader(ROOT / "paper/main.pdf").pages]
print("pages:", len(pages), "| words per page:", [len(p.split()) for p in pages])
if len(pages) > 4:
    print("--- page 5 ---"); print(pages[4][:1500])
