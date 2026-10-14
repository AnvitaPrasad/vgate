"""Plain-text abstract for the CMT form, expanded from paper/sections/abstract.tex + paper/tables/numbers.tex."""
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[1]
mac = dict(re.findall(r"\\newcommand\{\\(num\w+)\}\{(.*)\}", (ROOT / "paper/tables/numbers.tex").read_text()))
t = (ROOT / "paper/sections/abstract.tex").read_text()
t = re.sub(r"\\(num\w+)(\{\})?", lambda m: mac[m.group(1)], t)
t = t.replace("$-$", "-").replace("$+$", "+").replace("\\sys{}", "VGate").replace("\\%", "%").replace("---", " - ").replace("--", "-")
t = re.sub(r"\\emph\{([^}]*)\}", r"\1", t)
t = " ".join(t.split())
print(t)
(ROOT / "docs/submission/abstract_plain.txt").write_text(t + "\n")
