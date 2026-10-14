"""CVQA (PREREG v7): multiple-choice prompts and letter scoring; the earlier prompt ids must not move."""
from vgate import cvqa, prompts
from vgate.evaluation import mc_correct, mc_letter

REC = dict(question="What is the yellow building?", options=["a hotel", "place for lifeguards", "a store", "a restaurant"],
           evidence=[dict(title="T", text="some text")])


def test_prompt_ids_unchanged():
    assert (prompts.PROMPT_ID, prompts.PROMPT_ID_V3, prompts.PROMPT_ID_V5) == ("217aa02e", "c52a123f", "cd0e17b7")


def test_mc_prompts():
    ne, ev = prompts.build("ne", REC), prompts.build("ev", REC)
    assert ne.endswith("A. a hotel\nB. place for lifeguards\nC. a store\nD. a restaurant") and ne.startswith(prompts.MC)
    assert ev.startswith("Context:\n[1] T: some text") and "Question: What is the yellow building?" in ev
    assert prompts.build("ent", REC) == prompts.ENT
    assert prompts.build("ne", dict(question="q")) == prompts.NE.format(question="q")  # open-ended records are untouched


def test_mc_letter():
    o = REC["options"]
    assert [mc_letter(x, o) for x in ("B", "B.", "(B)", " b", "B. place for lifeguards", "Answer: C", "A restaurant", "a hotel", "")] == \
        ["B", "B", "B", "", "B", "", "D", "A", ""]
    assert mc_correct("D) a restaurant", "D", o) == 1 and mc_correct("A restaurant", "A", o) == 0


def test_page_parts():
    c = {"heading": ["Introduction", "History"], "content": [["first", " second "], ["third", ""]]}
    assert cvqa.page_parts(c) == ["first", "second", "third"] and cvqa.page_parts("plain") == ["plain"] and cvqa.page_parts(None) == []


def test_chunks_for_scripts_without_spaces():
    en = " ".join(f"w{i}" for i in range(250))
    assert [len(c.split()) for c in cvqa.chunks_of(en)] == [120, 120, 10]
    zh = "\u4e2d\u6587" * 600  # 1,200 characters, no spaces
    c = cvqa.chunks_of(zh)
    assert len(c) == 5 and all(len(v) == 240 for v in c) and "".join(c) == zh
    assert cvqa.units("")[0] == []


def test_published_gate_prompts():
    r = dict(question="Who designed this building?")
    c, u = prompts.build("gate_chen", r), prompts.build("gate_urag", r)
    assert c.startswith("You are an assistant capable of deciding whether a search is needed") and c.endswith("Text question: Who designed this building?\nImage: ")
    assert u.startswith("Classify the following query into one or more categories from: [no, paragraph, document, table, image, clip, video]")
    assert "Classify the following query: Who designed this building?\nProvide only the category or categories combined with '+'." in u
    assert set(prompts.GATE_SPECS) == {"gate_chen", "gate_urag"}
    assert (prompts.PROMPT_ID, prompts.PROMPT_ID_V3, prompts.PROMPT_ID_V5, prompts.PROMPT_ID_V7) == ("217aa02e", "c52a123f", "cd0e17b7", "24104ae8")
