"""Tests for vgate.evaluation on real InfoSeek (M2KR) / E-VQA rows. CPU only, < 1 min."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vgate import evaluation as ev  # noqa: E402


@pytest.fixture(scope="module")
def refs():
    try:
        return ev.load_infoseek_refs()
    except FileNotFoundError:  # a clone without the benchmark files (data/README.md says how to fetch them)
        pytest.skip("InfoSeek reference file not downloaded")


@pytest.fixture(scope="module")
def evqa():
    if not ev.EVQA_PARQUET.exists() if hasattr(ev.EVQA_PARQUET, "exists") else not __import__("os").path.exists(ev.EVQA_PARQUET):
        pytest.skip("E-VQA question file not downloaded")
    df = pd.read_parquet(ev.EVQA_PARQUET, columns=["question_id", "question", "answers", "gold_answer"])
    return df.set_index("question_id")


# ----------------------------------------------------------------------------- normalize
@pytest.mark.parametrize("raw,out", [
    ("Answer: Paris.", "Paris"),
    ("answer:  the U.S.\nBecause the image shows...", "the U.S"),
    ("<think>it is 1990? maybe</think>\nAnswer: 1991.", "1991"),
    ("<think>never closed", ""),
    ("  42 metres  ", "42 metres"),
    (None, ""),
])
def test_normalize_prediction(raw, out):
    assert ev.normalize_prediction(raw) == out


# ----------------------------------------------------------------------------- InfoSeek
def test_infoseek_string(refs):
    r = refs["infoseek_val_00000074"]  # regulator of handball; answer_eval = [IHF, Intl Handball Fed.]
    assert r["question_type"] == "String"
    assert ev.infoseek_correct("International Handball Federation", r) == 1
    assert ev.infoseek_correct("Answer: the IHF.", r) == 1  # alias + normalisation
    assert ev.infoseek_correct("FIFA", r) == 0
    r = refs["infoseek_val_00000070"]  # country of origin; many US aliases
    assert ev.infoseek_correct("United States of America", r) == 1
    assert ev.infoseek_correct("Canada", r) == 0


def test_infoseek_time(refs):
    r = refs["infoseek_val_00001448"]  # answer_eval = ['1813', '1815', '1814'] (+/-1 yr baked in)
    assert r["question_type"] == "Time"
    assert ev.infoseek_correct("1814", r) == 1
    assert ev.infoseek_correct("1815", r) == 1
    assert ev.infoseek_correct("1820", r) == 0
    r = refs["infoseek_val_00000562"]  # full date question
    assert ev.infoseek_correct("20 December 1990", r) == 1
    assert ev.infoseek_correct("21 December 1990", r) == 0


def test_infoseek_numerical(refs):
    r = refs["infoseek_val_00001309"]  # bridge length 237 m, wikidata_range [213.3, 260.7]
    assert r["question_type"] == "Numerical" and len(r["answer_eval"]) == 0
    assert ev.infoseek_correct("237", r) == 1
    assert ev.infoseek_correct("about 250 metres", r) == 1
    assert ev.infoseek_correct("220-250 m", r) == 1   # range inside answer range
    assert ev.infoseek_correct("300", r) == 0
    assert ev.infoseek_correct("100-150", r) == 0     # disjoint range
    assert ev.infoseek_correct("no idea", r) == 0     # -> [0, 0]
    # wikidata-bounded range [61, 63] (not +/-10%): 64 would pass a +/-10% rule but must fail here
    r = refs["infoseek_val_00024503"]
    assert list(r["wikidata_range"]) == [61.0, 63.0] and r["wikidata_value"] == 62.0
    assert ev.infoseek_correct("62", r) == 1
    assert ev.infoseek_correct("64", r) == 0


def test_numerical_reference_formats():
    # raw official format and scalar-only fallback (+/-10%)
    raw = {"question_type": "Numerical", "answer_eval": [{"wikidata": 62.0, "range": [61.0, 63.0]}]}
    assert ev.infoseek_correct("63", raw) == 1 and ev.infoseek_correct("64", raw) == 0
    scalar = {"question_type": "Numerical", "wikidata_range": None, "wikidata_value": 10.0}
    assert ev.infoseek_correct("10.9", scalar) == 1 and ev.infoseek_correct("11.5", scalar) == 0


def test_process_numerical_answer_official_quirks():
    assert ev.process_numerical_answer("9-10") == [9.0, 10.0]
    assert ev.process_numerical_answer("1,000 m") == 1000.0
    assert ev.process_numerical_answer("10 to 5") == 10.0  # descending pair -> first value
    assert ev.process_numerical_answer("unknown") == [0, 0]


def test_infoseek_score(refs):
    uq = ["infoseek_val_00000074", "infoseek_val_00001448", "infoseek_val_00001309"]
    ue = ["infoseek_val_00024503"]
    preds = {uq[0]: "IHF", uq[1]: "1900", uq[2]: "237", ue[0]: "62"}
    out = ev.infoseek_score(preds, {q: refs[q] for q in uq + ue})
    assert out["unseen_question_score"]["score"] == pytest.approx(66.67)
    assert out["unseen_question_score"]["score_time"] == 0.0
    assert out["unseen_entity_score"]["score"] == 100.0
    assert out["overall"]["score"] == 75.0 and out["overall"]["n"] == 4
    assert out["final_score"] == pytest.approx(round(2 / (1 / 66.67 + 1 / 100.0), 2), abs=0.01)


# ----------------------------------------------------------------------------- E-VQA
class _NoBEM:
    def score(self, *a):
        raise AssertionError("BEM should not run when EM hits")


def test_evqa_em_path(evqa):
    r = evqa.loc["EVQA_0"]
    assert ev.evqa_correct(r.gold_answer, r.question, r.answers, scorer=_NoBEM()) == (1, 1.0)
    # trailing period / case / leading "Answer:" handled by normalisation + preprocess_answer
    pred = "Answer: Help urinary tract infections and as an antioxidant"
    assert ev.evqa_correct(pred, r.question, r.answers, scorer=_NoBEM()) == (1, 1.0)
    r = evqa.loc["EVQA_5"]  # 'approximately 2 weeks': official digit map makes 'two' == '2'
    assert ev.evqa_correct("approximately two weeks", r.question, r.answers, scorer=_NoBEM())[0] == 1


def test_evqa_bem_path(evqa):
    pytest.importorskip("torch")
    scorer = ev.BEMScorer("cpu")
    # card / TF-Hub colab examples
    s = scorer.score(["why is the sky blue", "What does Ban Bossy encourage?"],
                     ["light scattering", "leadership in girls"],
                     ["scattering of light", "positions of power"])
    assert s[0] > 0.9 and s[1] < 0.5
    r = evqa.loc["EVQA_1"]  # reordered list: EM miss, BEM hit
    c, sc = ev.evqa_correct("north america, asia, europe", r.question, r.answers, scorer)
    assert c == 1 and 0.5 <= sc < 1.0
    rows = evqa.loc[[f"EVQA_{i}" for i in range(10)]]
    c, sc = ev.evqa_correct_batch(["banana"] * 10, list(rows.question), list(rows.answers), scorer)
    assert c.sum() == 0 and sc.max() < 0.5
    # batching must not change scores beyond float noise
    single = np.array([ev.evqa_correct("banana", q, a, scorer)[1] for q, a in zip(rows.question, rows.answers)])
    np.testing.assert_allclose(single, sc, atol=1e-5)


def test_okvqa_soft_accuracy_official_definition():
    from vgate.evaluation import okvqa_correct, okvqa_soft_acc
    answers = ["race"] * 4 + ["motocross"] * 4 + ["ride"] * 2
    assert okvqa_soft_acc("race", answers) == 1.0            # >= 3 matches in every leave-one-out subset
    assert okvqa_soft_acc("Ride.", answers) == pytest.approx((8 * (2 / 3) + 2 * (1 / 3)) / 10)
    assert okvqa_soft_acc("banana", answers) == 0.0
    assert okvqa_correct("ride", answers)[0] == 1 and okvqa_correct("banana", answers)[0] == 0
    assert okvqa_correct("x", ["x"] + ["y"] * 9)[0] == 0      # one annotator only -> soft acc 0.3 < 0.6


def test_okvqa_relaxed_sensitivity_label():
    from vgate.evaluation import okvqa_relaxed
    answers = ["cake"] * 6 + ["torte"] * 2 + ["dessert", "pie"]
    assert okvqa_relaxed("chocolate cake", answers, 0) == 1     # a >= 2-vote answer is contained in the prediction
    assert okvqa_relaxed("tort", answers, 0) == 1               # prediction contained in a >= 2-vote answer
    assert okvqa_relaxed("pie crust", answers, 0) == 0          # "pie" has one vote only
    assert okvqa_relaxed("banana", answers, 0) == 0
    assert okvqa_relaxed("no", ["no"] * 10, 0) == 0             # shorter than 3 characters: no containment credit
    assert okvqa_relaxed("banana", answers, 1) == 1             # never removes an officially correct label
