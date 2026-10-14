"""Leakage guard: gate features must never include retrieval-success or gold-entity fields."""
import json
import pathlib

import pytest

from vgate.data import FEATURE_BLACKLIST

PROC = pathlib.Path(__file__).resolve().parents[1] / "data/processed"


@pytest.mark.parametrize("name", ["lab_gold_in_top1", "Recall@5", "Pseudo Recall@1", "MRR", "pos_item_ids",
                                  "entity_id", "entity_text", "wiki_title", "oracle_text", "answers", "gold_answer"])
def test_blacklist_catches_leaky_columns(name):
    assert FEATURE_BLACKLIST.match(name)


@pytest.mark.parametrize("name", ["tok_entropy_mean", "min_logprob", "consistency", "ent_consistency", "p_true",
                                  "qpp_margin", "qpp_entropy", "qpp_nqc", "n_prompt"])
def test_blacklist_allows_feature_names(name):
    assert not FEATURE_BLACKLIST.match(name)


@pytest.mark.skipif(not (PROC / "evqa.jsonl").exists(), reason="processed data not built")
def test_processed_records_keep_labels_prefixed():
    rec = json.loads(open(PROC / "evqa.jsonl").readline())
    leaky = [k for k in rec if k.startswith(("Recall", "Pseudo", "MRR", "pos_item"))]
    assert not leaky, leaky
    assert {"lab_gold_in_top1", "lab_gold_in_top5"} <= set(rec)


def test_gate_feature_lists_pass_the_blacklist():
    """The lists the gates are actually fitted on contain no retrieval-success, gold or oracle field."""
    from vgate import uncertainty as U
    from vgate.data import FEATURE_BLACKLIST
    used = U.FREE + U.SAMPLE + U.ENTITY + U.PTRUE + U.PARA + U.QPP
    assert not [f for f in used if FEATURE_BLACKLIST.match(f)]
