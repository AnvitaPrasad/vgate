# Data

`data/raw/` and `data/processed/` are git-ignored; `data/splits/` (frozen, sha256-checked) is tracked.

| source | contents | size |
|---|---|---|
| HF `BByrneLab/EVQA-test-with-PreFLMR-Lretrieval` | E-VQA test (3,750 single-hop), gold section, 500 PreFLMR retrievals | 13.5 MB |
| HF `BByrneLab/multi_task_multi_modal_knowledge_retrieval_benchmark_M2KR` | `EVQA_data`/`EVQA_passages` (51,472 passages) test; `Infoseek_data` test (answer_eval, ranges, types) | ~60 MB |
| ReflectiVA release `evqa_inference_images.zip` | E-VQA test images (iNaturalist + Google Landmarks v2) | 392 MB |
| HF `BByrneLab/Infoseek-test-with-EvaClip-retrieval` | InfoSeek val (4,637), gold page, 100 EVA-CLIP retrievals | 8.5 MB |
| HF `BByrneLab/Infoseek-EchoSight-100K-KB` | 100K Wikipedia articles (sections), used for evidence text | 293 MB |
| HF `BByrneLab/BERAG-demo-InfoSeek-images` | InfoSeek val images (1,689) | 225 MB |

Expected layout: `data/raw/{evqa_preflmr,m2kr,evqa_images,infoseek_evaclip,infoseek_kb,infoseek_images/val}`.
`python -m vgate.data` builds `data/processed/{evqa,infoseek}.jsonl` and the image-grouped splits (seed 2027):
dev = 1,000 per dataset, confirm = the rest (2,750 / 3,637). Known quirk: 10 M2KR E-VQA passage_ids are duplicated
with different text; the first occurrence is kept.

OK-VQA (added 2026-10-04): M2KR `OKVQA_data` / `OKVQA_passages` test parquets (5,046 questions, 114,809 passages), the
COCO val2014 images of those questions under `data/raw/okvqa_images/val2014`, and `data/raw/okvqa_retrieval.parquet`
(bge-base-en-v1.5 top-20, built by `kaggle/okvqa-retrieval`). `python -m vgate.data okvqa` builds
`data/processed/okvqa.jsonl` and its splits (dev 1,000 / confirm 4,046).

## Terms of the sources (see `docs/strengthening/R4_release.md` for quotes and URLs)
Nothing under `data/raw` or `data/processed` is redistributed by this repository; fetch it from the sources.
- **Images.** iNaturalist 2021 (about half of E-VQA): non-commercial research and education only, and "You will NOT
  distribute the dataset images". Google Landmarks v2 `train` (the other half of E-VQA): CC BY per image, credit from
  `train_attribution.csv`. InfoSeek images come from OVEN: no per-image licence is stated. COCO val2014 (OK-VQA):
  Flickr terms per image.
- **Questions and answers.** InfoSeek: repository under Apache-2.0. E-VQA and OK-VQA: no licence is stated. M2KR
  repackages all three under an MIT tag; the upstream terms are not discussed there.
- **Passages.** Wikipedia text, CC BY-SA (3.0 Unported via WIT for E-VQA / InfoSeek; the OK-VQA dump date is unknown).
- **Models.** Qwen3-VL, InternVL3.5 and BEM: Apache-2.0. bge-base-en-v1.5: MIT. Qwen2.5-VL-3B: Qwen Research
  License (non-commercial).
