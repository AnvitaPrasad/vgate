# PREREG v6c — readers on the image-re-ranked evidence (pilot; E-VQA development split, n = 1000)

Gold passage among the five read: 57.7% with the retriever's order, 68.0% re-ranked.

Expectations: E1 held; E2 held; E3 held.

## Qwen3-VL-4B-Instruct

Never-retrieve 26.7%; always-retrieve with the retriever's top-5 52.9%, with the re-ranked five 59.3%: **+6.40 [+4.99, +7.76]** (80% image-cluster interval). Retrieval helps 31.7% -> 36.9% and harms 5.5% -> 4.3% of questions (harm: -1.20 [-1.85, -0.58]).
By photograph: inaturalist (n = 587) +5.45 [+3.69, +7.22]; landmarks (n = 413) +7.75 [+5.36, +10.02].
Where (share answered correctly with the retriever's five -> with the re-ranked five): gold gained (n = 118) 24.6 -> 56.8; gold lost (n = 15) 60.0 -> 40.0; gold in both (n = 562) 74.4 -> 78.8; gold in neither (n = 305) 23.9 -> 25.2.
Planner: AUROC floor / needed for a one-point lead 0.591 / 0.857 with the retriever's five, 0.616 / 0.9 with the re-ranked five.

## Qwen2.5-VL-3B-Instruct

Never-retrieve 24.6%; always-retrieve with the retriever's top-5 48.5%, with the re-ranked five 54.3%: **+5.80 [+4.29, +7.28]** (80% image-cluster interval). Retrieval helps 30.6% -> 36.0% and harms 6.7% -> 6.3% of questions (harm: -0.40 [-1.02, +0.20]).
By photograph: inaturalist (n = 587) +5.45 [+3.80, +7.11]; landmarks (n = 413) +6.30 [+3.70, +8.88].
Where (share answered correctly with the retriever's five -> with the re-ranked five): gold gained (n = 118) 26.3 -> 44.9; gold lost (n = 15) 60.0 -> 33.3; gold in both (n = 562) 68.5 -> 72.8; gold in neither (n = 305) 19.7 -> 24.9.
Planner: AUROC floor / needed for a one-point lead 0.575 / 0.824 with the retriever's five, 0.579 / 0.846 with the re-ranked five.

