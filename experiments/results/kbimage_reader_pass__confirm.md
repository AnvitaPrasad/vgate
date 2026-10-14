# PREREG v6d — readers on the image-re-ranked evidence (preregistered; E-VQA held-out split, n = 2750)

Gold passage among the five read: 60.0% with the retriever's order, 67.5% re-ranked.

Expectations: E1 held; E2 held; E3 held.

## Qwen3-VL-4B-Instruct

Never-retrieve 23.1%; always-retrieve with the retriever's top-5 51.2%, with the re-ranked five 55.6%: **+4.44 [+3.07, +5.84]** (95% image-cluster interval). Retrieval helps 32.9% -> 36.7% and harms 4.7% -> 4.1% of questions (harm: -0.62 [-1.16, -0.11]).
By photograph: inaturalist (n = 1413) +5.38 [+3.58, +7.14]; landmarks (n = 1337) +3.44 [+1.39, +5.62].
Where (share answered correctly with the retriever's five -> with the re-ranked five): gold gained (n = 267) 28.1 -> 58.4; gold lost (n = 60) 55.0 -> 31.7; gold in both (n = 1589) 70.9 -> 74.0; gold in neither (n = 834) 20.9 -> 21.5.
Planner: AUROC floor / needed for a one-point lead 0.606 / 0.881 with the retriever's five, 0.622 / 0.905 with the re-ranked five.

## Qwen2.5-VL-3B-Instruct

Never-retrieve 22.3%; always-retrieve with the retriever's top-5 48.6%, with the re-ranked five 52.8%: **+4.22 [+2.87, +5.62]** (95% image-cluster interval). Retrieval helps 31.0% -> 34.8% and harms 4.7% -> 4.3% of questions (harm: -0.36 [-0.91, +0.18]).
By photograph: inaturalist (n = 1413) +5.52 [+3.67, +7.31]; landmarks (n = 1337) +2.84 [+0.87, +4.83].
Where (share answered correctly with the retriever's five -> with the re-ranked five): gold gained (n = 267) 28.5 -> 51.7; gold lost (n = 60) 46.7 -> 23.3; gold in both (n = 1589) 66.7 -> 70.9; gold in neither (n = 834) 20.7 -> 21.0.
Planner: AUROC floor / needed for a one-point lead 0.607 / 0.879 with the retriever's five, 0.617 / 0.897 with the re-ranked five.

