# Loss decomposition (confirm split; PREREG v3a; accuracy points, 95% image-cluster bootstrap)

## Qwen3-VL-4B-Instruct

| | E-VQA | InfoSeek |
|---|---|---|
| never | 23.1 [21.4, 24.7] | 17.8 [16.4, 19.1] |
| always (top-5) | 51.2 [48.6, 53.5] | 27.4 [25.8, 29.0] |
| oracle choice of the two answers | 55.9 [53.4, 58.0] | 31.9 [30.3, 33.6] |
| ideal read/skip gate (read iff gold retrieved) | 50.7 [48.0, 53.0] | 27.9 [26.3, 29.5] |
| ideal gate − always | -0.5 [-1.4, 0.4] | 0.5 [-0.1, 1.1] |
| gold in top-5: reading all five | 70.3 [67.4, 72.7] | 33.9 [32.0, 36.0] |
| gold in top-5: gold passage alone | 90.2 [88.6, 91.8] | 47.0 [45.1, 48.9] |
| gold alone − all five | 19.9 [17.6, 22.6] | 13.1 [10.9, 15.3] |
| ideal passage selector | 62.7 [60.1, 65.1] | 36.9 [35.1, 38.6] |
| gold evidence for every question | 86.3 [84.9, 87.7] | 47.1 [45.4, 48.7] |
| gold in top-1 / top-5 (%) | 37.6 / 60.0 | 43.2 / 69.0 |

E-VQA, gold in top-5 and untruncated (n=1145): gold at rank 1 (n=723) 81.9 [78.9, 84.7]; at ranks 2–5 (n=422) 62.1 [57.1, 66.8]; difference 19.8 [14.2, 25.2] (closed-book wrong: 24.3 [17.5, 30.7]; closed-book right: 7.1 [-0.3, 15.0]).
Gold passage alone: rank 1 91.4 [89.1, 93.4]; ranks 2–5 91.7 [88.9, 94.2]; gold alone − all five at ranks 2–5: 29.6 [25.0, 34.3].
Same passage text in both prompts (untruncated; added after the held-out run): gold alone 91.5 [89.7, 93.1], gold alone − all five 16.9 [14.5, 19.6].
E-VQA, gold in top-5: untruncated 74.6 [71.6, 77.3] vs truncated at 160 words 60.5 [55.7, 64.8].
InfoSeek, right page in top-5: an answer string is inside the five chunks in 52.7% of cases; accuracy 54.5 [51.6, 57.4] with it, 11.0 [9.3, 13.0] without.

## Qwen2.5-VL-3B-Instruct

| | E-VQA | InfoSeek |
|---|---|---|
| never | 22.3 [20.6, 24.0] | 17.0 [15.7, 18.3] |
| always (top-5) | 48.6 [46.2, 50.8] | 26.6 [25.0, 28.2] |
| oracle choice of the two answers | 53.3 [50.9, 55.4] | 31.0 [29.4, 32.6] |
| ideal read/skip gate (read iff gold retrieved) | 47.1 [44.5, 49.3] | 26.6 [24.9, 28.1] |
| ideal gate − always | -1.5 [-2.5, -0.6] | 0.0 [-0.6, 0.6] |
| gold in top-5: reading all five | 66.0 [62.8, 68.5] | 33.4 [31.5, 35.3] |
| gold in top-5: gold passage alone | 86.0 [84.3, 87.7] | 48.4 [46.2, 50.4] |
| gold alone − all five | 20.1 [17.7, 22.9] | 15.0 [12.8, 17.3] |
| ideal passage selector | 59.1 [56.8, 61.3] | 36.9 [35.1, 38.8] |
| gold evidence for every question | 81.3 [79.6, 82.8] | 48.5 [46.7, 50.1] |
| gold in top-1 / top-5 (%) | 37.6 / 60.0 | 43.2 / 69.0 |

E-VQA, gold in top-5 and untruncated (n=1145): gold at rank 1 (n=723) 76.2 [73.2, 79.1]; at ranks 2–5 (n=422) 58.3 [52.7, 63.8]; difference 17.9 [11.9, 24.0] (closed-book wrong: 23.9 [16.5, 31.4]; closed-book right: -0.6 [-7.8, 6.8]).
Gold passage alone: rank 1 88.0 [85.3, 90.6]; ranks 2–5 86.0 [82.5, 89.3]; gold alone − all five at ranks 2–5: 27.7 [22.4, 32.9].
Same passage text in both prompts (untruncated; added after the held-out run): gold alone 87.2 [85.2, 89.3], gold alone − all five 17.6 [14.9, 20.6].
E-VQA, gold in top-5: untruncated 69.6 [66.4, 72.5] vs truncated at 160 words 57.7 [52.3, 62.4].
InfoSeek, right page in top-5: an answer string is inside the five chunks in 52.7% of cases; accuracy 53.1 [50.1, 55.9] with it, 11.4 [9.7, 13.2] without.

## InternVL3_5-4B-HF

| | E-VQA | InfoSeek |
|---|---|---|
| never | 19.9 [18.3, 21.7] | 9.5 [8.4, 10.6] |
| always (top-5) | 53.6 [51.1, 55.9] | 19.6 [18.1, 21.0] |
| oracle choice of the two answers | 57.7 [55.2, 59.9] | 23.1 [21.6, 24.7] |
| ideal read/skip gate (read iff gold retrieved) | 51.0 [48.2, 53.3] | 19.1 [17.6, 20.4] |
| ideal gate − always | -2.7 [-3.7, -1.8] | -0.5 [-1.1, 0.1] |
| gold in top-5: reading all five | 73.1 [70.6, 75.5] | 24.3 [22.4, 26.0] |
| gold in top-5: gold passage alone | 90.1 [88.6, 91.6] | 37.2 [35.3, 39.2] |
| gold alone − all five | 17.0 [14.5, 19.6] | 12.9 [10.8, 15.2] |
| ideal passage selector | 61.1 [58.9, 63.3] | 28.0 [26.3, 29.6] |
| gold evidence for every question | 87.7 [86.4, 89.0] | 37.7 [36.1, 39.3] |
| gold in top-1 / top-5 (%) | 37.6 / 60.0 | 43.2 / 69.0 |

E-VQA, gold in top-5 and untruncated (n=1145): gold at rank 1 (n=723) 85.5 [82.8, 88.1]; at ranks 2–5 (n=422) 64.2 [59.0, 69.3]; difference 21.3 [15.1, 27.4] (closed-book wrong: 23.9 [16.9, 30.7]; closed-book right: 9.9 [0.5, 20.5]).
Gold passage alone: rank 1 90.6 [88.2, 93.0]; ranks 2–5 90.8 [87.7, 93.4]; gold alone − all five at ranks 2–5: 26.5 [21.8, 31.5].
Same passage text in both prompts (untruncated; added after the held-out run): gold alone 90.7 [88.8, 92.5], gold alone − all five 13.0 [10.6, 15.6].
E-VQA, gold in top-5: untruncated 77.6 [74.9, 80.1] vs truncated at 160 words 62.9 [58.2, 67.3].
InfoSeek, right page in top-5: an answer string is inside the five chunks in 52.7% of cases; accuracy 39.5 [36.6, 42.2] with it, 7.3 [5.9, 9.0] without.

## Qwen3-VL-8B-Instruct

| | E-VQA | InfoSeek |
|---|---|---|
| never | 26.6 [24.8, 28.4] | 19.3 [17.9, 20.7] |
| always (top-5) | 54.8 [52.2, 57.1] | 31.6 [30.0, 33.3] |
| oracle choice of the two answers | 60.1 [57.6, 62.3] | 35.4 [33.8, 37.0] |
| ideal read/skip gate (read iff gold retrieved) | 53.9 [51.1, 56.2] | 31.6 [29.9, 33.2] |
| ideal gate − always | -1.0 [-2.0, 0.0] | -0.0 [-0.7, 0.6] |
| gold in top-5: reading all five | 74.0 [71.2, 76.5] | 39.3 [37.4, 41.3] |
| gold in top-5: gold passage alone | 89.8 [88.2, 91.2] | 50.2 [48.1, 52.1] |
| gold alone − all five | 15.7 [13.7, 18.2] | 10.9 [8.8, 12.9] |
| ideal passage selector | 63.3 [60.7, 65.6] | 39.1 [37.2, 40.9] |
| gold evidence for every question | 86.0 [84.5, 87.2] | 50.0 [48.5, 51.7] |
| gold in top-1 / top-5 (%) | 37.6 / 60.0 | 43.2 / 69.0 |

E-VQA, gold in top-5 and untruncated (n=1145): gold at rank 1 (n=723) 85.1 [82.4, 87.5]; at ranks 2–5 (n=422) 66.8 [61.5, 71.5]; difference 18.2 [12.8, 24.0] (closed-book wrong: 21.8 [14.8, 28.9]; closed-book right: 9.0 [1.8, 16.4]).
Gold passage alone: rank 1 91.3 [89.0, 93.3]; ranks 2–5 91.0 [88.2, 93.5]; gold alone − all five at ranks 2–5: 24.2 [19.6, 28.9].
Same passage text in both prompts (untruncated; added after the held-out run): gold alone 91.2 [89.4, 92.8], gold alone − all five 12.8 [10.7, 15.1].
E-VQA, gold in top-5: untruncated 78.3 [75.6, 80.7] vs truncated at 160 words 64.3 [59.2, 68.7].
InfoSeek, right page in top-5: an answer string is inside the five chunks in 52.7% of cases; accuracy 58.0 [55.1, 60.7] with it, 18.5 [16.5, 20.8] without.

## Expectations (PREREG v3a)

- D1: held
- D2: held
- D3: held
- D3 (v4c) InternVL3_5-4B-HF: held
- D3 (v4c) Qwen3-VL-8B-Instruct: held
