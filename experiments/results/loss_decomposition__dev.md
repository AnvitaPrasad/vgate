# Loss decomposition (dev split; PREREG v3a; accuracy points, 95% image-cluster bootstrap)

## Qwen3-VL-4B-Instruct

| | E-VQA | InfoSeek |
|---|---|---|
| never | 26.7 [23.7, 29.5] | 17.3 [14.7, 20.1] |
| always (top-5) | 52.9 [49.2, 56.6] | 25.7 [22.6, 28.8] |
| oracle choice of the two answers | 58.4 [54.8, 61.8] | 30.9 [27.8, 34.1] |
| ideal read/skip gate (read iff gold retrieved) | 52.5 [48.8, 56.0] | 26.8 [23.8, 29.9] |
| ideal gate − always | -0.4 [-2.0, 1.3] | 1.1 [-0.1, 2.3] |
| gold in top-5: reading all five | 74.0 [70.0, 77.8] | 33.9 [30.3, 37.4] |
| gold in top-5: gold passage alone | 91.2 [88.7, 93.5] | 46.7 [43.0, 50.4] |
| gold alone − all five | 17.2 [13.9, 20.6] | 12.8 [8.8, 16.4] |
| ideal passage selector | 62.4 [58.7, 65.8] | 35.5 [32.0, 39.0] |
| gold evidence for every question | 86.0 [83.7, 88.2] | 48.2 [45.1, 51.3] |
| gold in top-1 / top-5 (%) | 34.9 / 57.7 | 42.6 / 68.1 |

E-VQA, gold in top-5 and untruncated (n=408): gold at rank 1 (n=245) 88.6 [84.2, 92.4]; at ranks 2–5 (n=163) 63.2 [55.5, 70.7]; difference 25.4 [17.1, 33.7] (closed-book wrong: 31.0 [20.9, 41.8]; closed-book right: 11.4 [-0.9, 24.8]).
Gold passage alone: rank 1 91.4 [87.7, 94.7]; ranks 2–5 92.0 [87.6, 95.9]; gold alone − all five at ranks 2–5: 28.8 [21.9, 36.0].
Same passage text in both prompts (untruncated; added after the held-out run): gold alone 91.7 [88.7, 94.2], gold alone − all five 13.2 [9.7, 16.8].
E-VQA, gold in top-5: untruncated 78.4 [74.1, 82.7] vs truncated at 160 words 63.3 [55.9, 70.4].
InfoSeek, right page in top-5: an answer string is inside the five chunks in 51.4% of cases; accuracy 55.4 [49.7, 61.1] with it, 11.2 [7.7, 14.9] without.

## Qwen2.5-VL-3B-Instruct

| | E-VQA | InfoSeek |
|---|---|---|
| never | 24.6 [21.6, 27.5] | 15.7 [13.3, 18.1] |
| always (top-5) | 48.5 [44.9, 52.1] | 25.6 [22.6, 28.7] |
| oracle choice of the two answers | 55.2 [51.7, 58.6] | 30.0 [26.7, 33.0] |
| ideal read/skip gate (read iff gold retrieved) | 48.3 [44.7, 51.8] | 26.1 [23.2, 29.2] |
| ideal gate − always | -0.2 [-2.1, 1.8] | 0.5 [-0.5, 1.5] |
| gold in top-5: reading all five | 68.3 [64.2, 72.6] | 33.9 [30.3, 37.6] |
| gold in top-5: gold passage alone | 86.7 [83.8, 89.5] | 46.5 [42.7, 50.5] |
| gold alone − all five | 18.4 [14.9, 22.0] | 12.6 [8.5, 16.6] |
| ideal passage selector | 58.9 [55.3, 62.4] | 34.7 [31.2, 38.3] |
| gold evidence for every question | 81.3 [78.7, 83.8] | 48.4 [44.9, 51.6] |
| gold in top-1 / top-5 (%) | 34.9 / 57.7 | 42.6 / 68.1 |

E-VQA, gold in top-5 and untruncated (n=408): gold at rank 1 (n=245) 78.0 [72.7, 83.3]; at ranks 2–5 (n=163) 61.4 [53.8, 69.4]; difference 16.6 [7.0, 25.4] (closed-book wrong: 22.3 [10.9, 32.8]; closed-book right: 1.3 [-12.3, 15.6]).
Gold passage alone: rank 1 87.8 [83.3, 91.9]; ranks 2–5 87.1 [81.6, 91.9]; gold alone − all five at ranks 2–5: 25.8 [18.5, 32.7].
Same passage text in both prompts (untruncated; added after the held-out run): gold alone 87.5 [84.1, 90.8], gold alone − all five 16.2 [12.1, 20.3].
E-VQA, gold in top-5: untruncated 71.3 [66.7, 76.0] vs truncated at 160 words 61.0 [53.3, 68.4].
InfoSeek, right page in top-5: an answer string is inside the five chunks in 51.4% of cases; accuracy 55.1 [49.5, 60.6] with it, 11.5 [8.0, 15.2] without.

## InternVL3_5-4B-HF

| | E-VQA | InfoSeek |
|---|---|---|
| never | 22.4 [19.7, 25.0] | 9.8 [7.8, 12.0] |
| always (top-5) | 52.7 [48.9, 56.5] | 18.5 [15.8, 21.0] |
| oracle choice of the two answers | 57.9 [54.2, 61.5] | 22.3 [19.4, 25.2] |
| ideal read/skip gate (read iff gold retrieved) | 51.0 [47.4, 54.8] | 18.6 [15.9, 21.2] |
| ideal gate − always | -1.7 [-3.4, 0.1] | 0.1 [-0.9, 1.1] |
| gold in top-5: reading all five | 73.3 [69.5, 77.3] | 24.1 [20.6, 27.4] |
| gold in top-5: gold passage alone | 91.5 [89.1, 93.8] | 38.6 [34.8, 42.2] |
| gold alone − all five | 18.2 [14.5, 21.8] | 14.5 [10.5, 18.8] |
| ideal passage selector | 61.5 [57.8, 65.1] | 28.5 [25.2, 31.6] |
| gold evidence for every question | 86.6 [84.4, 88.8] | 39.4 [36.3, 42.2] |
| gold in top-1 / top-5 (%) | 34.9 / 57.7 | 42.6 / 68.1 |

E-VQA, gold in top-5 and untruncated (n=408): gold at rank 1 (n=245) 86.5 [82.1, 90.6]; at ranks 2–5 (n=163) 66.9 [59.9, 73.8]; difference 19.7 [12.0, 27.8] (closed-book wrong: 17.0 [7.6, 26.2]; closed-book right: 27.6 [13.8, 41.2]).
Gold passage alone: rank 1 93.9 [90.5, 96.9]; ranks 2–5 91.4 [86.6, 95.7]; gold alone − all five at ranks 2–5: 24.5 [17.0, 31.8].
Same passage text in both prompts (untruncated; added after the held-out run): gold alone 92.9 [90.0, 95.3], gold alone − all five 14.2 [10.3, 18.1].
E-VQA, gold in top-5: untruncated 78.7 [74.6, 82.6] vs truncated at 160 words 60.4 [53.0, 67.8].
InfoSeek, right page in top-5: an answer string is inside the five chunks in 51.4% of cases; accuracy 41.7 [36.0, 47.2] with it, 5.4 [3.2, 8.0] without.

## Qwen3-VL-8B-Instruct

| | E-VQA | InfoSeek |
|---|---|---|
| never | 29.0 [25.9, 31.9] | 17.4 [14.9, 20.0] |
| always (top-5) | 56.3 [52.6, 59.9] | 30.2 [27.0, 33.4] |
| oracle choice of the two answers | 60.7 [57.0, 64.2] | 34.1 [30.9, 37.4] |
| ideal read/skip gate (read iff gold retrieved) | 55.5 [51.8, 59.0] | 30.0 [26.8, 33.2] |
| ideal gate − always | -0.8 [-2.4, 0.8] | -0.2 [-1.1, 0.8] |
| gold in top-5: reading all five | 77.8 [74.3, 81.4] | 38.9 [35.5, 42.4] |
| gold in top-5: gold passage alone | 90.3 [87.8, 92.6] | 50.7 [47.2, 54.1] |
| gold alone − all five | 12.5 [9.4, 15.6] | 11.8 [8.0, 15.6] |
| ideal passage selector | 62.7 [59.0, 66.0] | 38.0 [34.4, 41.7] |
| gold evidence for every question | 84.9 [82.6, 87.2] | 52.2 [49.2, 55.0] |
| gold in top-1 / top-5 (%) | 34.9 / 57.7 | 42.6 / 68.1 |

E-VQA, gold in top-5 and untruncated (n=408): gold at rank 1 (n=245) 89.8 [85.7, 93.5]; at ranks 2–5 (n=163) 70.5 [63.3, 77.7]; difference 19.2 [11.1, 27.1] (closed-book wrong: 22.4 [11.8, 32.8]; closed-book right: 12.4 [2.6, 23.0]).
Gold passage alone: rank 1 91.8 [88.3, 95.0]; ranks 2–5 88.3 [83.1, 93.1]; gold alone − all five at ranks 2–5: 17.8 [10.8, 24.7].
Same passage text in both prompts (untruncated; added after the held-out run): gold alone 90.4 [87.6, 93.3], gold alone − all five 8.3 [5.0, 11.6].
E-VQA, gold in top-5: untruncated 82.1 [78.2, 86.0] vs truncated at 160 words 67.5 [60.3, 74.6].
InfoSeek, right page in top-5: an answer string is inside the five chunks in 51.4% of cases; accuracy 58.6 [53.2, 63.9] with it, 18.1 [14.0, 22.1] without.

## Expectations (PREREG v3a)

- D1: held
- D2: held
- D3: held
- D3 (v4c) InternVL3_5-4B-HF: held
- D3 (v4c) Qwen3-VL-8B-Instruct: held
