# Passage-level selection (PREREG v3): Qwen3-VL-4B-Instruct, split dev, grouped 5-fold out of fold; 80% intervals

## evqa (n=1000; verification pass available; closed-book re-run agreement 100.0%)

| policy | accuracy | − always | − never | tokens / query |
|---|---|---|---|---|
| always | 52.9 [50.4, 55.4] | — | +26.2 [+24.0, +28.5] | 1075 |
| oracle6 | 72.4 [70.3, 74.4] | +19.5 [+17.8, +21.1] | +45.7 [+43.7, +47.7] | 2588 |
| pick_benefit | 55.7 [53.3, 58.0] | +2.8 [+1.2, +4.4] | +29.0 [+26.7, +31.5] | 4933 |
| pick_correct | 55.5 [53.0, 57.9] | +2.6 [+1.1, +4.1] | +28.8 [+26.7, +31.1] | 4933 |
| pick_benefit__no_pv | 48.0 [45.6, 50.2] | -4.9 [-6.8, -3.1] | +21.3 [+18.9, +23.7] | 2588 |
| pick_correct__no_pv | 49.2 [46.8, 51.5] | -3.7 [-5.5, -2.0] | +22.5 [+20.4, +24.6] | 2588 |
| pick_benefit__no_ans | 55.4 [53.0, 57.8] | +2.5 [+0.9, +4.1] | +28.7 [+26.4, +31.1] | 4933 |
| pick_correct__no_ans | 55.4 [53.0, 57.8] | +2.5 [+1.0, +4.0] | +28.7 [+26.6, +30.9] | 4933 |
| never | 26.7 [24.7, 28.5] | -26.2 [-28.5, -24.0] | — | 249 |
| top1 | 48.4 [45.9, 50.9] | -4.5 [-6.1, -3.0] | +21.7 [+19.4, +24.0] | 695 |
| confident | 45.3 [43.0, 47.5] | -7.6 [-9.4, -5.8] | +18.6 [+16.5, +20.7] | 2588 |
| verify | 54.1 [51.7, 56.5] | +1.2 [-0.6, +3.0] | +27.4 [+25.3, +29.6] | 2987 |
| vote | 44.9 [42.6, 47.1] | -8.0 [-10.0, -6.1] | +18.2 [+16.0, +20.3] | 2588 |
| ev_keep | 44.9 [42.6, 47.2] | -8.0 [-9.5, -6.5] | +18.2 [+16.3, +20.2] | 1340 |

VGate-pick chooses: closed-book 1%, passage 1 59%, passage 2 17%, passage 3 10%, passage 4 9%, passage 5 4%. Accuracy reading passage k alone: k=1 48.4, k=2 34.4, k=3 28.6, k=4 25.6, k=5 25.0. Benefit target − correctness target: +0.2 [-1.0, +1.4].

## infoseek (n=1000; verification pass available; closed-book re-run agreement 100.0%)

| policy | accuracy | − always | − never | tokens / query |
|---|---|---|---|---|
| always | 25.7 [23.7, 27.7] | — | +8.4 [+6.4, +10.3] | 1019 |
| oracle6 | 39.5 [37.3, 41.7] | +13.8 [+12.4, +15.3] | +22.2 [+20.4, +24.1] | 2413 |
| pick_benefit | 27.3 [25.3, 29.3] | +1.6 [+0.4, +2.8] | +10.0 [+8.0, +12.0] | 4567 |
| pick_correct | 27.1 [25.0, 29.1] | +1.4 [+0.0, +2.9] | +9.8 [+8.1, +11.6] | 4567 |
| pick_benefit__no_pv | 26.8 [24.8, 28.8] | +1.1 [-0.1, +2.3] | +9.5 [+7.4, +11.5] | 2413 |
| pick_correct__no_pv | 26.5 [24.5, 28.5] | +0.8 [-0.7, +2.3] | +9.2 [+7.5, +11.0] | 2413 |
| pick_benefit__no_ans | 26.7 [24.7, 28.6] | +1.0 [-0.2, +2.2] | +9.4 [+7.6, +11.3] | 4567 |
| pick_correct__no_ans | 26.5 [24.4, 28.5] | +0.8 [-0.6, +2.3] | +9.2 [+7.5, +10.9] | 4567 |
| never | 17.3 [15.6, 19.1] | -8.4 [-10.3, -6.4] | — | 268 |
| top1 | 24.1 [22.0, 26.0] | -1.6 [-3.1, -0.3] | +6.8 [+4.7, +8.8] | 704 |
| confident | 24.8 [22.7, 26.7] | -0.9 [-2.2, +0.3] | +7.5 [+5.4, +9.4] | 2413 |
| verify | 24.6 [22.6, 26.6] | -1.1 [-2.6, +0.3] | +7.3 [+5.8, +8.8] | 2666 |
| vote | 25.8 [23.7, 27.9] | +0.1 [-1.4, +1.6] | +8.5 [+6.7, +10.4] | 2413 |
| ev_keep | 24.1 [22.2, 26.0] | -1.6 [-3.2, -0.1] | +6.8 [+5.6, +8.0] | 1305 |

VGate-pick chooses: closed-book 12%, passage 1 52%, passage 2 18%, passage 3 8%, passage 4 6%, passage 5 4%. Accuracy reading passage k alone: k=1 24.1, k=2 18.8, k=3 18.2, k=4 14.6, k=5 13.9. Benefit target − correctness target: +0.2 [-0.9, +1.3].

## okvqa (n=1000; verification pass available; closed-book re-run agreement 99.9%)

| policy | accuracy | − always | − never | tokens / query |
|---|---|---|---|---|
| always | 38.6 [36.7, 40.6] | — | -20.8 [-22.8, -18.7] | 985 |
| oracle6 | 69.4 [67.5, 71.3] | +30.8 [+28.9, +32.7] | +10.0 [+8.9, +11.3] | 2423 |
| pick_benefit | 60.2 [58.1, 62.1] | +21.6 [+19.6, +23.5] | +0.8 [-0.1, +1.7] | 4583 |
| pick_correct | 59.6 [57.6, 61.5] | +21.0 [+19.0, +22.9] | +0.2 [-0.4, +0.8] | 4583 |
| pick_benefit__no_pv | 60.0 [58.0, 61.9] | +21.4 [+19.4, +23.3] | +0.6 [-0.2, +1.4] | 2423 |
| pick_correct__no_pv | 59.6 [57.6, 61.5] | +21.0 [+18.9, +22.9] | +0.2 [-0.4, +0.8] | 2423 |
| pick_benefit__no_ans | 58.7 [56.7, 60.7] | +20.1 [+18.1, +22.0] | -0.7 [-1.4, +0.1] | 4583 |
| pick_correct__no_ans | 59.4 [57.4, 61.4] | +20.8 [+18.7, +22.8] | +0.0 [+0.0, +0.0] | 4583 |
| never | 59.4 [57.4, 61.4] | +20.8 [+18.7, +22.8] | — | 277 |
| top1 | 45.8 [43.7, 47.8] | +7.2 [+5.5, +8.8] | -13.6 [-15.6, -11.6] | 705 |
| confident | 46.8 [44.8, 48.8] | +8.2 [+6.7, +9.7] | -12.6 [-14.5, -10.7] | 2423 |
| verify | 51.3 [49.2, 53.2] | +12.7 [+10.9, +14.4] | -8.1 [-9.6, -6.6] | 2676 |
| vote | 52.2 [50.1, 54.2] | +13.6 [+11.9, +15.3] | -7.2 [-8.9, -5.5] | 2423 |
| ev_keep | 57.8 [55.8, 59.8] | +19.2 [+17.2, +21.1] | -1.6 [-2.3, -0.9] | 1279 |

VGate-pick chooses: closed-book 50%, passage 1 12%, passage 2 14%, passage 3 11%, passage 4 8%, passage 5 5%. Accuracy reading passage k alone: k=1 45.8, k=2 45.8, k=3 45.9, k=4 46.9, k=5 46.1. Benefit target − correctness target: +0.6 [-0.1, +1.3].

**Go / no-go (PREREG v3): NO-GO** — VGate-pick − always ≥ +3.0 with the interval above 0 on E-VQA and InfoSeek, and not more than 1.0 below never-retrieve on OK-VQA.
