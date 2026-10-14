# Pilot, exploratory analyses E1–E3 (dev split, 80% intervals; PREREG v3 addendum)

## evqa (n=1000)

**E1** top-5 answer equals the answer from passage k alone (%): k=1: 57.6, k=2: 26.2, k=3: 19.8, k=4: 18.7, k=5: 18.4; equals the closed-book answer: 16.3.
With the gold passage at ranks 2–5 (n=228): the top-5 answer equals passage 1's answer in 36.8% and the gold passage's answer in 47.4%.

**E2** reading-budget choice: never 26.7 [24.7, 28.5] (249.3 tok); top1_alone 48.4 [45.9, 50.9] (695.2 tok); always 52.9 [50.4, 55.4] (1074.6 tok); allocator 53.4 [50.8, 55.9] (1173.7 tok); oracle3 62.2 [59.8, 64.5] (— tok). Allocator − always 0.5 [-0.2, 1.2]; top-1 alone − always -4.5 [-6.1, -3.0]; shares {0: 0.023, 1: 0.188, 5: 0.789}.

**E3** early exit: cap 2: 49.1 [46.5, 51.6], − always -3.8 [-5.4, -2.2], 728.1 tok, 1.07 passages; cap 3: 49.3 [46.7, 51.8], − always -3.6 [-5.2, -2.0], 747.3 tok, 1.11 passages; cap 5: 49.5 [46.9, 52.0], − always -3.4 [-5.0, -1.8], 774.8 tok, 1.17 passages (always: 1074.6 tok).

## infoseek (n=1000)

**E1** top-5 answer equals the answer from passage k alone (%): k=1: 49.0, k=2: 35.6, k=3: 31.6, k=4: 31.5, k=5: 28.3; equals the closed-book answer: 23.5.
With the gold passage at ranks 2–5 (n=255): the top-5 answer equals passage 1's answer in 39.2% and the gold passage's answer in 47.5%.

**E2** reading-budget choice: never 17.3 [15.6, 19.1] (268.1 tok); top1_alone 24.1 [22.0, 26.0] (704.3 tok); always 25.7 [23.7, 27.7] (1019.3 tok); allocator 26.9 [24.8, 29.0] (1018.5 tok); oracle3 34.1 [31.9, 36.4] (— tok). Allocator − always 1.2 [0.3, 2.1]; top-1 alone − always -1.6 [-3.1, -0.3]; shares {0: 0.037, 1: 0.398, 5: 0.565}.

**E3** early exit: cap 2: 25.2 [23.1, 27.1], − always -0.5 [-1.8, 0.8], 794.9 tok, 1.21 passages; cap 3: 25.7 [23.6, 27.6], − always 0.0 [-1.3, 1.3], 851.8 tok, 1.34 passages; cap 5: 26.3 [24.2, 28.2], − always 0.6 [-0.7, 1.8], 937.4 tok, 1.54 passages (always: 1019.3 tok).

## okvqa (n=1000)

**E1** top-5 answer equals the answer from passage k alone (%): k=1: 56.1, k=2: 51.1, k=3: 50.0, k=4: 49.1, k=5: 51.4; equals the closed-book answer: 41.5.

**E2** reading-budget choice: never 59.4 [57.4, 61.4] (277.2 tok); top1_alone 45.8 [43.7, 47.8] (704.5 tok); always 38.6 [36.7, 40.6] (984.7 tok); allocator 58.8 [56.8, 60.7] (298.9 tok); oracle3 66.4 [64.5, 68.3] (— tok). Allocator − always 20.2 [18.1, 22.2]; top-1 alone − always 7.2 [5.5, 8.8]; shares {0: 0.966, 1: 0.022, 5: 0.012}.

**E3** early exit: cap 2: 59.5 [57.5, 61.5], − always 20.9 [18.8, 22.9], 1052.4 tok, 1.81 passages; cap 3: 59.6 [57.6, 61.5], − always 21.0 [19.0, 22.9], 1353.7 tok, 2.51 passages; cap 5: 59.8 [57.8, 61.7], − always 21.2 [19.1, 23.1], 1892.2 tok, 3.76 passages (always: 984.7 tok).

