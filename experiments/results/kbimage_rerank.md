# Re-ranking the top-20 with candidate page images (PREREG v6b pilot; E-VQA dev, n = 1000; 80% intervals)

Candidates with an image on disk: 97.3%. Gold page among the top-20: 78.7%. Weights chosen on the training folds: [1.0, 1.0, 1.0, 1.0, 1.0].

| % of questions | retriever order | re-ranked | difference |
|---|---|---|---|
| gold page among the five read | 60.5 | 70.9 | +10.4 [+8.8, +12.0] |
| gold passage among the five read | 57.7 | 68.0 | +10.3 [+8.8, +11.8] |
| gold page ranked first | 39.0 | 50.2 | +11.2 [+8.8, +13.4] |
| gold passage ranked first | 34.9 | 44.9 | +10.0 [+7.9, +12.0] |

**Go / no-go (PREREG v6b): GO** (needs the gold page among the five read to rise by at least 6 points, interval above +3).
