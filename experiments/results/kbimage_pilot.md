# Entity identification from candidate page images (PREREG v6 pilot; E-VQA dev, n=1000; 80% intervals)

Candidate titles 8053, with a lead image 3926.

| | dinov2 | clip |
|---|---|---|
| gold page among the candidates (%) | 60.5 | 60.5 |
| … of which the gold page has a lead image (%) | 98.3 | 98.3 |
| gold page first: retriever order (%) | 64.5 | 64.5 |
| gold page first: image similarity (%) | 75.2 | 66.0 |
| gold page first: z(retriever) + z(image) (%) | 74.5 | 70.9 |
| fused pick among the five is the gold passage (%) | 70.0 | 66.0 |
| accuracy, always-retrieve | 52.9 | 52.9 |
| accuracy, fused pick read alone | 51.7 | 50.9 |
| pick − always | [-1.2, -2.9, 0.43] | [-2.0, -3.78, -0.27] |
| accuracy, best page of the candidates when its passage is among the five | 51.7 | 50.9 |
| … − always | [-1.2, -2.9, 0.43] | [-2.0, -3.78, -0.27] |

By image source (dinov2): inaturalist (n=587): near-duplicate of a candidate's lead image 0.0%, gold pick 71.5% (retriever 62.2%), pick − always +0.00; landmarks (n=413): near-duplicate of a candidate's lead image 2.7%, gold pick 68.5% (retriever 58.7%), pick − always -2.91.

By image source (clip): inaturalist (n=587): near-duplicate of a candidate's lead image 0.0%, gold pick 66.0% (retriever 62.2%), pick − always -2.04; landmarks (n=413): near-duplicate of a candidate's lead image 2.4%, gold pick 66.1% (retriever 58.7%), pick − always -1.94.

**Go / no-go (PREREG v6): NO-GO** (encoder dinov2; needs ≥ 80% gold picks and ≥ +3.0 over always-retrieve with the interval above 0).
