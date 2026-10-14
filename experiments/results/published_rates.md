# The planner on published counts for a larger reader (GPT-4o on InfoSeek; arXiv:2405.18740v1, Tables 1, 3, 4)

1650 questions; evidence: a screenshot of reverse-image-search results. Helped 221 (13.39%), harmed 94 (5.7%) (sums of Table 3's eleven categories; their difference is Table 1's 39.21 -> 46.91). An oracle gate would lead always-retrieve by 5.7 points; a one-point lead needs a help-vs-harm AUROC of at least 0.588 on a concave curve and 0.772 at the binormal estimate.

| agent (Table 4) | reads % of helped | reads % of harmed | AUROC of the one decision | lead over always-retrieve from the identity | accuracy from the identity | printed |
|---|---|---|---|---|---|---|
| Decide + Consistency Check | 42.08 | 28.72 | 0.567 | -3.70 | 43.21 | 43.21 |
| Decide only | 82.81 | 67.02 | 0.579 | -0.42 | 46.49 | 46.48 |
| Consistency Check only | 46.15 | 39.36 | 0.534 | -3.76 | 43.15 | 43.15 |

The identity reproduces the three printed accuracies within 0.01 points; 3 of 3 decisions are below the floor.
Not comparable with our setting in every respect: the evidence is an image of a results page from a live search, not text passages; a GPT-4-Turbo judge; one run.
