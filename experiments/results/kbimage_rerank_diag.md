# Where the gain of the top-20 image re-ranking comes from (diagnosis of PREREG v6b; exploratory; E-VQA dev; 80% intervals)

Gold page among the five read, retriever order -> re-ranked (w = 1.0). "Near copies not used": a candidate image with cosine above 0.9 to the question's photograph gets the question's mean similarity instead.

| photographs | as run | near copies not used | gold lead image: median cosine | above 0.9 | above 0.95 |
|---|---|---|---|---|---|
| all | 60.5 -> 70.9 (+10.4 [+8.8, +12.0], n = 1000) | 60.5 -> 70.6 (+10.1 [+8.4, +11.7], n = 1000) | 0.538 | 3.6% | 1.7% |
| inaturalist | 51.8 -> 64.2 (+12.4 [+10.4, +14.6], n = 587) | 51.8 -> 64.2 (+12.4 [+10.4, +14.6], n = 587) | 0.541 | 0.0% | 0.0% |
| landmarks | 72.9 -> 80.4 (+7.5 [+5.1, +10.0], n = 413) | 72.9 -> 79.7 (+6.8 [+4.2, +9.3], n = 413) | 0.537 | 8.0% | 3.7% |

Extrapolation, not a measurement: the main reader is right on 72.4% of development questions when the gold page is among the five read and on 23.0% when it is not (always-retrieve: 52.9%). If those two rates stayed as they are, the re-ranking would be worth +5.1 points of always-retrieve accuracy as run and +5.0 with near copies not used. The questions that gain the gold page are harder than the average question that has it, so the reader pass can come out lower.
