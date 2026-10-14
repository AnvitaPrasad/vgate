# VGate — 3-minute demo video script

**Recorded 2026-10-11:** `demo/vgate_demo.mp4` (2 min 31 s, 1280×720, captions instead of voice), produced by
`scripts/record_video.py`, which drives the application with Playwright and follows the table below without the Live tab.

Target: ≤ 3:00, screen recording of the demo app plus voice-over (the call encourages a video of up to 3 minutes;
its link goes in the paper and in the submission form). Every number spoken is in `paper/tables/numbers.tex`; re-check
them against the final PDF before recording.

Start the app with `python demo/app.py --public` (three datasets, held-out split; `--public` shows only images
that may be redistributed, with their credit line). Tick **budget-free rule** before recording.

| time | screen | voice-over (≈ 150 words/min) |
|---|---|---|
| 0:00–0:20 | Title card, then the Inspector | "Vision-language models answer knowledge questions about images better with retrieved evidence — sometimes. Retrieval costs searches and tokens, and it can overturn a correct answer. So systems gate it per query. VGate builds such gates from logs, and audits what a gate has actually learned." |
| 0:20–0:50 | Inspector, E-VQA, `EVQA_613` (bridge photo, CC BY, credit shown). Point at Stage 1, Stage 2, then the answers table. | "In which city is this hotel? Closed-book, the model is right. VGate decides twice: before searching, from the model's hidden state, and before reading, from the retriever's scores. Here it searches, then declines to read — and keeps the correct answer that always-retrieve loses. Every pass is costed in tokens and GPU-seconds." |
| 0:50–1:10 | Harm auditor: sort by failure mode; open one case. | "The harm auditor lists every query where retrieval broke a correct answer, and whether the gate would have read the evidence." |
| 1:10–2:05 | **Audit your own log**: click "Use the sample log"; the frontiers appear, then the audit table. Point, in order, at: told the segment, segment classifier, benefit gate, gate per segment, the two unseen-segment rows, the logged-gate rows, oracle gate. Then scroll to the "should you gate?" table and point at the InfoSeek row (AUROC needed, the gate's AUROC, its lead) and the pooled row. Then the download of the fitted gate. | "This is the part we want you to try. Upload a log: two outcomes per query, any signals, and a segment — here, the benchmark. The gate reaches 45.7 percent. But a classifier that only predicts the segment reaches 45.6, and simply being told the segment, 46.0: the gate has learned which kind of traffic it is looking at. A gate fitted per segment adds three tenths of a point, all of it on one benchmark. Fitted on two segments and applied to the third, the gate falls far below. A decision column puts any gate through the same table — here a published prompt-based gate. The second table is a planner: from how often retrieval helps and harms, it says how well a score must separate the two before a gate can lead the better fixed policy by a point, and what this gate reached. Then click 'Use a published system's log': these are GPT-4o's answers and a trained gate's decisions as released by another group, six question sources. By token accuracy the logged gate is two points below the policy that is told the source." |
| 2:05–2:25 | Frontier tab: move the budget slider; point at the star (budget-free operating point). | "The frontier shows accuracy against cost for a benefit-trained and an error-trained gate; trained on its own traffic, the benefit target ranks queries better." |
| 2:25–2:45 | Learn from feedback: press "Run the stream"; point at the black curve meeting the blue one. | "Paired outcomes are expensive to log. This simulation learns the gate from feedback on the shown answer alone; it ends within 0.8 points of the gate fitted on paired data." |
| 2:45–3:00 | Live tab (if a GPU host is up; otherwise the end card): one uploaded image. End card: repository URL, paper title. | "And it runs live. VGate: auditing what retrieval gates learn." |

What the video must not claim: that the replay views run a model (they replay logged runs), per-query gains beyond the
segment other than the per-segment gate's 0.3 points (46.3 against 46.0; exploratory), that the planner's estimate is a bound, token savings on E-VQA / InfoSeek, or that the feedback learner is more than a simulation. On the published log say only what the table shows (binary correctness at the authors' 0.5 threshold, one fold-free 'told the segment' row; the paper's figure of 2.4 points uses their graded accuracy and a cross-fitted policy), and that their judge-model metric was not computed. Do not show benchmark images other than the credited Google-Landmarks ones
(`docs/strengthening/R4_release.md`).

Recording checklist: 1280×720, browser zoom 110%, cursor highlighting on, no personal data on screen. The three
example queries are in the held-out split and were chosen with the budget-free rule; confirm each one still shows
the stated decision after any re-run of `make confirm extend`.
