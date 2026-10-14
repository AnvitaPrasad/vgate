# CVQA: why the gate does not lead the better fixed policy (exploratory; see PREREG v7 for the registered outcome)

## 1. What the gate reads (held-out, gate fitted on development data)
- read: 19.8% of questions; of these retrieval helps 16.6%, harms 14.7%, changes nothing 68.7%
- skipped: 80.2% of questions; of these retrieval helps 6.3%, harms 13.8%, changes nothing 79.9%
- net effect of its reads: +0.38 points (it must find help without harm: of all help cases it reads 39.6%, of all harm cases 20.9%)

## 2. How well the score ranks queries (held-out)
- AUROC of the predicted benefit: help vs harm 0.626; help vs everything else 0.554; harm vs everything else (score negated) 0.613
- best possible threshold on this score, chosen with the held-out labels (optimistic): read the top 9.0% -> +0.73 points over never-retrieve (the registered rule read 19.8% -> +0.38)
  - top 2% by predicted benefit: helps 29.3%, harms 13.2% (base rates 8.3% / 14.0%)
  - top 5% by predicted benefit: helps 23.2%, harms 13.2% (base rates 8.3% / 14.0%)
  - top 10% by predicted benefit: helps 20.7%, harms 14.2% (base rates 8.3% / 14.0%)
  - top 20% by predicted benefit: helps 16.6%, harms 14.6% (base rates 8.3% / 14.0%)

## 3. Would more labelled traffic fix it? (learning curve; held-out gain over never-retrieve of VGate-full)
- fitted on 500 development questions: -0.33 (range -0.75 to -0.02 over 5 subsamples)
- fitted on 1000 development questions: -0.04 (range -0.44 to +0.32 over 5 subsamples)
- fitted on 2001 development questions: +0.38
- fitted on about 6,700 held-out questions, scored on the rest (5-fold, grouped): +0.73 [+0.21, +1.24], reading 15.3%

## 4. Which signals carry it (held-out gain over never-retrieve, gates fitted on development data)
- VGate-post: +0.60 (reads 13.6%)
- VGate-pre: +0.27 (reads 19.8%)
- VGate-full: +0.38 (reads 19.8%)
- error gate, free + retriever-score features: -0.57 (reads 22.3%)
- error gate, VGate-full's features: -0.61 (reads 31.2%)

## 5. Candidate rules, DEVELOPMENT data only (out of fold where fitted; 80% intervals)
- reference: never-retrieve 66.27, always-retrieve 60.72; the two answers name the same option on 67.5% of questions; best of the two answers 74.81
- rules that read the evidence for every question and then choose an answer (cost: both passes), gain over never-retrieve:
  - evidence answer unless it names no option: -2.10 [-3.60, -0.50] (uses the evidence answer on 92.5%)
  - the answer with the higher first-token probability: +2.50 [+1.42, +3.69] (uses the evidence answer on 31.9%)
  - evidence answer only if at least 0.2 more probable: +1.20 [+0.65, +1.76] (uses the evidence answer on 8.1%)
  - fitted selector on both passes' first-token probabilities, agreement and the gate's scalar signals (out of fold): +1.70 [+0.87, +2.60] (uses the evidence answer on 31.9%)
- the registered gate decides before reading; these rules decide after it and pay for both passes on every question.
