---
title: VGate log audit
emoji: 🔎
colorFrom: blue
colorTo: gray
sdk: static
app_file: index.html
short_description: Audit what a retrieval gate has learned from its log
license: mit
---

The log view of VGate (WSDM 2027 demonstration). It runs entirely in your browser (Python on Pyodide): nothing you
upload leaves your machine. The first load downloads the Python runtime and takes about a minute.

Upload a CSV with one row per query (`c_ne`, `c_ev`, optional signals, costs, `segment`, `decision`) or load one of the two
sample logs:

- the held-out log of the case study's main model on three benchmarks (outcomes and pre-read signals only; no benchmark
  text or images). Fitting its gates in the browser takes about a minute, and the last digit can differ from the paper (library versions);
- a published system's log: GPT-4o on the Mix set of Chen et al. (arXiv:2502.18023), scored from the answers they
  released (Apache-2.0) with their token accuracy (correct above 0.5), with their fine-tuned gate's search decisions as
  the logged gate. This one is instant.

The other views of the demo need the case study's per-query outputs or a GPU; see the source repository.
