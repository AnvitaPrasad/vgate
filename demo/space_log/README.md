---
title: VGate log audit
emoji: 🔎
colorFrom: blue
colorTo: gray
sdk: gradio
sdk_version: 6.29.0
app_file: app.py
python_version: 3.12
short_description: Audit what a retrieval gate has learned from its log
license: mit
---

The log view of VGate (WSDM 2027 demonstration), on CPU. Upload a CSV with one row per query (`c_ne`, `c_ev`, optional
signals, costs, `segment`, `decision`) or load one of the two sample logs:

- the held-out log of the case study's main model on three benchmarks (outcomes and pre-read signals only; no benchmark
  text or images);
- a published system's log: GPT-4o on the Mix set of Chen et al. (arXiv:2502.18023), scored from the answers they
  released (Apache-2.0) with their token accuracy (correct above 0.5), with their fine-tuned gate's search decisions as
  the logged gate.

The other views of the demo need the case study's per-query outputs or a GPU; see the source repository.
