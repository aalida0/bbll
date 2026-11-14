# Two-era OOF scoring closure — 14 November 2026

**Status: PASS for full-event OOF materialization and source closure.**

The analyst ran `python -m bdt_training.validate_scores` for both eras and
both topologies on the UChicago AF selected ROOT campaign. The reported
per-sample comparisons passed for every sample (16 signal + 10 prompt
background samples in each era). The validator checks exact event identity,
source entries, both category labels, (m_T(HH)), raw signed physics
weights, OOF fold assignments, score finiteness and range, and direct
held-out model predictions for a subset of events per sample.

| Era | Topology | Events | Result |
|---|---|---:|---|
| Run 2 | Resolved | 8,891,378 | PASS |
| Run 2 | Boosted | 6,294,727 | PASS |
| Run 3 | Resolved | 8,899,170 | PASS |
| Run 3 | Boosted | 6,225,647 | PASS |
| **Total** | **104 sample/topology outputs** | **30,310,922** | **PASS** |

The sum of topology counts is exactly the frozen preselection campaign
count in `bbll/analysis.yaml`; there is no gap or double counting
between resolved and boosted events.

No changes to the trained classifier or the selected ROOT files were made
for this validation. The 308/312 Run-3 luminosity correction remains
deferred to yield/template construction.

**Not yet established:** BDT threshold choices, preservation of smooth
background shapes, signed-background effective event counts in the
high-score region, treatment of fake-lepton contributions or detector/theory
nuisance parameters, and expected-limit improvement compared with a
cut-based baseline. These require separate SR and inference validation.
