# CAPA-SR Paper Plan

## Title
CAPA-SR: Content-Aware Progressive Attention for Lightweight Image Super-Resolution

## Core claim
By combining image-adaptive routing, progressive focused sparse interaction, and low-to-high multi-level reconstruction, the proposed model improves complex-structure super-resolution while preserving lightweight efficiency.

## Main evidence
- x4: Urban100 `26.87 -> 27.24` and Manga109 `31.31 -> 31.55` over `CATANet`.
- x4 complexity changes from `46.8G / 86.0 ms` to `46.1G / 85.1 ms`.
- Progressive ablation verifies cumulative gains from `DPR`, `PFSA`, and `LMR`.

## Figure plan
1. Intro figure: Urban100 x4 performance-efficiency snapshot highlighting the trade-off against `CATANet`, `PFT-light`, and `SwinIR-light`.
2. Method figure: overview of the outer backbone and the inner `CAPA Block`.
3. Experiment figure: cumulative gain curve for the core ablation from Model A to Model E.

## Section plan
1. Abstract: problem, method, strongest quantitative result.
2. Introduction: motivation, gap between content-aware grouping and focused attention, method overview, contributions.
3. Related Work: lightweight SR, content-aware token grouping, focused sparse attention.
4. Method: overall architecture, `DPR`, `PFSA`, `LMR`, selective `IRCA`.
5. Experiments: setup, x4 main results, complexity, ablations.
6. Conclusion: summary, limitations, future work.
7. Appendix: full x2/x3 tables, design-choice ablations, training milestones.
