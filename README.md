# Structural Non-Isolability in Knowledge-Based Robot Fault Diagnosis

Code and derived data for a study of what a knowledge-graph fault-diagnosis
pipeline **cannot** decide, and what changing its representation would fix.

The paper, *Structural Non-Isolability in Knowledge-Based Robot Fault Diagnosis:
Tie-Break Intervals and Grouped Validation* (SAFEPROCESS 2027 manuscript), is
`safeprocess2027_final.tex` / `safeprocess2027_final.pdf`. Everything else in
this repository is the experimental record behind it.

Built on [voraus-AD](https://github.com/vorausrobotik/voraus-ad-dataset), a
six-joint collaborative robot performing pick-and-place, with 12 injected
anomaly types over 755 anomalous cycles.

---

## What this repository shows

Coupling an anomaly detector to an FMEA knowledge base produces a ranked list of
candidate causes. On this benchmark that list has three problems, and they are
the subject of the code here.

**1. Most rank-1 answers are structurally tied.** In 514 of 755 anomalies
(68.1%) two or more failure modes carry identical affected joints and identical
manifesting signal families, so the retrieval score assigns them the same value
for every cycle. Which one is reported is decided by tie-breaking, not evidence.
On the five leave-setting-out splits used for every comparison below,
re-ordering the failure modes in the knowledge base moves the five-split mean
top-1 across `[0.119, 0.200]` — `0.082` wide, more than twice the split-to-split
95% interval (`±0.018`), and containing the trivial baseline below (`0.187`).
Cluster assignment, by contrast, is invariant at `0.342` under all 203 orderings
on every split. (Swept over all 755 cycles instead of the held-out halves, the
same interval is `[0.117, 0.199]` and the invariant cluster accuracy `0.340`.)

**2. A constant beats the ranking.** A *prevalence prior* — rank the failure
modes by how often they occurred in the fit half, emit that same ranking for
every cycle, read no evidence — scores higher than the deployed retrieval on
top-1, top-3, top-5, MRR and cluster accuracy. It loses only on balanced recall
(mean per-category recall), which is the one metric a constant predictor cannot
game, and which is therefore the metric the rest of the analysis uses.

**3. The validation protocol decides a deployment question.** The 72
`setting` values in voraus-AD are repeated instantiations of single physical
faults, 4–20 cycles each. Under a random split the pipeline certifies a
representation repair for the distal-collision cluster at `0.830` and adopts it
on all five splits; under leave-setting-out validation the same pipeline rejects
it. Nothing in the ungrouped run looks wrong, and permuted-label controls do not
catch it.

Against that, the part that holds: identifying which modes are mutually
non-isolable, searching a grammar of discarded physical dimensions for one that
separates them, and issuing the result as a **repair certificate** carrying
held-out accuracy, three controls and a cost class. On the load-torque cluster
this reaches `0.840` held-out three-way accuracy against `0.562` for the deployed
representation, and lifts balanced recall from `0.095` (prior) through `0.119`
(retrieval) to `0.213` — the only stage that clears both trivial references.

---

## Install

Python 3.12.

```bash
python -m venv .venv
```

```bash
.venv/bin/pip install -r requirements.txt
```

On Windows the second command is `.venv\Scripts\pip install -r requirements.txt`.

---

## Reproduce without downloading the dataset

Most results are reproducible from the derived evidence records shipped in
`data/`, with no need for the 1 GB parquet. Scripts import each other flatly, so
run them from inside `pipeline/`.

```bash
cd pipeline && python baselines.py
```

That reproduces the prevalence prior, the balanced-recall comparison and the
tie-break interval — both the leave-setting-out sweep the paper reports and the
full-set sweep — and writes `results/baselines.json`.

```bash
cd pipeline && python distinguishability.py
```

Structural twin clusters, route assignment, cluster accuracy and the
margin-based abstention. Add `--probe` for the separability test that shows the
clusters are a property of the representation rather than of the robot.

```bash
cd pipeline && python repair.py --control --labels --sensitivity
```

The repair certificates, the leave-setting-out vs random protocol comparison, the
flat-classifier control, the label-budget curve, the granularity ablation and the
sensitivity sweep. This is the longest run of the four.

```bash
cd pipeline && python held_out_eval.py
```

Held-out validation of the retrieval weights over five stratified splits.

The results that need the full dataset are the detector's own AUROC, AU-PR and
operating point (TPR / FPR), because the detector is trained on raw telemetry
(`mvtflow/`), and the semi-synthetic instantiation sweep (`synth_power.py`),
which injects faults into raw normal cycles.

---

## Data

| Path | Contents |
|---|---|
| `data/records_cache.json` | Per-cycle **evidence records**: detector verdict and score, dominant joint and cycle phase, top-k attributed signals with observed/expected/z, and the fixed-slot signature block. Everything downstream reads this. |
| `data/repair_features.json` | The `unread` supplement — channels present in the parquet that the attribution never reads. By definition these cannot come from the evidence record. |
| `data/payload_torque.json` | Carried-payload torque measured from normal cycles by pose matching (−17.2 N·m on J2 against +14.0 on J3). Used by the semi-synthetic fault injection. |

These are **derived** from voraus-AD, not a copy of it. To regenerate them, or to
train the detector, download
`voraus-ad-dataset-100hz.parquet` from
[the dataset release](https://github.com/vorausrobotik/voraus-ad-dataset) and
place it in `data/`.

---

## Where each number comes from

Every reported figure traces to a script and a JSON key. No number is a constant
written by hand.

| Result | Script | Output |
|---|---|---|
| Prevalence prior, balanced recall | `pipeline/baselines.py` | `results/baselines.json` |
| Tie-break interval on the leave-setting-out splits (paper Sec. 5.2, Fig. 1) | `pipeline/baselines.py` | `results/baselines.json` → `tiebreak_grouped` |
| Tie-break interval on the full 755-cycle set | `pipeline/baselines.py` | `results/baselines.json` → `tiebreak` |
| Twin clusters, routes, cluster accuracy, abstention | `pipeline/distinguishability.py` | `results/distinguishability_results.json` |
| Repair certificates, protocol comparison, two-stage, flat control, granularity ablation, sensitivity | `pipeline/repair.py` | `results/repair_results.json` |
| Label-budget curve and its 20-split robustness check | `pipeline/repair.py` (`label_curve_vs_control`) | `results/label_budget_curve.json` |
| Held-out retrieval weights | `pipeline/held_out_eval.py` | `results/held_out_seeds.json` |
| Instantiation-count sweep, gripping pair | `pipeline/synth_power.py` | `results/synth_instantiation_power.json` |
| Instantiation-count sweep, collision pair | `pipeline/sim_experiments.py --c` | `results/sim_collision_power.json` |
| Full-chain retrieval over all 755 anomalies | `pipeline/integration_test.py` | `results/integration_results.csv` |
| Detector-gradient cross-check (a negative result) | `pipeline/eval_retrieve.py` | `results/gradient_crosscheck.json` |
| Detector AUROC, AU-PR, TPR / FPR (needs the parquet) | `mvtflow/train.py` | `results/detector_metrics.json` (written on training; not shipped) |

Figures are rebuilt from those JSON files:

```bash
python make_figures.py                            # every figure
python make_figures.py fig_teaser fig_signatures  # only the two the paper uses
```

The six-page paper has room for two figures, so it uses the most important ones
and merges where it can:

| Paper | File | Contents |
|---|---|---|
| Fig. 1 | `figures/fig_teaser.pdf` | tie-break interval and balanced recall, merged into one figure |
| Fig. 2 | `figures/fig_signatures.pdf` | structural signatures of the 14 failure modes |

The other figures in `figures/` (protocol, certificates, instantiation power,
granularity, label budget) belong to the long version. Their numbers are in the
paper's tables and text.

---

## Layout

```
safeprocess2027_final.tex / .pdf   the paper
make_figures.py    rebuilds figures/ from results/
pipeline/          the diagnosis pipeline and every evaluation script
  line_kg.py         FMEA knowledge graph, 14 failure modes, retrieval scoring
  attribution_bridge.py  numeric anomaly -> structured evidence record
  distinguishability.py  structural signatures, twin clusters, routes
  repair.py          feature grammar, certificates, leave-setting-out protocol
  baselines.py       the trivial references the ranking has to clear
  synth_faults.py    semi-synthetic fault injection into real normal cycles
  sim_cell.py        rigid-body cell used only for contact instantiations
mvtflow/           normalizing-flow detector (needs the parquet to train)
data/              derived evidence records; see Data above
results/           every reported number, as JSON
figures/           vector PDFs, rebuilt by make_figures.py
voraus-AD_data_source_and_licence_record.pdf   provenance and licence record for the dataset
```

Not every file in `pipeline/` is used by the analysis above. `kb_console.py`,
`kb_feedback.py`, `llm_reasoning.py` and `eval_faithfulness.py` belong to a
separate line of work on the interactive interface and the explanation layer;
they are kept here so that imports resolve and so the pipeline is not presented
in a more finished state than it is.

---

## A note on the protocol

If you use voraus-AD for any learned comparison, **group your splits by
`setting`**. Each value is one physical fault instantiation repeated 4–20 times,
so a random split places near-duplicate cycles on both sides. In our case that
did not merely inflate a number — it reversed which repairs were adopted. The
effect is invisible without grouping, and permuting labels does not reveal it.

`pipeline/repair.py` implements this as `_grouped_split`, and the inner
cross-validation of the greedy search is grouped as well.

---

## Limitations worth knowing before you build on this

- The retrieval stage does not clear a prevalence prior. It should be read as a
  mechanism for producing candidates with provenance, not as a competitive
  ranker.
- Two of the four twin clusters — distal collision and gripping — are never
  ranked first on any of the 755 cycles, so their certificates are computed over
  cycles selected by true label rather than by the pipeline's own output.
- A flat 12-way classifier over the same features reaches `0.525` against the
  two-stage system's `0.309`, and wins at every label budget above eight.
- Simulation results support claims about the *protocol* only. Accuracy
  magnitudes from the simulated cell move with the noise floor and are not
  properties of real collisions.

---

## License

Code is released under the MIT License (`LICENSE`).

Derived data in `data/` and `results/` is released under CC BY 4.0. It is derived
from voraus-AD, which carries its own license — check the
[dataset repository](https://github.com/vorausrobotik/voraus-ad-dataset) and
comply with both.

## Citation

The paper is a SAFEPROCESS 2027 manuscript; full citation details will be added
here on acceptance. Until then, please cite the paper's title and the archived
code and data:

> Zhou, C. and Liang, C.J. *Structural Non-Isolability in Knowledge-Based Robot
> Fault Diagnosis: Tie-Break Intervals and Grouped Validation.* Code and data:
> https://doi.org/10.5281/zenodo.22089173

```bibtex
@misc{zhou2026nonisolability,
  author = {Zhou, Chenxuan and Liang, Christy Jie},
  title  = {Structural Non-Isolability in Knowledge-Based Robot Fault Diagnosis:
            Tie-Break Intervals and Grouped Validation -- code and data},
  year   = {2026},
  doi    = {10.5281/zenodo.22089173},
  url    = {https://doi.org/10.5281/zenodo.22089173}
}
```
