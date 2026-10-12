# Structural Non-Isolability in Knowledge-Based Robot Fault Diagnosis

Code and derived data for a study of what an FMEA-based diagnosis layer for an
industrial robot can and cannot decide, and which change of representation helps.

The paper, *Structural Non-Isolability in Knowledge-Based Robot Fault Diagnosis:
Tie-Break Intervals and Grouped Validation* (SAFEPROCESS 2027 manuscript), is
`safeprocess2027_final.tex` / `safeprocess2027_final.pdf`. Everything else in
this repository is the experimental record behind it.

The data come from [voraus-AD](https://github.com/vorausrobotik/voraus-ad-dataset):
a six-axis robot doing pick-and-place, 12 anomaly types, 755 anomalous cycles
from 72 fault instantiations (`setting` values).

---

## Main results

The diagnosis layer scores the 14 failure modes of a hand-written FMEA knowledge
base against the per-signal deviations of a cycle and returns a ranked list.
Unless stated otherwise, numbers are means over five leave-setting-out splits,
in which every fault instantiation is wholly in the training half or wholly in
the test half.

**1. Ties.** In 514 of 755 anomalous cycles two or three failure modes share the
top score. Every tie lies inside one of four groups of modes whose declared
signatures (joint scope, signal families, phase hint) are identical or, in the
load-torque group, contain one another. 498 of the ties are in the load-torque
group.

**2. The top-1 figure depends on the declaration order.** Only 12 relative
orders of the tied modes can change the result. Over them, top-1 accuracy ranges
from `0.119` to `0.200` (mean `0.160`, equal to top-1 with tied modes sharing the
credit). The group of the top-ranked mode is the same under every order
(group accuracy `0.342`).

**3. A prevalence prior is a better reference than uniform chance.** Ranking the
modes by their training-half frequency, the same for every cycle, gives top-1
`0.187`, inside the range above, and is better than the retrieval on top-3,
top-5, MRR and group accuracy under every order. On balanced recall (mean
per-category recall, tied modes sharing credit) the retrieval is slightly ahead,
`0.105` against `0.095`.

**4. Certificates.** For each group, a greedy search over named statistics of
the existing signals looks for one that separates the members, and reports the
result with held-out accuracy, three controls and a cost class. For the
load-torque group the sign of the deviation does it: `0.840` within the group on
unseen instantiations, against `0.562` for the deployed features. The free set
{sign, step} reaches `0.869`. Collision: not adopted (no grouped search can run
with one carton instantiation per training half). Gripping: undetermined (one
instantiation per mode). Commutation/over-temperature: untestable (no
over-temperature instance).

**5. Grouped validation changes the decision.** Under a random split over cycles
the search adopts a collision repair on every split. With one setting held out
at a time, the same search and rule adopt it on 4 of 9 folds, and its held-out
accuracy (`0.67` to `0.75`) is below the majority rate (`0.766`). A
permuted-label control over cycles does not detect this; a permutation over
whole settings does.

**6. Controls.** A flat 12-way classifier over the same features reaches `0.525`
top-1 against `0.309` for the two-stage system, and is ahead from about 8
labelled cycles. Sending every cycle to the load-torque resolver gives `0.349`,
so the two-stage gain comes from the resolver, not from the routing.

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

Most results run from the derived evidence records in `data/`; the 1 GB parquet
is not needed. Scripts import each other flatly, so run them from inside
`pipeline/`.

| Command | What it computes | Output |
|---|---|---|
| `python baselines.py` | prevalence prior, balanced recall (declared order and tie-shared), exact tie-break enumeration, tie-robust comparison, set-valued coverage, constant router | `results/baselines.json` |
| `python distinguishability.py` | twin groups, routes and the margin `tau`, tie composition, phase-weight sweep, knowledge-base columns, localization, authoring sensitivity | `results/distinguishability_results.json` |
| `python repair.py --control --labels --sensitivity` | repair certificates, random vs leave-setting-out comparison, flat classifier, label-budget curve, granularity ablation, sensitivity sweep (the longest run) | `results/repair_results.json`, `results/label_budget_curve.json` |
| `python certificate_checks.py` | per-split searched dimensions, nested fixed configuration, routing, within-group confusion, all dimension subsets, mode-level signature term, label budget over 20 splits | `results/certificate_checks.json` |
| `python protocol_check.py` | collision group under random and grouped validation, nine-fold comparison, setting-level permutation tests (about 20 minutes) | `results/protocol_check.json` |
| `python held_out_eval.py` | the deployed retrieval weights re-chosen on each leave-setting-out training half; the earlier stratified experiment under `legacy` | `results/held_out_seeds.json` |

These need the parquet: the detector (`mvtflow/`), the semi-synthetic
instantiation sweep (`synth_power.py`), the full-chain run over all 755
anomalies (`integration_test.py`), which rebuilds the evidence records from raw
telemetry, and the separability probe (`distinguishability.py --probe`).

---

## Data

| Path | Contents |
|---|---|
| `data/records_cache.json` | Per-cycle **evidence records** for the 755 anomalous cycles: dominant joint and cycle phase, the five channels with the largest peak z (observed, expected, z) and the fixed-slot signature block. The `verdict` field is the ground-truth label; no detector score is stored, so every diagnosis number assumes perfect detection. |
| `data/repair_features.json` | The `unread` supplement: channels present in the parquet that the attribution never reads. |
| `data/payload_torque.json` | Carried-payload torque measured from normal cycles by pose matching (-17.2 N·m on J2, +14.0 on J3). Used by the semi-synthetic fault injection. |

These are **derived** from voraus-AD, not a copy of it. To regenerate them, or to
train the detector, download `voraus-ad-dataset-100hz.parquet` from
[the dataset release](https://github.com/vorausrobotik/voraus-ad-dataset) and
place it in `data/`.

---

## Where the paper's numbers come from

| Paper | Source |
|---|---|
| Sec. 3, retrieval weights re-chosen on training halves | `held_out_seeds.json` |
| Sec. 4, ties, groups, phase, routes, `tau`, authoring sensitivity | `distinguishability_results.json` |
| Table 1 and Sec. 5.1 | `baselines.json` → `prior_vs_retrieval`, `tie_robust`, `balanced_recall_expected`, `balanced_recall` |
| Fig. 1 and Sec. 5.2 | `baselines.json` → `tiebreak_exact`, `set_valued` |
| Table 2 and Sec. 6 | `repair_results.json` (certificates, fixed configuration, summary), `certificate_checks.json` (searched dims, nested, subsets, within-group pairs), `baselines.json` → `constant_router` |
| Sec. 7 | `protocol_check.json`, `repair_results.json` → `leakage_check` |
| Sec. 8 | `certificate_checks.json` → `label_budget`, `routing_fixed`; `repair_results.json` → `control_flat_classifier` |
| Conclusion, mode-level signature term | `certificate_checks.json` → `mode_signature` |
| Detector AUROC (Sec. 3) | `mvtflow/train.py` → `results/detector_metrics.json` (needs the parquet; not shipped) |

Other results files: `synth_instantiation_power.json` (`synth_power.py`, needs
the parquet), `sim_collision_power.json` and `sim_experiments.json`
(`sim_experiments.py`, a simulated cell that is not a model of the voraus arm),
`integration_results.csv` (`integration_test.py`), `gradient_crosscheck.json`
(`eval_retrieve.py --gradient`, which reads an interface cache that is not
shipped).

---

## Figures and building the paper

```bash
python make_figures.py                                    # every figure
python make_figures.py fig_tiebreak fig_signatures_column # the two the paper uses
```

| Paper | File | Contents |
|---|---|---|
| Fig. 1 | `figures/fig_tiebreak.pdf` | top-1 under all 12 relevant tie-break orders, and group accuracy |
| Fig. 2 | `figures/fig_signatures_column.pdf` | knowledge-base columns of the modes in the four groups |

The other figures in `figures/` belong to the long version.

To build the paper, put IFAC's official `ifacconf.cls` (from the
[IFAC author guide](https://www.ifac-control.org/conferences/author-guide)) next
to `safeprocess2027_final.tex` and run `pdflatex` twice. Without it,
`tools/ifacconf_local.cls` can stand in: copy it next to the `.tex` as
`ifacconf.cls`. It is a local approximation for compiling and page-counting
only, not for submission. The committed PDF was built with it: six pages.

---

## Layout

```
safeprocess2027_final.tex / .pdf   the paper
make_figures.py    rebuilds figures/ from results/
pipeline/          the diagnosis pipeline and every evaluation script
  line_kg.py             FMEA knowledge graph, 14 failure modes, retrieval scoring
  attribution_bridge.py  raw cycle -> evidence record
  distinguishability.py  signatures, twin groups, routes
  repair.py              dimensions, certificates, leave-setting-out protocol
  baselines.py           prior, tie-break enumeration, balanced recall
  certificate_checks.py  checks on the load-torque certificate and label budget
  protocol_check.py      random vs grouped validation for the collision group
  held_out_eval.py       retrieval weights re-chosen on training halves
  synth_faults.py        semi-synthetic fault injection into real normal cycles
  sim_cell.py            simulated cell used only for contact instantiations
mvtflow/           normalizing-flow detector (needs the parquet to train)
data/              derived evidence records; see Data above
results/           every reported number, as JSON
figures/           vector PDFs, rebuilt by make_figures.py
tools/             ifacconf_local.cls, a stand-in for IFAC's class
voraus-AD_data_source_and_licence_record.pdf   provenance and licence record for the dataset
```

`kb_console.py`, `kb_feedback.py`, `llm_reasoning.py` and `eval_faithfulness.py`
belong to separate work on the interactive interface and the explanation layer;
they are kept so that imports resolve.

---

## On the validation protocol

For any learned comparison on voraus-AD, group the splits by `setting`, in the
inner validation as well as the outer split. Each value is one physical fault
instantiation repeated 4 to 20 times, so a random split puts cycles of the same
instantiation on both sides. In this study that changed which repair was
adopted. `pipeline/repair.py` implements the grouped split as `_grouped_split`,
and the inner cross-validation of the search is grouped as well.

---

## Limitations

- Every diagnosis number assumes perfect detection: all 755 labelled anomalous
  cycles are diagnosed.
- The knowledge base was written by the authors from the voraus-AD anomaly
  descriptions. The tie count depends on it: declaring axis weight's phase hint
  as transfer instead of any leaves 315 ties instead of 514.
- The retrieval does not beat the prevalence prior. It produces candidates with
  provenance; it is not a competitive ranker.
- The collision and gripping groups are never ranked first, so their
  certificates are computed on cycles selected by true label.
- A flat 12-way classifier over the same features is more accurate than the
  whole pipeline once more than a few labels are available.
- Simulated results (`sim_*`) describe the simulated cell only.

---

## License

Code is released under the MIT License (`LICENSE`).

Derived data in `data/`, `results/` and `pipeline/repair_adopted.json` (a
classifier fitted on voraus-AD cycles), and the figures in `figures/` drawn from
`results/` (all but `fig_signatures`), are an adaptation of the voraus-AD dataset
(Brockmann, Rudolph, Rosenhahn and Wandt, *The voraus-AD dataset for anomaly
detection in robot applications*, IEEE Transactions on Robotics; rights holder
voraus robotik GmbH), which is licensed under
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). Its
ShareAlike and NonCommercial terms carry over, so this material is released
under CC BY-NC-SA 4.0 too: non-commercial use only, with attribution, and
adaptations under the same licence. It contains per-cycle values and labels
computed from `voraus-ad-dataset-100hz.parquet`; the dataset itself is not
redistributed here (see the
[dataset repository](https://github.com/vorausrobotik/voraus-ad-dataset)).
`voraus-AD_data_source_and_licence_record.pdf` records the source and licence.

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
