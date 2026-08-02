# Additional experiments

These modules are analyses and extensions of the primary predictive-bound
pipeline. They are grouped so that the production entry points remain easy to
find.

- `analysis/`: post-processes stored DAPRO variance, weighting, and allocation
  diagnostics.
- `autoif_cross_class/`: calibrates and evaluates on different AutoIF classes.
- `distribution_shift/`: train+calibration/test and attacker-only shifts.
- `allocation_focus/`: per-sample allocation and difficulty-quartile audits.
- `variance_components/`: freezes data, Phase-I fitting, or acquisition RNG and
  runs crossed decompositions over several Phase-I sizes.
- `budget_distribution/`: realized-budget histograms, tail-overrun tables, and
  concentration diagnostics.
- `limited_training_budget/`: evaluates survival models trained after uniform
  acquisition of only a specified fraction of trajectory labels.
- `judge_noise/`: calibration-only false-negative, false-positive, and combined
  judge-label noise.
- `cross_setup/`: evaluates a model checkpoint on a different dataset setup.
- `dapro_projection/`: measures Phase-I-to-Phase-II projection and budget error.
- `metrics/`: estimates and merges safety metrics under each allocator.

Each runnable experiment with multiple steps has a script under its local
`scripts/` directory (normally `run_all.sh`). Run scripts from any directory;
they resolve and switch to the repository root themselves. Configuration is
supplied through the environment variables declared near the top of each
script.

Every new experiment follows the same lifecycle: construct immutable seed
shards, write a completion manifest, perform a strict merge that refuses
partial results, then generate PDF figures and copy-paste-ready LaTeX. Run all
eight requested experiments with:

```bash
TEST_SETUP='...' ATTACKER_TEST_SETUP='...' \
  bash src/predictive_bounds/experiments/scripts/run_all_ablations.sh
```

`TEST_SETUP` and `ATTACKER_TEST_SETUP` are explicit because those held-out
server caches are installation-specific. All other parameters have paper
defaults and can be overridden with the environment variables in each runner.
For a genuine cross-dataset training shift, additionally set
`MODEL_DATASET_NAME`, `CALIBRATION_DATASET_NAME`, and `TEST_DATASET_NAME`;
calibration and test must name the same target domain, while the source model
domain must differ in its dataset name, setup, or both.

To run several shift triples, set `SHIFT_CONFIG_FILE` to a JSON list. Each
entry contains `shift_type`, the three `*_dataset_setup` fields, optional
shared or role-specific dataset names, and any numeric overrides. An optional
`name` chooses the figure subdirectory. Both shift shell scripts filter this
one matrix to their own experiment type, so the combined runner can consume a
single server-specific file.
The AutoIF cross-class, allocation-focus, variance, judge-noise, and
limited-training runners accept `ALL_CONFIGS=1`; they then import the relevant
dataset/target-model configurations from `full_bounds/config.py`. Add
`AVAILABLE_ONLY=1` to restrict that matrix to prediction caches present on the
current machine. The combined runner enables `ALL_CONFIGS=1` by default.

The analysis modules have dedicated runners in `analysis/scripts/`. The
A-weighted runner requires `MERGED_CSV` and `EXPERIMENT_SUFFIX`; the optimal
DAPRO runner requires one `INPUT=DATASET=PATH` environment value and accepts
additional `--input DATASET=PATH` arguments.
