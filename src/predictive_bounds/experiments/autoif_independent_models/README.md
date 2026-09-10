# Independent AutoIF model artifacts

These two entry points default to AutoIF with Qwen2.5-14B as both agent and
target.  They do not read or write each other's artifacts.

Generate all-current-time UPB quantiles from the existing probability cache:

```bash
python -m src.predictive_bounds.experiments.autoif_independent_models.generate_current_time_quantiles
```

The saved dictionary contains `estimated_quantiles` with shape
`(6000, 200, 3000)`, `quantile_levels`, and convention metadata.  Float32 output
is about 14.4 GB and generation temporarily needs about twice that disk space.
Load it lazily with:

```python
artifact = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
quantiles = artifact["estimated_quantiles"]
```

Train the two disjoint-half models and write their separate results:

```bash
python -m src.predictive_bounds.experiments.autoif_independent_models.train_independent_split_models \
  --device cuda:0
```

`upb_model_quantiles.pt` contains only the UPB model and its 3,000 candidate
quantiles.  `difficulty_model_scores.pt` contains only the other model and the
absolute mean-time errors.  Every difficulty row also records its AutoIF class,
integer class index, original `autoif_helper_dataset.csv` row, prompt SHA-256,
and calibration/test membership.  Class assignment is validated through unique
normalized prompt matching, split-index reconstruction, and an independent
replay of the event-time permutation before model training begins.

`difficulty_by_class.csv` has one row for each of the ten classes and numeric,
sortable columns for mean/median/quantile absolute error, RMSE, signed error,
estimated time, restricted time to event, successful-task time to event, and
success rate.  For example:

```python
summary = pd.read_csv(
    "alg_playground_model/autoif_independent_split_models/"
    "difficulty_by_class.csv"
)
hardest = summary.sort_values("absolute_error_mean", ascending=False)
```

Both model files record their original training indices, so disjointness can be
audited after the run. Existing outputs are never overwritten unless
`--overwrite` is passed. Alternate CSV locations can be supplied with
`--autoif-data-path`, `--classifications-path`, and `--class-summary-output`.
