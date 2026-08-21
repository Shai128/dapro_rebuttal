# Final-paper figure generation

Run the three task-specific scripts from the repository root with the project
environment:

```powershell
C:\Users\Shai1\anaconda3\envs\oqr\python.exe -m src.predictive_bounds.experiments.full_bounds.summarize_lpb_paper --quality low
C:\Users\Shai1\anaconda3\envs\oqr\python.exe -m src.predictive_bounds.experiments.full_bounds.summarize_upb_paper --quality low
C:\Users\Shai1\anaconda3\envs\oqr\python.exe -m src.evaluation.summarize_paper --quality low
```

Use `--quality high` for 300-DPI JPEGs. Add `--strict` to fail if any requested
method or diagnostic is missing. Without `--strict`, supported figures are
generated and all omissions are written to `figures/paper/data/*_schema_gaps.csv`.

Outputs are organized for direct copying to Overleaf:

- `figures/paper/main`: main-text figures;
- `figures/full`: LPB and UPB appendix figures;
- `figures/metrics/dataset_*`: metric-estimation appendix figures;
- `figures/paper/data`: source inventory, selected seed-level rows, schema-gap
  reports, and figure manifests.

LPB filenames use the requested dataset stem directly. UPB filenames insert
`_upb_` before the metric so LPB and UPB figures cannot overwrite one another.
The file `estimated_rmttu_boxplot.jpg` plots the standard restricted mean
`E[min(T, 200)]`; it never silently plots the historical conditional event-time
diagnostic.

`Budget Used per Sample` is method-aware: Static uses assigned
`sum(C_i) / n`, while DAPRO uses actual event-stopped/generated turns per
sample across policy fitting, CRC control, and deployment.
Allocation-only diagnostics (assigned budget, observed events, and all
`A_i/pi_i` weight plots, including the `alpha=0.1` target) intentionally omit
Uncalibrated and Oracle because neither is a finite-budget allocation method.
