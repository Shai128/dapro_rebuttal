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
generated and source gaps are reported to the terminal. The figure commands
write image files only.

Outputs are organized for direct copying to Overleaf:

- `figures/paper/main`: main-text figures;
- `figures/paper/full/lpb`: complete LPB figures;
- `figures/paper/full/metric/dataset_*`: complete metric-estimation figures;
- `figures/paper/ablations`: DAPRO ablation figures.

The UPB generator contributes the combined AutoIF main-text panels but does not
write a separate full UPB figure suite. Each main-text experiment exports two
equal-sized plot panels and one compact shared-legend image, so the legend is
included only once in LaTeX.

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
