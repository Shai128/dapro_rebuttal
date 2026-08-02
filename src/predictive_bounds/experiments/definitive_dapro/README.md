# Historical projection-DAPRO experiment

This directory reproduces the earlier comparison between assumption-based
projection DAPRO and the constant-probability reference. It is retained as a
versioned ablation and is **not** the final public DAPRO experiment. The method
studied here is fixed at:

- Phase-I size `N1=200`;
- raw target event at `alpha=0.10`;
- objective weights `(A + 0.001) / 1.001`;
- two empirical score bins per interaction time;
- terminal inclusion-probability floor `pi >= 0.005`, enforced inside the
  Phase-I budget correction;
- projection-error reserve of one expected interaction per Phase-II row.

Run all five real-data settings from the repository root:

```bash
bash src/predictive_bounds/experiments/definitive_dapro/scripts/run.sh
```

or on Windows PowerShell:

```powershell
& src/predictive_bounds/experiments/definitive_dapro/scripts/run.ps1
```

Both runners accept environment overrides for `PYTHON`, `DEVICE`,
`SEED_START`, `SEED_END`, and `EXPERIMENT_SUFFIX`. They require the cached
model predictions under `alg_playground_model/` and do not regenerate model or
red-team data.

Regenerate the report tables and figures from the checked-in audit summaries:

```bash
python -m src.predictive_bounds.experiments.definitive_dapro.analyze
```

Outputs are written to `outputs/dapro_definitive_report/`. The 50-seed final
comparison and the older 100-seed candidate screen are kept separate because
they were produced by different, explicitly versioned runs.

For the current public CRC-DAPRO method, dynamic-schedule screen, row-cap
ablation, and matched confirmation, use
`src/predictive_bounds/experiments/crc_dynamic_schedules/README.md`.
