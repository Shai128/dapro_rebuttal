# Predictive-bound ablations

The ablations isolate two implementation choices:

- `phase1_optimization.py`: separates calibration-split, Phase-I policy, and
  acquisition randomness and measures their effect on variance and budget use.
- `random_floor.py`: compares terminal propensity-floor strategies and reports
  expected/realized budget and inverse-propensity diagnostics.

Linux examples:

```bash
bash src/predictive_bounds/ablations/scripts/run_phase1_optimization.sh
bash src/predictive_bounds/ablations/scripts/run_random_floor.sh
```

Windows PowerShell examples:

```powershell
.\src\predictive_bounds\ablations\scripts\run_phase1_optimization.ps1
.\src\predictive_bounds\ablations\scripts\run_random_floor.ps1
```

All defaults can be overridden with the environment variables listed near the
top of the corresponding script. Both modules also expose `--help`.
