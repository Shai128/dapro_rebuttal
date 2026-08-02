# Full predictive-bound comparison

This package constructs and merges the seven-method comparison requested for the
paper, then generates all grouped figures and one LaTeX table file.

The methods are raw uncalibrated model output, Static Optimized, empirical
Constant, the genuinely distinct CRC power-reach schedule with exponent 2,
Locally Adaptive, row-capped CRC-DAPRO, and an exact infinite-generation-budget
oracle. The oracle observes every calibration outcome, has unit weights, and is
shown in figures/tables but is excluded from every boldface "best method"
comparison. The requested
`1 - lambda ** (alpha * t)` schedule is not used as the improved constant
because continuously tuning `lambda` makes every positive `alpha` exactly the
same family after reparameterization. Power-reach exponent 2 is the strongest
distinct time-only schedule from the earlier screen.

The configuration matrix follows the manuscript:

- 90% LPBs for toxicity, Qwen- and Llama-Guard-judged red team,
  hallucination, and AutoIF;
- budget 20 except hallucination and Llama-Guard red team, which use 10;
- 70% AutoIF UPBs with budget 30 and `tau_prior=0.97`;
- calibration size 3,000, horizon 200, and 50 splits by default;
- Qwen2.5, Llama3.1, Phi-4 Mini, and Gemma3 target models.

Run the complete server matrix from the repository root:

```bash
bash src/predictive_bounds/experiments/full_bounds/scripts/run_all.sh
```

For a local machine containing only some cached target models:

```powershell
& src/predictive_bounds/experiments/full_bounds/scripts/run_all.ps1 `
    -AvailableOnly -SeedEnd 10 -Quality low
```

The Python entry point supports repeatable `--stage` filters, `--config`
filters, `--target-model` filters, and `--dry-run`. Figures are written to
`figures/full/`. Low-quality mode enforces a maximum of 100 KiB per JPEG.
`make_tables.py` writes one copy-paste-ready file containing a table for each
dataset, bound type, and target model.
