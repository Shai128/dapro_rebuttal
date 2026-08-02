# CRC and dynamic-schedule audit

This experiment compares the variance-aligned two-bin DAPRO policy with and
without an independent budget-control fold.  It also evaluates low-capacity
time schedules whose scalar aggressiveness is selected by conformal risk
control (CRC).

The methods are:

- two-bin DAPRO with projection reserves 0 and 1;
- two-bin CRC-DAPRO with 100/100 and 200/100 policy-fit/control splits;
- empirical and CRC constant continuation;
- the requested conditional schedule
  `p_t = 1 - lambda ** (alpha * t)` for `alpha in {0.5, 1, 2}`;
- cumulative power reach `R_t = p ** (t ** alpha)` for the same exponents.

The requested three alpha values are deliberately retained as an executable
check even though they define the same family after the reparameterization
`beta = lambda ** alpha`.

The selected public method is the row-capped CRC-DAPRO variant with `N1=200`,
100 policy-fit rows, 100 independent CRC rows, and a label-free per-row
candidate-cost cap of twice the target budget.  Reproduce the cap screen and
the independent 20-seed confirmation with:

```bash
bash src/predictive_bounds/experiments/crc_dynamic_schedules/scripts/run_row_caps.sh
bash src/predictive_bounds/experiments/crc_dynamic_schedules/scripts/run_rowcap2_confirm.sh
```

The corresponding PowerShell runners have the same base names and `.ps1`
extensions.  Checked-in confirmation summaries are under
`outputs/crc_rowcap2_confirm/`.

Run from the repository root on Linux:

```bash
bash src/predictive_bounds/experiments/crc_dynamic_schedules/scripts/run.sh
```

or Windows PowerShell:

```powershell
& src/predictive_bounds/experiments/crc_dynamic_schedules/scripts/run.ps1
```

Set `PYTHON`, `DEVICE`, `SEED_START`, `SEED_END`, and `EXPERIMENT_SUFFIX` to
override the defaults.  Then summarize the merged results with:

```bash
python -m src.predictive_bounds.experiments.crc_dynamic_schedules.analyze \
  --suffix crc_dynamic_schedules_v1
```
