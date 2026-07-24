# DAPRO: Dynamic Allocation via Projected Optimization for LLM Safety & Utility Evaluation

Welcome to the **DAPRO LLM Evaluation** project! This repository provides a comprehensive pipeline to robustly evaluate and estimate the safety and utility metrics of Language Models (LLMs) and conversational agents. 

Specifically, this project allows you to:
- Generate multi-turn conversational data for safety and utility testing.
- Construct mathematically provable uncertainty estimates, including **Lower Probability Bounds (LPB)** and **Upper Probability Bounds (UPB)**.
- Estimate safety metrics dynamically using budget allocators.
- Optimize safety metric evaluations under compute constraints using **DAPRO** (Dynamic Allocation via Projected Optimization) and other budget allocation algorithms.

---

## 📖 Project Pipeline Overview

The evaluation pipeline is broken down into modular steps. You can run the entire pipeline end-to-end or execute specific parts as needed.

### 1. Data Generation (`src/multi_turn_data_generation/`)
First, we generate adversarial or standard multi-turn conversations to evaluate the model's safety behavior over extended interactions.
- Run `main.py` inside `multi_turn_data_generation/` to orchestrate and produce raw JSONL conversation logs. **Note: You must run this generation step before computing embeddings.**

### 2. Embedding Computation (`src/multi_turn_data_generation/embedding.py`)
Once conversations are generated, we extract semantic embeddings to represent the dialogue history and responses efficiently.
- Run `embedding.py` to process the JSONL logs and compute tensor embeddings (saved as `.pt` files).

### 3. Survival Model Training (`src/train_model/`)
We treat safety evaluation as a survival analysis problem—predicting *when* a model might fail (or be jailbroken) across conversational turns. 
- A **Transformer Survival Model** is trained on the embeddings to estimate survival probabilities and failure quantiles.

### 4. Safety Evaluation & Bounds Construction (`src/safety_evaluation/`)
With the model trained, we construct confidence bounds and estimate metrics. This is where **Budget Allocators** (like DAPRO, Naive, Uniform, etc.) are utilized to efficiently allocate "compute budget" (e.g., the number of attack attempts to simulate) to different samples.
- **LPB / UPB Construction:** Run `construct_calibrated_bound.py --bound-type lpb` (or `upb`) to compute calibrated lower and upper safety bounds.
- **Metrics Estimation:** Run `estimate_metrics.py` to simulate trajectories and calculate exact metrics (like Cumulative Jailbreak Rate or Cost Per Jailbreak) using IPCW weighting.

### 5. Result Merging
The evaluation outputs are generated across multiple seeds and methods. We merge them into unified DataFrames for easy downstream analysis.
- Run `merge_bounds_results.py` or `merge_estimate_metrics_results.py` to aggregate the `.csv` outputs.

### 6. Visualization & Demonstration (`notebooks/`)
Finally, explore the results interactively! The `notebooks/` directory contains rich Jupyter Notebooks that load the merged dataframes and plot coverage bounds, budget usages, and safety metric comparisons.

---

## 🛠️ Environment Setup

To run this project, ensure you have Python 3.8+ installed. We recommend creating a virtual environment:

```bash
# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

# Install requirements (Assuming standard ML stack)
pip install -r requirements.txt
# If requirements.txt is missing, install the core dependencies:
pip install torch pandas numpy tqdm lifelines scikit-survival transformers jupyterlab matplotlib seaborn
```
*(Make sure to install the appropriate PyTorch version for your CUDA/MPS hardware if you are using GPU acceleration).*

---

## 🚀 How to Run the Pipeline

### Step 1: Generate Data and Embeddings
Navigate to the multi-turn data directory to generate conversations and process embeddings:
```bash
cd src/multi_turn_data_generation

# 1. Generate multi-turn conversation logs
# (Make sure to provide your specific dataset parameters)
python main.py 

# 2. Compute embeddings for generated logs (only AFTER generation is complete)
python embedding.py --idx-start 0 --idx-end 100
```
*(Return to the project root directory before running the next steps).*

### Step 2: Train the Transformer Model
Train the survival model using the generated embeddings.
```bash
python -m src.train_model.train --dataset-name <your_dataset> --dataset-setup <your_setup>
```

### Step 3: Run Safety Evaluation (Metrics or Bounds)
Run evaluations from the project root so Python can resolve the `src.` imports properly.

**To Estimate Safety Metrics:**
```bash
python -m src.safety_evaluation.estimate_metrics \
    --data-type real \
    --dataset-name <dataset_name> \
    --dataset-setup <setup> \
    --budget-per-sample 40 \
    --seed-start 0 --seed-end 10
```

**To Construct Calibrated Bounds (LPB/UPB):**
```bash
python -m src.safety_evaluation.construct_calibrated_bound \
    --bound-type lpb \
    --data-type real \
    --dataset-name <dataset_name> \
    --dataset-setup <setup> \
    --budget-per-sample 10 \
    --seed-start 0 --seed-end 10
```

**To Run the Cross-Setup LPB Experiment:**

This experiment loads the survival-model checkpoint associated with `MODEL_DATASET_SETUP`, applies it to embeddings from `EVALUATION_DATASET_SETUP`, constructs all LPB methods, verifies completion manifests, and merges the results.

```bash
bash src/safety_evaluation/scripts/cross_setup_lpb.sh
```

The script defaults to the red-team Qwen-target model setup evaluated on the red-team Gemma-target setup. Override any setting with environment variables, for example:

```bash
SEED_END=10 BUDGET_PER_SAMPLE=20 DEVICE=cuda:0 \
  bash src/safety_evaluation/scripts/cross_setup_lpb.sh
```

The source setup must already have a trained survival-model checkpoint, and both setups must have matching embedding feature dimensions and time horizons. Open `notebooks/visualize_cross_setup_lpb.ipynb` after the script completes.

**To Run the AutoIF Cross-Class LPB Experiment:**

This experiment calibrates exclusively on `Programming & Technology` AutoIF tasks and evaluates exclusively on `Marketing & Social Media` tasks:

```bash
bash src/safety_evaluation/scripts/autoif_cross_class_lpb.sh
```

The script first verifies a one-to-one correspondence between
`src/multi_turn_data_generation/data/autoif_helper_dataset.csv` and
`src/multi_turn_data_generation/data/classified_instructions.csv` by matching
normalized target text. It then reproduces the dataset loader's split and shuffle
permutations so each class remains attached to the correct survival tensor.

The AutoIF generation results, embeddings, survival tensors, and trained checkpoint
for the selected `DATASET_SETUP` must exist. Generate the missing helper CSV from
the repository root with:

```bash
python -m src.multi_turn_data_generation.data.generate_autoif_dataset \
  --num-samples 10000
```

This writes directly to the path expected by the experiment. Settings can be
overridden with environment variables, for example:

```bash
CAL_SIZE=600 TEST_SIZE=0 SEED_END=10 \
  bash src/safety_evaluation/scripts/autoif_cross_class_lpb.sh
```

`TEST_SIZE=0` evaluates on every eligible row of the requested test class.

**To Evaluate DAPRO Projection Accuracy and Budget Control:**

This experiment compares DAPRO's projected continuation probabilities with the
oracle probabilities obtained by solving on the complete calibration sample. It
measures per-step and cumulative-product error, induced inclusion-probability and
IPCW error, expected-budget error, and the decomposition

```text
realized budget gap = projected expected budget gap + sampling gap.
```

Run the evaluator and strict merger with:

```bash
bash src/safety_evaluation/scripts/evaluate_dapro_projection.sh
```

The defaults evaluate Platt and beta projections with probability and quantile
scores over 50 seeds. Override configurations through environment variables:

```bash
SEED_END=10 PROJECTIONS="platt beta" SCORES="prob" N1_VALUES="50 100 200" \
  bash src/safety_evaluation/scripts/evaluate_dapro_projection.sh
```

Open `notebooks/visualize_dapro_projection_metrics.ipynb` after merging. The
notebook visualizes projection errors, cumulative errors, expected versus
realized budgets, their error decomposition, and the relationship between
cumulative projection error and expected-budget control.

**To Run the Phase-I Optimization-versus-Adaptivity Ablation:**

This paired 50-split analysis compares the existing static optimized policy,
generic random adaptive censoring, a score-percentile heuristic, and the
existing DAPRO Phase-I optimizer. It also produces event-time and initial-score
stratifications, tail-concentration tables, and publication figures. The runner
requires cached trajectories and cached survival-model predictions and will
refuse to regenerate either.

```bash
bash src/safety_evaluation/scripts/phase1_optimization_ablation.sh
```

For a one-split cached-data dry run, append `--dry-run` to the Python command in
the script. When data caches are not present, the analysis code itself can be
validated without any model or API calls using:

```bash
python -m src.safety_evaluation.phase1_optimization_ablation --dry-run-fixture
```

### Step 4: Merge Results
After running your evaluations across multiple seeds, merge the results into a single dataset:
```bash
# For metrics
python -m src.safety_evaluation.merge_estimate_metrics_results \
    --dataset-name <dataset_name> \
    --dataset-setup <setup> \
    --budget-per-sample 40

# For Bounds (LPB/UPB)
python -m src.safety_evaluation.merge_bounds_results \
    --bound-type lpb \
    --dataset-name <dataset_name> \
    --dataset-setup <setup> \
    --budget-per-sample 10
```

### Step 5: Visualize
Start Jupyter Lab and open the notebooks in the `notebooks/` directory to see the plotted results!
```bash
jupyter lab
```
Check out `notebooks/README.md` for a detailed guide on which notebook to use for your specific demonstration or visualization needs.

---

## 🧠 Key Concepts: Budget Allocators

This project introduces several **Budget Allocators** that determine how much compute effort to spend testing a specific conversation. Instead of blindly testing all conversations for a fixed number of turns (which is incredibly expensive), our allocators smartly distribute the budget:

- **Naive / Uniform Allocators:** Spends budget equally or randomly across all samples.
- **Optimized / Adaptive Allocators:** Allocates budget based on the survival model's probability estimates, spending more computational effort on inputs that appear more likely to fail.
- **DAPRO Allocator:** Uses a two-step projection algorithm to dynamically calibrate and assign budget, ensuring mathematically valid bounds while dramatically saving compute costs.

---

## 🧪 Replicating Paper Experiments

To easily replicate all the experiments conducted in our paper, we provide automated shell scripts located across the different pipeline stages. 

These `.sh` files wrap the underlying Python pipeline to execute across all tested configurations, datasets, and bounds:

**1. Data Generation & Embeddings:** (`src/multi_turn_data_generation/scripts/`)
- `generate_all_data.sh`: Automates the generation of conversational logs across all datasets.
- `embeddings.sh`: Automates the semantic embedding extraction.

**2. Model Training:** (`src/train_model/scripts/`)
- `train_model.sh`: Automates the training of the survival transformer models across different setups.

**3. Safety Evaluation:** (`src/safety_evaluation/scripts/`)
- `calibrate.sh`: Runs the LPB calibration experiments.
- `calibrate_upb.sh`: Runs the UPB calibration experiments.
- `compute_metrics.sh`: Runs the safety metric estimation.

To execute any of these, run them from the project root. For example:
```bash
bash src/safety_evaluation/scripts/calibrate.sh
```

---

## 📜 Citation

If you use DAPRO in your research, please cite our paper:

[How Many Iterations to Jailbreak? Dynamic Budget Allocation for Multi-Turn LLM Evaluation](https://arxiv.org/search/?query=Shai+Feldman+Yaniv+Romano+Dynamic+Budget+Allocation&searchtype=all) by Shai Feldman and Yaniv Romano.

```bibtex
@article{feldman2026dapro,
  title={How Many Iterations to Jailbreak? Dynamic Budget Allocation for Multi-Turn LLM Evaluation},
  author={Feldman, Shai and Romano, Yaniv},
  journal={arXiv preprint arXiv:2605.06605},
  year={2026}
}
```
