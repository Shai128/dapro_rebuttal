import abc
import dataclasses

import numpy as np
import torch


def summarize_expected_budget(
        total_expected_budget: float,
        n_samples: int,
        budget_per_sample: float,
        *,
        cost_semantics: str,
) -> dict:
    """Return the common expected-budget diagnostics used by allocators."""
    if n_samples <= 0:
        raise ValueError("`n_samples` must be positive.")
    total_expected_budget = float(total_expected_budget)
    configured_total_budget = float(budget_per_sample) * n_samples
    expected_per_sample = total_expected_budget / n_samples
    gap = total_expected_budget - configured_total_budget
    tolerance = 1e-7 * max(1.0, n_samples)
    return {
        "configured_total_budget": configured_total_budget,
        "configured_budget_per_sample": float(budget_per_sample),
        "total_expected_budget": total_expected_budget,
        "total_expected_budget_per_sample": expected_per_sample,
        "total_expected_budget_gap": gap,
        "total_expected_budget_gap_per_sample": (
            expected_per_sample - float(budget_per_sample)
        ),
        "total_expected_budget_valid": int(gap <= tolerance),
        "expected_budget_cost_semantics": cost_semantics,
    }


@dataclasses.dataclass
class BudgetAllocationResult:
    f: torch.Tensor
    C: torch.Tensor
    C_probs: torch.Tensor
    total_budget_used: int = None
    mean_weight: float = None
    max_weight: float = None
    additional_metrics: dict = None
    # Optional N-by-J probability of reaching each row-specific candidate
    # bound.  UPB calibration needs this candidate-specific propensity for
    # the survivor event 1{T > f_j}; a single terminal/event propensity is
    # insufficient for a history-adaptive allocation.
    candidate_C_probs: torch.Tensor = None
    # Optional N-by-M conditional continuation probabilities of the executed
    # policy.  Their row-wise cumulative products are the probabilities of
    # observing successive prefixes.  Sequential augmented-HT estimators use
    # this richer pathwise propensity record; terminal-only estimators may
    # continue to rely on ``C_probs`` and ``candidate_C_probs``.
    continuation_probabilities: torch.Tensor = None


def candidate_reach_probabilities(
        continuation_probabilities: torch.Tensor,
        candidates: torch.Tensor,
        *,
        infinity_value: int = 201,
) -> torch.Tensor:
    """Gather cumulative reach probabilities at row-specific UPB values."""
    probabilities = continuation_probabilities.to(torch.float64)
    if probabilities.ndim != 2 or candidates.ndim != 2:
        raise ValueError("Continuation probabilities and candidates must be matrices.")
    if len(probabilities) != len(candidates):
        raise ValueError("Probability and candidate rows must agree.")
    cumulative = probabilities.cumprod(dim=1)
    finite = candidates < infinity_value
    indices = candidates.to(torch.long).clamp(
        min=1, max=probabilities.shape[1]
    ) - 1
    gathered = cumulative.gather(1, indices)
    # The infinite UPB is deterministically valid and never reweighted.
    return torch.where(finite, gathered, torch.ones_like(gathered))

class BudgetAllocator(abc.ABC):
    def __init__(self, budget_per_sample: float, taus_range: torch.Tensor, tau_prior: float):
        self.budget_per_sample = budget_per_sample
        self.taus_range = taus_range
        self.tau_prior = tau_prior
        # Optional, explicit stream used only for stochastic acquisition.
        # Leaving it as ``None`` preserves the historical single-stream
        # behavior.  Fixed-data experiments set this after constructing the
        # calibration method so Phase-I fitting and Phase-II sampling can be
        # replicated independently.
        self.acquisition_seed = None
        # Optional common-random-number table aligned with the rows passed to
        # ``allocate_budget``.  The experiment driver constructs this table
        # with NumPy on CPU and maps it from original dataset indices before
        # installing it on every allocator.  Individual allocators may then
        # select their Phase-II rows without changing the draw assigned to an
        # example.
        self.acquisition_uniforms = None

    def reset_acquisition_rng(self) -> None:
        """Reset the legacy Torch acquisition stream.

        New experiments install ``acquisition_uniforms`` and do not depend on
        this process-global stream.  Retaining the method keeps direct and
        historical callers that do not supply common random numbers backward
        compatible.
        """
        if self.acquisition_seed is None:
            return
        seed = int(self.acquisition_seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

    def set_acquisition_randomness(
            self,
            *,
            seed: int | None,
            uniforms: np.ndarray | torch.Tensor | None,
    ) -> None:
        """Install a dedicated acquisition stream for one experiment.

        ``uniforms`` must be aligned with the full row order supplied to
        :meth:`allocate_budget`; allocator-specific Phase-I/Phase-II splits
        select rows from it.  Keeping the source table on CPU avoids
        device-specific random-number streams.
        """
        self.acquisition_seed = None if seed is None else int(seed)
        if uniforms is None:
            self.acquisition_uniforms = None
            return
        values = (
            uniforms.detach().cpu().numpy()
            if torch.is_tensor(uniforms)
            else np.asarray(uniforms)
        )
        if values.ndim != 2:
            raise ValueError(
                "`uniforms` must have shape (number of samples, time); "
                f"got {values.shape}."
            )
        if not np.issubdtype(values.dtype, np.floating):
            raise ValueError("`uniforms` must have a floating-point dtype.")
        if not np.all(np.isfinite(values)):
            raise ValueError("`uniforms` must contain only finite values.")
        if np.any(values < 0) or np.any(values >= 1):
            raise ValueError("`uniforms` must lie in [0, 1).")
        contiguous = np.ascontiguousarray(values)
        # Main experiments provide one immutable table to every allocator, so
        # keep that storage shared.  Direct callers that provide a writable
        # array receive a defensive copy.
        if contiguous.flags.writeable:
            contiguous = contiguous.copy()
        contiguous.setflags(write=False)
        self.acquisition_uniforms = contiguous

    def get_acquisition_uniforms(
            self,
            n_samples: int,
            n_times: int,
            *,
            device: torch.device | str,
            dtype: torch.dtype,
    ) -> torch.Tensor | None:
        """Return device-local common random numbers aligned to input rows.

        If no table was installed but an explicit acquisition seed exists,
        generate the same NumPy table locally.  If neither exists, return
        ``None`` so historical direct callers retain their Torch-RNG behavior.
        """
        if n_samples < 0 or n_times < 0:
            raise ValueError("`n_samples` and `n_times` must be nonnegative.")
        if self.acquisition_uniforms is not None:
            values = self.acquisition_uniforms
            if values.shape[0] != n_samples:
                raise ValueError(
                    "Acquisition uniform row count does not match allocator "
                    f"input: {values.shape[0]} != {n_samples}."
                )
            if values.shape[1] < n_times:
                raise ValueError(
                    "Acquisition uniform table is too narrow: "
                    f"{values.shape[1]} < {n_times}."
                )
            values = values[:, :n_times]
        elif self.acquisition_seed is not None:
            values = np.random.default_rng(
                int(self.acquisition_seed)
            ).random((n_samples, n_times))
        else:
            return None
        return torch.tensor(values, dtype=dtype, device=device)

    @abc.abstractmethod
    def allocate_budget(self, probability_est: torch.Tensor, x: torch.Tensor, t: torch.Tensor,
                        quantile_est: torch.Tensor) -> BudgetAllocationResult:
        pass

    @property
    @abc.abstractmethod
    def name(self) -> str:
        pass
