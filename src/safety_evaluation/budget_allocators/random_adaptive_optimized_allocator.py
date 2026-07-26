import numpy as np
import torch

from src.safety_evaluation.budget_allocators.budget_allocator import BudgetAllocator, BudgetAllocationResult
from src.safety_evaluation.budget_allocators.vectorized_adaptive_allocator_patch import simulate_process_vectorized
from src.safety_evaluation.calibration.calibration_utils import get_prior


class RandomAdaptiveOptimizedBudgetAllocator(BudgetAllocator):
    """
    Implements a Lagrangian Shadow Price method for optimal budget allocation.

    Mechanism:
    1. Uses the fully observed validation set (first 100 samples) to calibrate a global
       threshold 'lambda' (shadow price). This lambda balances the trade-off between
       budget cost and event discovery probability.
    2. Applies this threshold to the test set in a decentralized, parallel manner.
    3. Ensures a minimum exploration probability to maintain valid IPW properties.
    """

    def __init__(self, conditional_grid, budget_per_sample, taus_range, tau_prior, m_upper_bound, reach_t_max_is_success=False):
        super().__init__(budget_per_sample, taus_range, tau_prior)
        self.conditional_grid = conditional_grid
        self.min_pi = 0.005
        self.reach_t_max_is_success = reach_t_max_is_success
        # self.budget_per_sample *= 500

    @property
    def name(self) -> str:
        return "random_adaptive_optimized"

    def allocate_budget(self, probability_est: torch.Tensor, x: torch.Tensor, t: torch.Tensor,
                        quantile_est: torch.Tensor) -> BudgetAllocationResult:
        device = self.conditional_grid.device
        N, T_max_curr, T_max_future = self.conditional_grid.shape
        total_budget = self.budget_per_sample * N
        val_size = 100
        perm = np.random.permutation(N)
        val_idxs = perm[:val_size]
        test_idxs = perm[val_size:]
        # --- Data Splitting ---
        # Validation Set: Used to learn the optimal policy parameters (lambda)
        val_grid = self.conditional_grid[val_idxs]

        val_prior_q = get_prior(quantile_est[val_idxs], self.taus_range, self.tau_prior)
        t_val = t[val_idxs]
        val_budget_used = torch.minimum(t_val + 1, val_prior_q + 1).sum().item()
        if total_budget < val_budget_used:
            raise ValueError("Total budget is too small")

        # Test Set: The data we need to mine
        test_grid = self.conditional_grid[test_idxs]

        # val_grid, test_grid = cal(val_grid, test_grid, t_val)

        test_prior_q = get_prior(quantile_est[test_idxs], self.taus_range, self.tau_prior)
        t_test = t[test_idxs]

        # Global Target Budget for the test set
        # We assume the budget density (budget per sample) should be consistent
        target_budget_avg = (self.budget_per_sample * N - val_budget_used) / (N - val_size)
        val_expected_remaining = torch.ones_like(val_grid[..., 0], device=device)
        test_expected_remaining = torch.ones_like(test_grid[..., 0], device=device)

        lam_low, lam_high = 0.0, 1.0

        for _ in range(25):
            mid = (lam_low + lam_high) / 2

            _, val_C_probs, val_expected_cost = simulate_process_vectorized(
                val_expected_remaining,
                val_prior_q,
                t_val,
                mid,
                stochastic=False,
                reach_t_max_is_success=self.reach_t_max_is_success,
            )

            assert val_expected_cost <= val_budget_used + 1e-6
            avg_cost = val_expected_cost / val_size

            if abs(avg_cost - target_budget_avg) < 1e-10:
                break

            if avg_cost < target_budget_avg:
                lam_low = mid
            else:
                lam_high = mid

        best_lambda = (lam_low + lam_high) / 2

        test_C, test_C_probs, test_total_used = simulate_process_vectorized(
            test_expected_remaining,
            test_prior_q,
            t_test,
            best_lambda,
            stochastic=True,
            reach_t_max_is_success=self.reach_t_max_is_success,
        )

        test_avg_cost = test_total_used / len(test_C)
        # print(f"err: {val_avg_cost - test_avg_cost}")
        total_budget_used = test_total_used + val_budget_used

        # Reconstruct full-size tensors
        # Val C: Standard logic (stop at event or horizon)
        prior_q = torch.empty(N, device=device)
        prior_q[val_idxs] = val_prior_q
        prior_q[test_idxs] = test_prior_q
        val_C = val_prior_q + 1  # torch.minimum(t[:val_size], val_prior_q)
        # For Validation set, we don't care about C_probs (set to 1.0 or dummy)
        val_C_probs = torch.ones(val_size, device=device)

        # Concatenate
        final_C = torch.empty(N, device=device)
        final_C[val_idxs] = val_C
        final_C[test_idxs] = test_C.to(final_C.dtype)

        # final_C[final_C > t] = torch.max(prior_q[final_C > t], final_C[final_C > t])
        final_C_probs = torch.empty(N, device=device, dtype=test_C_probs.dtype)
        final_C_probs[val_idxs] = val_C_probs.to(final_C_probs.dtype)
        final_C_probs[test_idxs] = test_C_probs.to(final_C_probs.dtype)
        mean_val_weight = (1/final_C_probs).mean().item()
        max_val_weight = (1/final_C_probs).max().item()

        print(f"random adaptive weights {mean_val_weight} | lambda {best_lambda} | total_budget_used {total_budget_used} "
              f"| total_budget {total_budget} | # observed: {(final_C > t).float().sum().item()}"
              f"| achieved prior: {(final_C.squeeze() >= prior_q).float().sum().item()}")
        return BudgetAllocationResult(quantile_est, final_C, final_C_probs, total_budget_used, mean_weight=mean_val_weight,
                                      max_weight=max_val_weight)
