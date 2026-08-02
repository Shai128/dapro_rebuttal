import torch


def _enforce_terminal_probability_floor(
    probabilities: torch.Tensor,
    prior_q: torch.Tensor,
    terminal_pi_min: float | None,
    terminal_floor_mode: str = "mixture",
) -> torch.Tensor:
    if terminal_floor_mode not in {"mixture", "hard", "none"}:
        raise ValueError(
            "`terminal_floor_mode` must be one of: mixture, hard, none."
        )
    if terminal_pi_min is None or terminal_floor_mode == "none":
        return probabilities
    if not 0 < terminal_pi_min <= 1:
        raise ValueError("`terminal_pi_min` must lie in (0, 1].")
    probabilities64 = probabilities.to(torch.float64)
    q = prior_q.to(
        device=probabilities.device,
        dtype=torch.long,
    ).clamp(min=0, max=probabilities.shape[1])
    time = torch.arange(
        probabilities.shape[1],
        device=probabilities.device,
    ).unsqueeze(0)
    active = time < q.unsqueeze(1)
    raw_path = torch.where(
        active,
        probabilities64.clamp_min(torch.finfo(torch.float64).tiny),
        torch.ones((), dtype=torch.float64, device=probabilities.device),
    )
    raw_cumulative = raw_path.cumprod(dim=1)
    if terminal_floor_mode == "mixture":
        floored_cumulative = (
            terminal_pi_min
            + (1 - terminal_pi_min) * raw_cumulative
        )
    else:
        # Minimal cumulative clamp: prefixes that already exceed the floor
        # remain unchanged.
        floored_cumulative = raw_cumulative.clamp_min(terminal_pi_min)
    previous_mixed = torch.cat(
        [
            torch.ones(
                (len(probabilities), 1),
                dtype=torch.float64,
                device=probabilities.device,
            ),
            floored_cumulative[:, :-1],
        ],
        dim=1,
    )
    mixed_conditionals = (
        floored_cumulative / previous_mixed.clamp_min(
            torch.finfo(torch.float64).tiny
        )
    ).clamp(max=1.0)
    return torch.where(active, mixed_conditionals, probabilities64)


@torch.no_grad()
def precompute_expected_remaining(
    grid: torch.Tensor,
    prior_q: torch.Tensor,
    *,
    sample_chunk_size: int | None = 256,
) -> torch.Tensor:
    """
    Vectorized replacement for the per-time `get_mpc_decision` belief calculation.

    Quantiles and event times are one-based interaction counts.  At zero-based
    current index ``t``, the remaining acquisition horizon is therefore
    ``prior_q - t`` and the conditional expected cost is:

        belief = grid[:, t, t:]
        remaining = clamp(prior_q - t, 0, belief.shape[1])
        expected = sum_r belief[r] * min(r + 1, remaining)

    The function vectorizes over all time indices.  Optional chunking is only
    over samples to keep peak memory bounded; there is no Python time loop.
    """
    if grid.ndim != 3:
        raise ValueError(f"`grid` must have shape (N,T,F); got {tuple(grid.shape)}.")

    n, t_max, f_max = grid.shape
    if f_max < t_max:
        raise ValueError(
            "The original code slices grid[:, t, t:], so the future dimension "
            f"must be at least the current-time dimension; got T={t_max}, F={f_max}."
        )

    prior_q = prior_q.to(device=grid.device, dtype=torch.long).reshape(-1)
    if len(prior_q) != n:
        raise ValueError("`prior_q` must have one value per sample.")

    t_idx = torch.arange(t_max, device=grid.device, dtype=torch.long)
    u_idx = torch.arange(f_max, device=grid.device, dtype=torch.long)

    # Shape (1,T,F).  `relative == r` corresponds to grid[:, t, t+r].
    relative = u_idx.view(1, 1, f_max) - t_idx.view(1, t_max, 1)
    upper_mask = relative >= 0
    relative_weight = (relative + 1).clamp_min(0).to(grid.dtype)

    max_available = (f_max - t_idx).clamp_min(0).view(1, t_max)

    if sample_chunk_size is None:
        sample_chunk_size = n
    if sample_chunk_size <= 0:
        raise ValueError("`sample_chunk_size` must be positive or None.")

    output = torch.empty((n, t_max), dtype=grid.dtype, device=grid.device)

    for start in range(0, n, sample_chunk_size):
        end = min(start + sample_chunk_size, n)
        g = grid[start:end]
        q = prior_q[start:end]

        remaining = (q[:, None] - t_idx[None, :]).clamp_min(0)
        remaining = torch.minimum(remaining, max_available).to(torch.long)

        upper_grid = torch.where(
            upper_mask,
            g,
            torch.zeros((), dtype=grid.dtype, device=grid.device),
        )
        truncated_steps = torch.minimum(
            relative_weight,
            remaining.to(grid.dtype).unsqueeze(-1),
        )
        output[start:end] = (upper_grid * truncated_steps).sum(dim=-1)
        del upper_grid, truncated_steps

    return output


def _process_probabilities_and_required_mask(
    expected_remaining: torch.Tensor,
    prior_q: torch.Tensor,
    true_t: torch.Tensor,
    lam: float | torch.Tensor,
    *,
    pi_func=None,
    terminal_pi_min: float | None = None,
    terminal_floor_mode: str = "mixture",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build policy probabilities and the invariant eligible-step mask."""
    if expected_remaining.ndim != 2:
        raise ValueError(
            "`expected_remaining` must have shape (N,T); "
            f"got {tuple(expected_remaining.shape)}."
        )

    n, t_max = expected_remaining.shape
    device = expected_remaining.device
    dtype = expected_remaining.dtype

    prior_q = prior_q.to(device=device, dtype=torch.long).reshape(-1)
    true_t = true_t.to(device=device, dtype=torch.long).reshape(-1)
    if len(prior_q) != n or len(true_t) != n:
        raise ValueError("`prior_q` and `true_t` must have one value per sample.")

    lam_tensor = torch.as_tensor(lam, dtype=dtype, device=device)
    if torch.any(lam_tensor < 0):
        raise ValueError("`lam` must be nonnegative.")
    if pi_func is None:
        target_terminal_probability = torch.rsqrt(
            lam_tensor * expected_remaining + 1e-12
        ).clamp(max=1.0)
        # Equation (33) specifies the desired cumulative reach probability.
        # The continuation probability is the ratio to the probability of
        # reaching the previous step.  A cumulative minimum gives the same
        # recurrence without a Python time loop.
        cumulative_terminal_probability = torch.cummin(
            target_terminal_probability,
            dim=1,
        ).values
        previous_terminal_probability = torch.cat(
            [
                torch.ones((n, 1), dtype=dtype, device=device),
                cumulative_terminal_probability[:, :-1],
            ],
            dim=1,
        )
        pi = (
            cumulative_terminal_probability
            / previous_terminal_probability.clamp_min(torch.finfo(dtype).tiny)
        ).clamp(max=1.0)
    else:
        pi = pi_func(lam_tensor)
    pi = _enforce_terminal_probability_floor(
        pi,
        prior_q,
        terminal_pi_min,
        terminal_floor_mode,
    )

    time = torch.arange(t_max, device=device, dtype=torch.long).view(1, t_max)
    # Event times and prior horizons are interaction counts.  An endpoint of
    # one therefore activates exactly the zero-based coordinate t=0.
    required_mask = (
        (time < true_t[:, None])
        & (time < prior_q[:, None])
    )
    return pi, required_mask, prior_q, true_t


@torch.no_grad()
def expected_acquisition_cost(
    expected_remaining: torch.Tensor,
    prior_q: torch.Tensor,
    true_t: torch.Tensor,
    lam: float | torch.Tensor,
    *,
    pi_func=None,
    terminal_pi_min: float | None = None,
    terminal_floor_mode: str = "mixture",
) -> torch.Tensor:
    """Return only the deterministic expected cost as a scalar tensor.

    Shadow-price tuning calls the deterministic simulator dozens of times but
    discards its censoring vector and terminal propensities.  This path shares
    the exact policy construction with :func:`simulate_process_vectorized` and
    avoids computing those unused outputs.  Keeping the scalar on-device also
    lets callers decide when a host synchronization is actually required.
    """
    pi, required_mask, _, _ = _process_probabilities_and_required_mask(
        expected_remaining,
        prior_q,
        true_t,
        lam,
        pi_func=pi_func,
        terminal_pi_min=terminal_pi_min,
        terminal_floor_mode=terminal_floor_mode,
    )
    # The required mask is always an initial prefix.  Values after that prefix
    # are multiplied by zero, so replacing inactive probabilities with ones is
    # unnecessary for the expected-cost reduction.
    cumulative_reach_probability = pi.to(torch.float64).cumprod(dim=1)
    return (
        cumulative_reach_probability
        * required_mask.to(expected_remaining.dtype)
    ).sum()


@torch.no_grad()
def simulate_process_vectorized(
    expected_remaining: torch.Tensor,
    prior_q: torch.Tensor,
    true_t: torch.Tensor,
    lam: float | torch.Tensor,
    *,
    stochastic: bool = False,
    reach_t_max_is_success: bool = False,
    uniforms: torch.Tensor | None = None,
    pi_func=None,
    terminal_pi_min: float | None = None,
    terminal_floor_mode: str = "mixture",
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """
    Fully vectorized replacement for the old `for t_curr in range(T_max_curr)`.

    Returns
    -------
    sim_C
        Same censoring/count convention as the original implementation.
    terminal_reach_probability
        For every sample i, exactly

            prod_{t=0}^{min(true_t[i], prior_q[i])-1} pi_i(t)

        truncated to the stored time width.  This is the probability of
        acquiring the earlier of the event and prior-horizon interaction under
        the paper's one-based count convention.  It is deliberately independent
        of the realized Bernoulli path.
    total_cost
        Expected total cost in deterministic mode and realized total cost in
        stochastic mode.

    Notes
    -----
    This vectorization is exact for the current policy because `pi_i(t)` depends
    on `(grid, prior_q, t, lambda)` but not on earlier sampled decisions or on
    `current_cum_prob`.  If sequential steering based on cumulative probability
    is re-enabled, the time recurrence cannot be removed this way.
    """
    pi, required_mask, prior_q, true_t = (
        _process_probabilities_and_required_mask(
            expected_remaining,
            prior_q,
            true_t,
            lam,
            pi_func=pi_func,
            terminal_pi_min=terminal_pi_min,
            terminal_floor_mode=terminal_floor_mode,
        )
    )
    n, t_max = expected_remaining.shape
    device = expected_remaining.device
    dtype = expected_remaining.dtype

    pi_on_required_path = torch.where(
        required_mask, pi, torch.ones((), dtype=dtype, device=device)
    )

    # Correct event/prior reach propensity.  It must depend only on the policy
    # and latent endpoint, never on the sampled stopping path.
    terminal_log_probability = torch.where(
        required_mask,
        torch.log(pi.to(torch.float64).clamp_min(torch.finfo(torch.float64).tiny)),
        torch.zeros((), dtype=torch.float64, device=device),
    ).sum(dim=1)
    terminal_reach_probability = torch.exp(
        terminal_log_probability.clamp_min(-700.0)
    )

    if not stochastic:
        # Probability of paying for step t is the product through t.
        cumulative_reach_probability = pi_on_required_path.to(torch.float64).cumprod(dim=1)
        total_cost = (
            cumulative_reach_probability
            * required_mask.to(dtype)
        ).sum().item()
        sim_C = torch.zeros(n, dtype=torch.long, device=device)
        return sim_C, terminal_reach_probability, total_cost

    if uniforms is None:
        # Generate in time-major order.  This matches the logical order of the
        # legacy calls `torch.rand(N)` at each time.  It may advance the global
        # RNG farther than the legacy loop if every sampled path dies early.
        required_steps = required_mask.sum(dim=1)
        n_time_draws = int(required_steps.max().item()) if n else 0
        uniforms = torch.ones((n, t_max), dtype=dtype, device=device)
        if n_time_draws:
            uniforms[:, :n_time_draws] = torch.rand(
                (n_time_draws, n), dtype=dtype, device=device
            ).transpose(0, 1)
    else:
        uniforms = uniforms.to(device=device, dtype=dtype)
        if uniforms.shape != (n, t_max):
            raise ValueError(
                f"`uniforms` must have shape {(n, t_max)}; "
                f"got {tuple(uniforms.shape)}."
            )

    # Inactive coordinates are treated as automatic successes so only failures
    # while the trajectory is eligible to advance terminate its prefix.
    decision_success = (~required_mask) | (uniforms <= pi)
    failures_so_far = (~decision_success).to(torch.int32).cumsum(dim=1)
    prefix_survived = failures_so_far.eq(0)

    # `advanced[i,t]` is exactly the old per-time `keep[i]`.
    advanced = required_mask & prefix_survived
    number_advanced = advanced.sum(dim=1).to(torch.long)
    total_cost = float(number_advanced.sum().item())

    succeeded = (
        (number_advanced >= prior_q)
        | (number_advanced >= true_t)
    )
    if reach_t_max_is_success:
        succeeded = succeeded | (number_advanced == t_max)

    sim_C = torch.where(
        succeeded,
        prior_q,
        number_advanced,
    )

    return sim_C, terminal_reach_probability, total_cost
