import torch


@torch.no_grad()
def precompute_expected_remaining(
    grid: torch.Tensor,
    prior_q: torch.Tensor,
    *,
    sample_chunk_size: int | None = 256,
) -> torch.Tensor:
    """
    Vectorized replacement for the per-time `get_mpc_decision` belief calculation.

    This exactly reproduces, up to floating-point summation order, the current
    loop logic:

        belief = grid[:, t, t:]
        remaining = clamp(prior_q - t + 1, 0, belief.shape[1])
        expected =
            sum_{r <= remaining} belief[r] * (r + 1)
            + remaining * sum_{r > remaining} belief[r]

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

        remaining = (q[:, None] - t_idx[None, :] + 1).clamp_min(0)
        remaining = torch.minimum(remaining, max_available).to(torch.long)

        # In the legacy loop, relative positions 0,...,remaining are placed in
        # the weighted prefix.  If remaining equals the full slice length, the
        # gather endpoint is clipped to the final available future index.
        endpoint = (t_idx[None, :] + remaining).clamp_max(f_max - 1)
        gather_idx = endpoint.unsqueeze(-1)

        upper_grid = torch.where(
            upper_mask,
            g,
            torch.zeros((), dtype=grid.dtype, device=grid.device),
        )

        total_probability = upper_grid.sum(dim=-1)

        cumulative_probability = upper_grid.cumsum(dim=-1)
        prefix_probability = cumulative_probability.gather(
            dim=-1, index=gather_idx
        ).squeeze(-1)
        del cumulative_probability

        cumulative_weighted = (upper_grid * relative_weight).cumsum(dim=-1)
        prefix_weighted = cumulative_weighted.gather(
            dim=-1, index=gather_idx
        ).squeeze(-1)
        del cumulative_weighted, upper_grid

        output[start:end] = (
            prefix_weighted
            + remaining.to(grid.dtype)
            * (total_probability - prefix_probability)
        )

    return output


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
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """
    Fully vectorized replacement for the old `for t_curr in range(T_max_curr)`.

    Returns
    -------
    sim_C
        Same censoring/count convention as the original implementation.
    terminal_reach_probability
        For every sample i, exactly

            prod_{t=0}^{min(true_t[i], prior_q[i])} pi_i(t)

        truncated to the stored time width.  This is the probability of reaching
        the earlier of the event index and prior-horizon index under the code's
        zero-based indexing convention.  It is deliberately independent of the
        realized Bernoulli path.
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

    pi = torch.rsqrt(lam_tensor * expected_remaining + 1e-12).clamp(max=1.0)

    time = torch.arange(t_max, device=device, dtype=torch.long).view(1, t_max)

    # This is exactly the old:
    #   (~(true_t < t_curr)) & (t_curr <= prior_q)
    # i.e. t_curr <= min(true_t, prior_q).
    required_mask = (
        (time <= true_t[:, None])
        & (time <= prior_q[:, None])
    )

    pi_on_required_path = torch.where(
        required_mask, pi, torch.ones((), dtype=dtype, device=device)
    )

    # Correct event/prior reach propensity.  It must depend only on the policy
    # and latent endpoint, never on the sampled stopping path.
    terminal_reach_probability = pi_on_required_path.prod(dim=1)

    if not stochastic:
        # Probability of paying for step t is the product through t.
        cumulative_reach_probability = pi_on_required_path.cumprod(dim=1)
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
        (number_advanced > prior_q)
        | (number_advanced > true_t)
    )
    if reach_t_max_is_success:
        succeeded = succeeded | (number_advanced == t_max)

    sim_C = torch.where(
        succeeded,
        prior_q + 1,
        number_advanced,
    )

    return sim_C, terminal_reach_probability, total_cost
