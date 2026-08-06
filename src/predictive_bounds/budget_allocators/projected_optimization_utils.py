import numpy as np
import torch
from sklearn.isotonic import IsotonicRegression
from torch import nn
from scipy.optimize import minimize


def sample_c(C_probs, prior_quantile_est):
    C = torch.where(torch.rand(len(prior_quantile_est), device=C_probs.device) < C_probs, prior_quantile_est,
                    0).int().reshape(
        -1, 1)
    return C


class BudgetProjectedLayer(nn.Module):
    def __init__(self, T, target_budget):
        super().__init__()
        self.T = T
        self.target_budget = target_budget

    def forward(self, Y_raw, mask):
        """
        Finds a scalar delta such that if we shift Y_raw everywhere by delta,
        the UNWEIGHTED cumulative budget is satisfied exactly.
        """
        device = Y_raw.device

        # 1. Calculate Raw Cumulative Probabilities
        # Mask Y_raw so invalid steps don't mess up the cumsum
        Y_masked = Y_raw * mask.float()

        # P_cum_raw[i, t] = exp( y_0 + ... + y_t )
        log_P_cum_raw = torch.cumsum(Y_masked, dim=1)
        P_cum_raw = torch.exp(log_P_cum_raw) * mask.float()

        # 2. Formulate the Root Finding Problem
        # We want to find a shift 'delta' such that Y_new = Y_raw + delta
        # P_cum_new(t) = exp( Sum(y_k + delta) )
        #              = exp( Sum(y_k) + (t+1)*delta )
        #              = P_cum_raw(t) * (e^delta)^(t+1)

        # Let x = e^delta. We want to find x.
        # Target = Sum_{i,t} [ P_cum_raw(i,t) * x^(t+1) ]  <-- Unweighted sum

        # Pre-calculate coefficients C_t = Sum_over_i( P_cum_raw(i, t) )
        C = torch.sum(P_cum_raw, dim=0)  # Shape (T,)

        # 3. Newton's Method to find x
        # f(x) = Sum( C_t * x^(t+1) ) - Budget = 0
        x = torch.tensor(1.0, device=device)  # Start guess (delta=0)
        powers = torch.arange(1, self.T + 1, device=device, dtype=torch.float32)

        for _ in range(12):  # 10 iterations is plenty for high precision
            # f(x)
            x_pow = torch.pow(x, powers)
            f_val = torch.dot(C, x_pow) - self.target_budget

            # f'(x) = Sum( C_t * (t+1) * x^t )
            # Efficient calc: Sum( term * (t+1) ) / x
            deriv_terms = C * powers * x_pow
            f_deriv = torch.sum(deriv_terms) / (x + 1e-8)

            # Update
            diff = f_val / (f_deriv + 1e-8)
            x = x - diff
            x = torch.clamp(x, 1e-6, 5.0)  # Clamp for safety

            if torch.abs(f_val) < 1e-3 * self.target_budget:
                break

        # 4. Apply Shift
        # shift = log(x) -> Y_new = Y_raw + log(x)
        shift = torch.log(x)
        Y_projected = Y_raw + shift
        Y_projected.clamp_(max=0.)
        return Y_projected


def solve_pytorch_stable(S, lengths, target_budget_per_sample=16, lr=0.001, max_iter=200):
    # --- 1. SETUP ---
    if not isinstance(S, torch.Tensor):
        S = torch.tensor(S, dtype=torch.float32)
    if not isinstance(lengths, torch.Tensor):
        lengths = torch.tensor(lengths, dtype=torch.float32)

    device = S.device
    N, T = S.shape

    # Masks and Weights
    range_t = torch.arange(T, device=device).unsqueeze(0)
    mask = range_t < lengths.unsqueeze(1)

    # Weights for budget: [1, 2, 3, ..., T]
    weights = torch.arange(1, T + 1, device=device, dtype=torch.float32)
    TOTAL_BUDGET = N * target_budget_per_sample

    # --- 2. INITIALIZATION ---
    # We optimize Y = log(P).
    # Initialize close to 0 (P=1) so we start with a "cheap" objective
    # but put a small penalty at t=0 to give it room to move.
    Y = torch.zeros((N, T), device=device)
    Y[:, 0] = torch.log(1 / (((1 / target_budget_per_sample) * lengths.sqrt().mean()) * lengths.sqrt()))
    # Y[:, 0] = np.log(0.1)  # Start with P(t=0) approx 0.13
    Y[:, 1:] = np.log(1)  # Start with P(t=0) approx 0.13
    Y.clamp_(max=0.0)
    curr_budget = torch.sum(torch.exp(torch.cumsum(Y * mask.float(), dim=1)) * mask.float()).item()
    print(f"initial loss: {(1 / Y[:, 0].exp()).mean().item():2f} | initial budget: {curr_budget:2f}/{TOTAL_BUDGET:1f}")
    Y = torch.nn.Parameter(Y)

    # Use Adam. It handles the different scales of gradients better than SGD.
    optimizer = torch.optim.Adam([Y], lr=lr)

    iso_reg = IsotonicRegression(increasing=True, out_of_bounds='clip')

    # Pre-compute numpy indices for Isotonic Projection
    S_np = S.detach().cpu().numpy()
    mask_np = mask.detach().cpu().numpy()
    sort_indices = []

    for t in range(T):
        valid_idx = np.where(mask_np[:, t])[0]
        if len(valid_idx) > 0:
            s_col = S_np[valid_idx, t]
            idx = np.argsort(s_col)
            # Store (rows, sorted_rows)
            sort_indices.append((valid_idx, valid_idx[idx]))
        else:
            sort_indices.append(None)

    print(f"Starting Stable PyTorch Optimization...")
    projector = BudgetProjectedLayer(T, TOTAL_BUDGET)

    # --- 3. OPTIMIZATION LOOP ---
    for it in range(max_iter):
        optimizer.zero_grad()

        # A. Compute Log-Probability of the Sequence (R_i)
        # We assume independent probabilities for the sequence product:
        # log(Prod P) = Sum(log P)
        Y_valid = projector(Y, mask)
        # Mask Y so invalid steps are 0 (log(1)=0) in the sum
        Y_masked = Y_valid * mask.float()
        R = torch.sum(Y_masked, dim=1)  # Shape (N,)

        # B. STABLE OBJECTIVE: LogSumExp
        # We want to minimize Sum( exp(-R) )
        # Equivalent to minimizing Log( Sum( exp(-R) ) )
        # This is exactly torch.logsumexp(-R)
        objective_loss = torch.exp(-R).mean()
        # curr_budget = torch.sum(torch.exp(torch.cumsum(Y * mask.float(), dim=1)) * mask.float())
        # budget_loss = torch.relu(curr_budget - TOTAL_BUDGET)
        total_loss = objective_loss  # + 0.02*budget_loss

        # Backward
        total_loss.backward()
        optimizer.step()

        # --- D. Projections (Hard Constraints) ---
        with torch.no_grad():

            # 1. Monotonicity (Isotonic Regression)
            # Apply periodically (e.g., every 5 steps) to save CPU transfer time
            if it % 10 == 0:
                Y_data = Y.detach().cpu().numpy()
                did_change = False
                for t in range(T):
                    if sort_indices[t] is not None:
                        rows, sorted_rows = sort_indices[t]
                        y_col_sorted = Y_data[sorted_rows, t]
                        tensor_y_col_sorted = Y[sorted_rows, t]
                        if torch.any(tensor_y_col_sorted[:-1] > tensor_y_col_sorted[1:]):
                            # Violation found, fix it
                            y_new = iso_reg.fit_transform(np.arange(len(y_col_sorted)), y_col_sorted)
                            # Map back
                            Y_data[sorted_rows, t] = y_new
                            did_change = True

                if did_change:
                    Y.copy_(torch.from_numpy(Y_data).to(device))

            # 2. Upper Bound (Prob <= 1)
            Y.clamp_(max=0.0)

        # --- E. Logging ---
        if it % 200 == 0:
            # Calculate the REAL objective (Sum 1/P) for display
            # (Use standard sum for display, it might be large but useful to see)
            real_obj = torch.mean(torch.exp(-R)).item()
            # P_cum = torch.exp(torch.cumsum(Y_masked, dim=1)) * mask.float()
            used_budget = torch.sum(torch.exp(torch.cumsum(Y_masked, dim=1)) * mask.float()).item()
            print(
                f"Iter {it}: Log-Obj={total_loss.item():.4f} (Real={real_obj:.2e}) | Budget={used_budget:.1f}/{TOTAL_BUDGET}")

    # Final Return
    P_final = torch.exp(Y_valid).detach().cpu().numpy()
    P_final = P_final * mask_np
    print(P_final)
    return P_final


def project_to_test_ir(optimal_P, val_scores, test_scores, val_prior_q, t_val, T_max_curr, device):
    calibration_curves = []
    range_t = torch.arange(T_max_curr, device=device).unsqueeze(0)
    max_val = torch.minimum(val_prior_q, t_val)
    mask_val = range_t < max_val.unsqueeze(1)
    for t in range(T_max_curr):
        # Initialize a regressor for this specific time step
        # We enforce that the curve must be increasing (y_min <= y_max)
        iso_reg = IsotonicRegression(increasing=True, out_of_bounds='clip')
        # Fit the curve: Learn mapping from Score -> Optimal Probability
        if len(val_scores[mask_val[:, t], t]) == 0:
            iso_reg.fit(val_scores[:, t].detach().cpu().numpy(), np.ones_like(val_scores[:, t].detach().cpu().numpy()))
        else:
            iso_reg.fit(val_scores[mask_val[:, t], t].detach().cpu().numpy(),
                        optimal_P[mask_val[:, t].detach().cpu().numpy(), t])

        calibration_curves.append(iso_reg)

    # --- INFERENCE PHASE ---
    # For a new test sample with scores 's_test' (size 200)
    p_test = torch.zeros(len(test_scores), T_max_curr, device=device)

    for t in range(T_max_curr):
        # Instant lookup: O(1) complexity
        p_test[:, t] = torch.Tensor(calibration_curves[t].predict(test_scores[:, t].detach().cpu().numpy())).to(device)

    return p_test


def sigmoid(x):
    x = np.asarray(x, dtype=np.float64)
    result = np.empty_like(x)
    nonnegative = x >= 0
    result[nonnegative] = 1 / (1 + np.exp(-x[nonnegative]))
    exp_x = np.exp(x[~nonnegative])
    result[~nonnegative] = exp_x / (1 + exp_x)
    return result.item() if result.ndim == 0 else result


def project_to_test_platt(optimal_P, val_scores, test_scores, val_prior_q, t_val, T_max_curr, device,
                          epsilon=1e-4):
    if torch.is_tensor(optimal_P) and isinstance(optimal_P, torch.Tensor):
        optimal_P = optimal_P.to(device)
    else:
        optimal_P = torch.Tensor(optimal_P).to(device)
    p_test = torch.zeros(len(test_scores), T_max_curr, device=device)

    range_t = torch.arange(T_max_curr, device=device).unsqueeze(0)
    max_val = torch.minimum(val_prior_q, t_val)
    mask_val = range_t < max_val.unsqueeze(1)

    for t in range(T_max_curr):
        mask = mask_val[:, t]

        # Transfer data to CPU/NumPy
        x_train = val_scores[mask, t].detach().cpu().numpy()
        y_train = optimal_P[mask, t] if isinstance(optimal_P, torch.Tensor) else torch.tensor(optimal_P)[mask, t]
        y_train = y_train.detach().cpu().numpy()
        x_test = test_scores[:, t].detach().cpu().numpy()

        if len(x_train) == 0:
            p_test[:, t] = 1.0
            continue

        # Objective Function: MSE
        def objective(params):
            w, b = params
            preds = sigmoid(w * x_train + b)
            return np.mean((preds - y_train) ** 2)

        # Bounds: w must be >= 0 (monotonicity), b is unconstrained
        bounds = [(0, None), (None, None)]

        # Initial guess: w=1.0, b=0.0
        init_params = [1.0, 0.0]

        # SciPy L-BFGS-B exact optimization
        res = minimize(objective, np.array(init_params), method='L-BFGS-B', bounds=bounds)
        w_opt, b_opt = res.x

        # Predict on test data
        p_pred = sigmoid(w_opt * x_test + b_opt)

        # Clip to prevent the singularity (1/0) in your non-convex loss
        p_pred = np.clip(p_pred, epsilon, 1.0 - epsilon)

        # Transfer back to GPU
        p_test[:, t] = torch.tensor(p_pred, dtype=torch.float32, device=device)

    return p_test


def project_cumulative_probabilities_to_test_platt(
        optimal_P,
        val_scores,
        test_scores,
        val_prior_q,
        t_val,
        T_max_curr,
        device,
        epsilon=1e-6,
):
    """Project oracle cumulative reach probabilities instead of conditionals.

    Independent errors in projected conditional probabilities multiply across
    time and can radically change the terminal propensity.  This alternative
    first converts the Phase-I optimum to cumulative reach probabilities,
    learns a monotone score map for that directly relevant quantity at every
    time, and finally enforces temporal non-increase.  The returned tensor is
    cumulative; it must be converted back to conditional probabilities after
    deployment-budget correction.
    """
    optimal = torch.as_tensor(
        optimal_P,
        dtype=torch.float64,
        device=device,
    )
    scores = test_scores.to(device)
    cumulative_predictions = torch.ones(
        (len(scores), T_max_curr),
        dtype=torch.float64,
        device=device,
    )
    time = torch.arange(T_max_curr, device=device).unsqueeze(0)
    active_lengths = torch.minimum(val_prior_q, t_val).to(torch.long)
    active = time < active_lengths.unsqueeze(1)
    optimal_path = torch.where(
        active,
        optimal.clamp(min=epsilon, max=1.0),
        torch.ones((), dtype=torch.float64, device=device),
    )
    optimal_cumulative = optimal_path.cumprod(dim=1)

    for step in range(T_max_curr):
        mask = active[:, step]
        if not mask.any():
            # No Phase-I trajectory identifies this step.  Keeping cumulative
            # reach unchanged corresponds to automatic continuation there.
            continue

        x_train = val_scores[mask, step].detach().cpu().numpy()
        y_train = (
            optimal_cumulative[mask, step]
            .detach()
            .cpu()
            .numpy()
        )
        x_test = scores[:, step].detach().cpu().numpy()

        if np.ptp(y_train) <= 1e-12:
            predicted = np.full_like(
                x_test,
                y_train[0],
                dtype=np.float64,
            )
        else:
            def objective(params):
                slope, intercept = params
                prediction = sigmoid(slope * x_train + intercept)
                return np.mean((prediction - y_train) ** 2)

            initial_probability = float(np.clip(
                np.mean(y_train),
                epsilon,
                1 - epsilon,
            ))
            initial_intercept = np.log(
                initial_probability / (1 - initial_probability)
            )
            result = minimize(
                objective,
                np.array([1.0, initial_intercept]),
                method='L-BFGS-B',
                bounds=[(0, None), (None, None)],
            )
            if not result.success or not np.all(np.isfinite(result.x)):
                raise RuntimeError(
                    "Cumulative Platt projection failed at step "
                    f"{step}: {result.message}"
                )
            slope, intercept = result.x
            predicted = sigmoid(slope * x_test + intercept)

        cumulative_predictions[:, step] = torch.as_tensor(
            np.clip(predicted, epsilon, 1.0),
            dtype=torch.float64,
            device=device,
        )

    # Reach probabilities must be non-increasing along every trajectory.
    return torch.cummin(cumulative_predictions, dim=1).values


def project_to_test_spline(optimal_P, val_scores, test_scores, val_prior_q, t_val, T_max_curr, device,
                           num_knots=7, l2_lambda=0.01, epsilon=1e-4):
    if torch.is_tensor(optimal_P) and isinstance(optimal_P, torch.Tensor):
        optimal_P = optimal_P.to(device)
    else:
        optimal_P = torch.Tensor(optimal_P).to(device)
    p_test = torch.zeros(len(test_scores), T_max_curr, device=device)

    range_t = torch.arange(T_max_curr, device=device).unsqueeze(0)
    max_val = torch.minimum(val_prior_q, t_val)
    mask_val = range_t < max_val.unsqueeze(1)

    for t in range(T_max_curr):
        mask = mask_val[:, t]

        # Transfer data to CPU/NumPy
        x_train = val_scores[mask, t].detach().cpu().numpy()
        y_train = optimal_P[mask, t] if isinstance(optimal_P, torch.Tensor) else torch.tensor(optimal_P)[mask, t]
        y_train = y_train.detach().cpu().numpy()
        x_test = test_scores[:, t].detach().cpu().numpy()

        if len(x_train) == 0:
            p_test[:, t] = 1.0
            continue

        # Define fixed knots for stability
        x_min, x_max = x_train.min(), x_train.max()
        if x_min == x_max:
            x_max = x_min + 1e-5
        knots = np.linspace(x_min, x_max, num_knots)

        # Precompute basis matrices: max(0, X - knot)
        # shape: (N_samples, num_knots)
        basis_train = np.maximum(0, x_train[:, None] - knots[None, :])
        basis_test = np.maximum(0, x_test[:, None] - knots[None, :])

        # Objective Function: MSE + L2 Regularization (Crucial for bounding beta)
        def objective(params):
            w = params[:-1]  # First 'num_knots' params are weights
            b = params[-1]  # Last param is the bias

            # Spline output mapped through sigmoid
            spline_out = basis_train @ w + b
            preds = sigmoid(spline_out)

            mse = np.mean((preds - y_train) ** 2)
            l2_penalty = l2_lambda * np.sum(w ** 2)

            return mse + l2_penalty

        # Bounds: weights w must be >= 0 (to guarantee monotonicity)
        # Bias b is free (None, None)
        bounds = [(0, None)] * num_knots + [(None, None)]

        # Initialize weights to 1/num_knots, bias to 0
        init_params = np.ones(num_knots + 1)
        init_params[:-1] = 1.0 / num_knots
        init_params[-1] = 0.0

        # SciPy optimization
        res = minimize(objective, init_params, method='L-BFGS-B', bounds=bounds)
        w_opt = res.x[:-1]
        b_opt = res.x[-1]

        # Predict on test data
        spline_out_test = basis_test @ w_opt + b_opt
        p_pred = sigmoid(spline_out_test)

        # Clip to prevent the singularity (1/0)
        p_pred = np.clip(p_pred, epsilon, 1.0 - epsilon)

        # Transfer back to GPU
        p_test[:, t] = torch.tensor(p_pred, dtype=torch.float32, device=device)

    return p_test


def project_to_test_beta(optimal_P, val_scores, test_scores, val_prior_q, t_val, T_max_curr, device, epsilon=1e-4):
    if torch.is_tensor(optimal_P) and isinstance(optimal_P, torch.Tensor):
        optimal_P = optimal_P.to(device)
    else:
        optimal_P = torch.Tensor(optimal_P).to(device)
    p_test = torch.zeros(len(test_scores), T_max_curr, device=device)

    range_t = torch.arange(T_max_curr, device=device).unsqueeze(0)
    max_val = torch.minimum(val_prior_q, t_val)
    mask_val = range_t < max_val.unsqueeze(1)

    for t in range(T_max_curr):
        mask = mask_val[:, t]

        # Transfer data to CPU/NumPy
        x_train = val_scores[mask, t].detach().cpu().numpy()
        y_train = optimal_P[mask, t] if isinstance(optimal_P, torch.Tensor) else torch.tensor(optimal_P)[mask, t]
        y_train = y_train.detach().cpu().numpy()
        x_test = test_scores[:, t].detach().cpu().numpy()

        if len(x_train) == 0:
            p_test[:, t] = 1.0
            continue

        # 1. Clip inputs to strictly prevent log(0) crashes
        x_train_safe = np.clip(x_train, epsilon, 1.0 - epsilon)
        x_test_safe = np.clip(x_test, epsilon, 1.0 - epsilon)

        # 2. Precompute the log terms so the optimizer doesn't recalculate them every step
        log_x_train = np.log(x_train_safe)
        log_1m_x_train = np.log(1.0 - x_train_safe)

        log_x_test = np.log(x_test_safe)
        log_1m_x_test = np.log(1.0 - x_test_safe)

        # 3. Objective Function: MSE
        def objective(params):
            c, d, b = params
            # Beta calibration formula: c*ln(x) - d*ln(1-x) + b
            logits = c * log_x_train - d * log_1m_x_train + b
            preds = sigmoid(logits)
            return np.mean((preds - y_train) ** 2)

        # 4. Bounds: c >= 0 and d >= 0 to strictly guarantee monotonicity. b is free.
        bounds = [(0, None), (0, None), (None, None)]

        # Initial guess: c=1.0, d=1.0, b=0.0 (Identity mapping on the log-odds scale)
        init_params = [1.0, 1.0, 0.0]

        # 5. SciPy L-BFGS-B exact optimization
        res = minimize(objective, np.array(init_params), method='L-BFGS-B', bounds=bounds)
        c_opt, d_opt, b_opt = res.x

        # 6. Predict on test data
        logits_test = c_opt * log_x_test - d_opt * log_1m_x_test + b_opt
        p_pred = sigmoid(logits_test)

        # 7. Clip the output to prevent the 1/0 singularity in your downstream non-convex loss
        p_pred = np.clip(p_pred, epsilon, 1.0 - epsilon)

        # 8. Transfer back to GPU
        p_test[:, t] = torch.tensor(p_pred, dtype=torch.float32, device=device)

    return p_test


def expected_acquisition_cost(
        continuation_probabilities: torch.Tensor,
        active_lengths: torch.Tensor,
) -> float:
    """Return mean expected interactions under strict count-scale lengths."""
    probabilities = continuation_probabilities.to(torch.float64)
    lengths = active_lengths.to(
        device=probabilities.device,
        dtype=torch.long,
    ).reshape(-1)
    if probabilities.ndim != 2 or len(lengths) != len(probabilities):
        raise ValueError("Probability rows and active lengths must agree.")
    time = torch.arange(
        probabilities.shape[1],
        device=probabilities.device,
    ).unsqueeze(0)
    active = time < lengths.unsqueeze(1)
    path_probabilities = torch.where(
        active,
        probabilities,
        torch.ones((), dtype=torch.float64, device=probabilities.device),
    )
    cumulative = path_probabilities.cumprod(dim=1)
    return (cumulative * active.to(torch.float64)).sum(dim=1).mean().item()


def enforce_terminal_probability_floor(
        continuation_probabilities: torch.Tensor,
        prior_q: torch.Tensor,
        terminal_pi_min: float | None,
) -> torch.Tensor:
    """Mix the policy with an always-continue exploration policy.

    With mixture mass ``epsilon=terminal_pi_min``, the cumulative probability
    of reaching step ``t`` is

        epsilon + (1 - epsilon) * prod_{s<=t} p_s.

    This guarantees every event/prior terminal propensity is at least epsilon
    without a numerically fragile product clamp.  The returned conditional
    probabilities induce exactly those cumulative reach probabilities.
    """
    probabilities = continuation_probabilities.to(torch.float64)
    if terminal_pi_min is None:
        return probabilities
    if not 0 < terminal_pi_min <= 1:
        raise ValueError("`terminal_pi_min` must lie in (0, 1].")
    q = prior_q.to(
        device=probabilities.device,
        dtype=torch.long,
    ).reshape(-1).clamp(min=0, max=probabilities.shape[1])
    if len(q) != len(probabilities):
        raise ValueError("`prior_q` must have one value per probability row.")

    time = torch.arange(
        probabilities.shape[1],
        device=probabilities.device,
    ).unsqueeze(0)
    active = time < q.unsqueeze(1)
    raw_path = torch.where(
        active,
        probabilities.clamp(
            min=torch.finfo(torch.float64).tiny,
            max=1.0,
        ),
        torch.ones((), dtype=torch.float64, device=probabilities.device),
    )
    raw_cumulative = raw_path.cumprod(dim=1)
    mixed_cumulative = (
        terminal_pi_min
        + (1 - terminal_pi_min) * raw_cumulative
    )
    previous_mixed = torch.cat(
        [
            torch.ones(
                (len(probabilities), 1),
                dtype=torch.float64,
                device=probabilities.device,
            ),
            mixed_cumulative[:, :-1],
        ],
        dim=1,
    )
    mixed_conditionals = (
        mixed_cumulative / previous_mixed.clamp_min(
            torch.finfo(torch.float64).tiny
        )
    ).clamp(max=1.0)
    return torch.where(active, mixed_conditionals, probabilities)


def correct_projected_probabilities_to_budget(
        validation_probabilities: torch.Tensor,
        deployment_probabilities: torch.Tensor,
        validation_active_lengths: torch.Tensor,
        validation_prior_q: torch.Tensor,
        deployment_prior_q: torch.Tensor,
        target_budget_per_sample: float,
        terminal_pi_min: float | None = None,
        tolerance: float = 1e-7,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Fit a shared logit intercept so the projected map meets its budget.

    The intercept is learned only from Phase-I rows and then applied unchanged
    to Phase II. Any terminal-propensity floor is enforced inside each Phase-I
    budget evaluation, so the returned Phase-I policy meets both its expected
    budget and positivity constraints. Phase-II expected cost is a separate
    generalization diagnostic, not a guarantee of this correction.
    """
    validation_structural_one = validation_probabilities.to(torch.float64) == 1
    deployment_structural_one = deployment_probabilities.to(torch.float64) == 1
    validation_raw = validation_probabilities.to(torch.float64).clamp(
        1e-12, 1 - 1e-12
    )
    deployment_raw = deployment_probabilities.to(torch.float64).clamp(
        1e-12, 1 - 1e-12
    )
    validation_logits = torch.logit(validation_raw)
    deployment_logits = torch.logit(deployment_raw)

    def shifted(logits, shift, q, structural_one):
        probabilities = torch.sigmoid(logits + shift)
        probabilities = torch.where(
            structural_one,
            torch.ones_like(probabilities),
            probabilities,
        )
        return enforce_terminal_probability_floor(
            probabilities,
            q,
            terminal_pi_min,
        )

    def validation_cost(shift):
        candidate = shifted(
            validation_logits,
            shift,
            validation_prior_q,
            validation_structural_one,
        )
        return expected_acquisition_cost(
            candidate,
            validation_active_lengths,
        )

    raw_base_cost = expected_acquisition_cost(
        validation_probabilities,
        validation_active_lengths,
    )
    pre_intercept_mixed_cost = validation_cost(0.0)
    # The projection clips probabilities to values as small as 1e-4, while
    # other projection families can get closer to 0 or 1. Wide finite bounds
    # therefore make the two endpoints genuine numerical min/max policies.
    low, high = -80.0, 80.0
    low_cost = validation_cost(low)
    high_cost = validation_cost(high)
    if low_cost > target_budget_per_sample + tolerance:
        raise ValueError(
            "Projected policy budget is infeasible under the probability "
            f"floor: target={target_budget_per_sample}, minimum={low_cost}."
        )
    if high_cost <= target_budget_per_sample + tolerance:
        intercept = high
        achieved = high_cost
        boundary = "maximum"
    else:
        for _ in range(80):
            mid = (low + high) / 2
            mid_cost = validation_cost(mid)
            if mid_cost <= target_budget_per_sample:
                low, low_cost = mid, mid_cost
            else:
                high, high_cost = mid, mid_cost
            if high - low < 1e-10:
                break
        intercept = low
        achieved = low_cost
        boundary = None

    validation_corrected = shifted(
        validation_logits,
        intercept,
        validation_prior_q,
        validation_structural_one,
    )
    deployment_corrected = shifted(
        deployment_logits,
        intercept,
        deployment_prior_q,
        deployment_structural_one,
    )
    return validation_corrected, deployment_corrected, {
        "projection_raw_base_phase1_expected_cost": raw_base_cost,
        "projection_pre_intercept_mixed_phase1_expected_cost": (
            pre_intercept_mixed_cost
        ),
        # Backward-compatible key, now explicitly defined as the unmixed base
        # projection cost.
        "projection_raw_phase1_expected_cost": raw_base_cost,
        "projection_budget_logit_shift": intercept,
        "projection_corrected_phase1_expected_cost": achieved,
        "projection_budget_boundary": boundary,
    }


def _continuation_from_cumulative(
        cumulative_probabilities: torch.Tensor,
        prior_q: torch.Tensor,
) -> torch.Tensor:
    """Convert a valid cumulative-reach schedule to conditional probabilities."""
    cumulative = cumulative_probabilities.to(torch.float64).clamp(
        min=torch.finfo(torch.float64).tiny,
        max=1.0,
    )
    cumulative = torch.cummin(cumulative, dim=1).values
    q = prior_q.to(
        device=cumulative.device,
        dtype=torch.long,
    ).reshape(-1).clamp(min=0, max=cumulative.shape[1])
    if len(q) != len(cumulative):
        raise ValueError("`prior_q` must have one value per cumulative row.")
    time = torch.arange(
        cumulative.shape[1],
        device=cumulative.device,
    ).unsqueeze(0)
    active = time < q.unsqueeze(1)
    active_cumulative = torch.where(
        active,
        cumulative,
        torch.ones((), dtype=torch.float64, device=cumulative.device),
    )
    previous = torch.cat(
        [
            torch.ones(
                (len(cumulative), 1),
                dtype=torch.float64,
                device=cumulative.device,
            ),
            active_cumulative[:, :-1],
        ],
        dim=1,
    )
    conditional = (
        active_cumulative
        / previous.clamp_min(torch.finfo(torch.float64).tiny)
    ).clamp(max=1.0)
    return torch.where(
        active,
        conditional,
        torch.ones((), dtype=torch.float64, device=cumulative.device),
    )


def correct_projected_cumulative_probabilities_to_budget(
        validation_cumulative: torch.Tensor,
        deployment_cumulative: torch.Tensor,
        validation_active_lengths: torch.Tensor,
        validation_prior_q: torch.Tensor,
        deployment_prior_q: torch.Tensor,
        target_budget_per_sample: float,
        terminal_pi_min: float | None = None,
        tolerance: float = 1e-7,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Budget-correct cumulative reach probabilities with one shared shift.

    A monotone logit transform preserves both score ordering and temporal
    non-increase.  Fitting the shift on Phase I therefore changes the exact
    terminal quantity optimized by DAPRO without compounding a separate shift
    at every continuation step.
    """
    validation_raw = validation_cumulative.to(torch.float64).clamp(
        1e-12,
        1 - 1e-12,
    )
    deployment_raw = deployment_cumulative.to(torch.float64).clamp(
        1e-12,
        1 - 1e-12,
    )
    if validation_raw.ndim != 2 or deployment_raw.ndim != 2:
        raise ValueError("Cumulative probability arrays must be two-dimensional.")
    if validation_raw.shape[1] != deployment_raw.shape[1]:
        raise ValueError("Validation and deployment widths must agree.")

    validation_lengths = validation_active_lengths.to(
        device=validation_raw.device,
        dtype=torch.long,
    ).reshape(-1)
    if len(validation_lengths) != len(validation_raw):
        raise ValueError(
            "`validation_active_lengths` must have one value per validation row."
        )
    time = torch.arange(
        validation_raw.shape[1],
        device=validation_raw.device,
    ).unsqueeze(0)
    validation_active = time < validation_lengths.unsqueeze(1)
    structural_columns = ~validation_active.any(dim=0)

    validation_logits = torch.logit(validation_raw)
    deployment_logits = torch.logit(deployment_raw)

    def shifted_cumulative(logits, shift, q):
        shifted = torch.sigmoid(logits + shift)
        shifted[:, structural_columns] = 1.0
        shifted = torch.cummin(shifted, dim=1).values
        if terminal_pi_min is not None:
            if not 0 < terminal_pi_min <= 1:
                raise ValueError("`terminal_pi_min` must lie in (0, 1].")
            shifted = terminal_pi_min + (1 - terminal_pi_min) * shifted
        q_tensor = q.to(
            device=shifted.device,
            dtype=torch.long,
        ).reshape(-1).clamp(min=0, max=shifted.shape[1])
        if len(q_tensor) != len(shifted):
            raise ValueError("`prior_q` must have one value per cumulative row.")
        active = (
            torch.arange(shifted.shape[1], device=shifted.device).unsqueeze(0)
            < q_tensor.unsqueeze(1)
        )
        return torch.where(
            active,
            shifted,
            torch.ones((), dtype=torch.float64, device=shifted.device),
        )

    def direct_cost(cumulative, lengths):
        active = (
            torch.arange(cumulative.shape[1], device=cumulative.device)
            .unsqueeze(0)
            < lengths.unsqueeze(1)
        )
        return (
            cumulative * active.to(torch.float64)
        ).sum(dim=1).mean().item()

    raw_validation = torch.cummin(validation_raw, dim=1).values
    raw_base_cost = direct_cost(raw_validation, validation_lengths)

    def validation_cost(shift):
        return direct_cost(
            shifted_cumulative(
                validation_logits,
                shift,
                validation_prior_q,
            ),
            validation_lengths,
        )

    pre_intercept_mixed_cost = validation_cost(0.0)
    low, high = -80.0, 80.0
    low_cost = validation_cost(low)
    high_cost = validation_cost(high)
    if low_cost > target_budget_per_sample + tolerance:
        raise ValueError(
            "Cumulative projected policy budget is infeasible under the "
            f"probability floor: target={target_budget_per_sample}, "
            f"minimum={low_cost}."
        )
    if high_cost <= target_budget_per_sample + tolerance:
        intercept = high
        achieved = high_cost
        boundary = "maximum"
    else:
        for _ in range(80):
            mid = (low + high) / 2
            mid_cost = validation_cost(mid)
            if mid_cost <= target_budget_per_sample:
                low, low_cost = mid, mid_cost
            else:
                high, high_cost = mid, mid_cost
            if high - low < 1e-10:
                break
        intercept = low
        achieved = low_cost
        boundary = None

    validation_corrected_cumulative = shifted_cumulative(
        validation_logits,
        intercept,
        validation_prior_q,
    )
    deployment_corrected_cumulative = shifted_cumulative(
        deployment_logits,
        intercept,
        deployment_prior_q,
    )
    validation_corrected = _continuation_from_cumulative(
        validation_corrected_cumulative,
        validation_prior_q,
    )
    deployment_corrected = _continuation_from_cumulative(
        deployment_corrected_cumulative,
        deployment_prior_q,
    )
    return validation_corrected, deployment_corrected, {
        "projection_space": "cumulative_probability",
        "projection_raw_base_phase1_expected_cost": raw_base_cost,
        "projection_pre_intercept_mixed_phase1_expected_cost": (
            pre_intercept_mixed_cost
        ),
        "projection_raw_phase1_expected_cost": raw_base_cost,
        "projection_budget_logit_shift": intercept,
        "projection_corrected_phase1_expected_cost": achieved,
        "projection_budget_boundary": boundary,
    }


def adaptive_budget_allocation(
        continuation_probabilities,
        prior_q,
        event_times,
        T_max_curr,
        device,
        reach_t_max_is_success=False,
        uniforms=None,
):
    sim_N = prior_q.shape[0]
    if uniforms is not None:
        uniforms = uniforms.to(
            device=device,
            dtype=continuation_probabilities.dtype,
        )
        if uniforms.shape != (sim_N, T_max_curr):
            raise ValueError(
                "`uniforms` must have shape "
                f"{(sim_N, T_max_curr)}; got {tuple(uniforms.shape)}."
            )
    sim_C = torch.zeros(sim_N, dtype=torch.long, device=device)
    sim_active = torch.ones(sim_N, dtype=torch.bool, device=device)
    sim_cum_prob = torch.ones(sim_N, dtype=torch.float32, device=device)
    total_cost = 0.0
    for t_curr in range(T_max_curr):
        # Event times and prior horizons are one-based interaction counts.
        sim_active = sim_active & (event_times > t_curr) & (prior_q > t_curr)

        if not sim_active.any():
            break

        pi = continuation_probabilities[:, t_curr]

        # Random sampling for Test
        rand = (
            torch.rand(
                sim_N,
                device=device,
            )
            if uniforms is None
            else uniforms[:, t_curr]
        )
        # Exact Bernoulli(pi), including the pi == 0 boundary for supplied
        # common random numbers in [0, 1).
        keep = (rand < pi) & sim_active

        # Count actual advancements (transitions)
        step_cost = keep.sum().item()
        total_cost += step_cost

        # Update state
        sim_C[keep] += 1
        sim_cum_prob = torch.where(keep, sim_cum_prob * pi, sim_cum_prob)
        sim_active = sim_active & keep

    succeeded = (sim_C >= prior_q) | (sim_C >= event_times)
    if reach_t_max_is_success:
        succeeded = succeeded | (sim_C == T_max_curr)
    sim_C = torch.where(succeeded, prior_q, sim_C)

    range_t = torch.arange(T_max_curr, device=device).unsqueeze(0)
    max_test = torch.minimum(prior_q, event_times)
    mask = range_t < max_test.unsqueeze(1)
    log_probabilities = torch.where(
        mask,
        torch.log(
            continuation_probabilities.to(torch.float64).clamp_min(
                torch.finfo(torch.float64).tiny
            )
        ),
        torch.zeros((), dtype=torch.float64, device=device),
    )
    C_probs = torch.exp(log_probabilities.sum(dim=-1).clamp_min(-700.0))

    return sim_C, total_cost, C_probs


def construct_final_result(N, val_idxs, val_prior_q, test_idxs, test_prior_q, test_C, test_C_probs, device):
    prior_q = torch.empty(N, device=device)
    prior_q[val_idxs] = val_prior_q
    prior_q[test_idxs] = test_prior_q
    val_C = val_prior_q
    # For Validation set, we don't care about C_probs (set to 1.0 or dummy)
    val_size = len(val_prior_q)
    val_C_probs = torch.ones(val_size, device=device)

    # Concatenate
    final_C = torch.empty(N, device=device, dtype=torch.long)
    final_C[val_idxs] = val_C.to(torch.long)
    final_C[test_idxs] = test_C.to(torch.long)

    # final_C[final_C > t] = torch.max(prior_q[final_C > t], final_C[final_C > t])
    final_C_probs = torch.empty(N, device=device, dtype=test_C_probs.dtype)
    final_C_probs[val_idxs] = val_C_probs.to(final_C_probs.dtype)
    final_C_probs[test_idxs] = test_C_probs.to(final_C_probs.dtype)

    return final_C, final_C_probs


def split_to_two_sets(conditional_grid, prior_q, censored_event_time, scores, budget_per_sample, val_size=100,
                      reach_t_max_is_success=False):
    N = len(conditional_grid)
    total_budget = budget_per_sample * N
    perm = np.random.permutation(N)
    val_idxs = perm[:val_size]
    test_idxs = perm[val_size:]
    # --- Data Splitting ---
    # Validation Set: Used to learn the optimal policy parameters (lambda)
    val_grid = conditional_grid[val_idxs]
    val_prior_q = prior_q[val_idxs]
    t_val = censored_event_time[val_idxs]
    val_budget_used = torch.minimum(t_val, val_prior_q).sum().item()
    if total_budget < val_budget_used:
        raise ValueError("Total budget is too small")

    test_grid = conditional_grid[test_idxs]

    test_prior_q = prior_q[test_idxs]
    t_test = censored_event_time[test_idxs]

    target_budget_avg = (budget_per_sample * N - val_budget_used) / (N - val_size)

    val_scores = scores[val_idxs]
    test_scores = scores[test_idxs]
    val_max_steps = torch.minimum(val_prior_q, t_val)

    # if reach_t_max_is_success:
    #     val_max_steps[val_max_steps == conditional_grid.shape[1]] = conditional_grid.shape[1]+1

    return val_idxs, test_idxs, val_grid, val_prior_q, t_val, val_scores, test_grid, test_prior_q, t_test, test_scores, val_max_steps, target_budget_avg, val_budget_used
