import hashlib


def _short_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:10]


def _format_number(value: float) -> str:
    return format(float(value), ".12g")


def get_dapro_projection_experiment_name(
        dataset_name: str,
        dataset_setup: str,
        budget_per_sample: float,
        cal_size: int,
        tau_prior: float,
        m_upper_bound: float,
) -> str:
    return (
        f"dapro_projection_{dataset_name}"
        f"_setup-{_short_hash(dataset_setup)}"
        f"_budget-{_format_number(budget_per_sample)}"
        f"_cal-{int(cal_size)}"
        f"_tau-{_format_number(tau_prior)}"
        f"_max-{_format_number(m_upper_bound)}"
    )


def get_dapro_projection_metadata(
        dataset_name: str,
        dataset_setup: str,
        budget_per_sample: float,
        cal_size: int,
        tau_prior: float,
        m_upper_bound: float,
) -> dict:
    return {
        "experiment_type": "dapro_projection_evaluation",
        "dataset_name": dataset_name,
        "dataset_setup": dataset_setup,
        "budget_per_sample": float(budget_per_sample),
        "cal_size": int(cal_size),
        "tau_prior": float(tau_prior),
        "m_upper_bound": float(m_upper_bound),
    }
