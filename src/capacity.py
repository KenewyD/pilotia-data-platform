import math

def simulate_capacity(agents_fte, absence_rate, productivity_per_fte_day, workdays, expected_volume):
    available_fte = agents_fte * (1 - absence_rate)
    capacity = available_fte * productivity_per_fte_day * workdays
    gap = capacity - expected_volume
    required_fte = expected_volume / max(1e-9, productivity_per_fte_day * workdays)
    additional_fte = max(0.0, required_fte - available_fte)
    utilization = expected_volume / max(capacity, 1e-9)
    risk = min(1.0, max(0.0, (utilization - 0.80) / 0.35))
    recommendation = (
        "Capacité suffisante pour le scénario simulé."
        if gap >= 0
        else f"Prévoir environ {math.ceil(additional_fte)} ETP supplémentaires sur la période."
    )
    return {
        "available_fte": available_fte, "capacity": capacity, "gap": gap,
        "required_fte": required_fte, "additional_fte": additional_fte,
        "utilization": utilization, "risk": risk, "recommendation": recommendation,
    }
