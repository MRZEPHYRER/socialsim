"""Canonical passive statistics helpers used by Analysis v2."""

import math


def finite(values):
    return [float(value) for value in values if isinstance(value, (int, float)) and math.isfinite(float(value))]


def safe_mean(values):
    values = finite(values)
    return sum(values) / len(values) if values else 0.0


def safe_median(values):
    values = sorted(finite(values))
    if not values:
        return 0.0
    middle = len(values) // 2
    return values[middle] if len(values) % 2 else (values[middle - 1] + values[middle]) / 2


def safe_percentile(values, percentile):
    values = sorted(finite(values))
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    position = (len(values) - 1) * percentile / 100.0
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def gini(values):
    values = sorted(value for value in finite(values) if value >= 0)
    if not values or sum(values) == 0:
        return 0.0
    total = sum(values)
    return sum((2 * index - len(values) - 1) * value for index, value in enumerate(values, 1)) / (len(values) * total)


def distribution_summary(values):
    values = finite(values)
    total = sum(sorted(values)[-max(1, int(len(values) * 0.1)):]) if values else 0.0
    return {
        "count": len(values),
        "mean": safe_mean(values),
        "median": safe_median(values),
        "p10": safe_percentile(values, 10),
        "p25": safe_percentile(values, 25),
        "p50": safe_percentile(values, 50),
        "p75": safe_percentile(values, 75),
        "p90": safe_percentile(values, 90),
        "gini": gini(values),
        "top_10_share": total / sum(values) if values and sum(values) else 0.0,
    }


def lorenz_curve(values):
    """Return population and value shares for a non-negative distribution."""
    ordered = sorted(value for value in finite(values) if value >= 0)
    if not ordered:
        return [0.0, 1.0], [0.0, 1.0]
    total = sum(ordered)
    population = [0.0] + [index / len(ordered) for index in range(1, len(ordered) + 1)]
    if total == 0:
        return population, population.copy()
    cumulative = [0.0]
    running = 0.0
    for value in ordered:
        running += value
        cumulative.append(running / total)
    return population, cumulative


def coefficient_of_variation(values):
    """Population standard deviation divided by the absolute mean."""
    values = finite(values)
    if not values:
        return math.nan
    mean = safe_mean(values)
    if abs(mean) <= 1e-15:
        return 0.0 if all(abs(value) <= 1e-15 for value in values) else math.nan
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return math.sqrt(variance) / abs(mean)


def normalized_shares(values):
    """Normalize non-negative observations into shares without inventing a denominator."""
    values = [max(0.0, value) for value in finite(values)]
    total = sum(values)
    if total <= 0:
        return [math.nan for _ in values]
    return [value / total for value in values]


def hhi(values, *, already_shares=False):
    """Herfindahl-Hirschman index on [0, 1] shares or non-negative levels."""
    shares = finite(values) if already_shares else normalized_shares(values)
    shares = [value for value in shares if math.isfinite(value) and value >= 0]
    if not shares:
        return math.nan
    total = sum(shares)
    if total <= 0:
        return math.nan
    normalized = [value / total for value in shares]
    return sum(value * value for value in normalized)


def extended_distribution_summary(values):
    """Cross-sectional summary used only when authoritative micro observations exist."""
    values = finite(values)
    ordered = sorted(values)
    non_negative = [value for value in ordered if value >= 0]
    total = sum(non_negative)
    top_count = max(1, math.ceil(len(non_negative) * 0.10)) if non_negative else 0
    bottom_count = math.floor(len(non_negative) * 0.50)
    p10 = safe_percentile(values, 10)
    p90 = safe_percentile(values, 90)
    return {
        "count": len(values),
        "mean": safe_mean(values),
        "median": safe_median(values),
        "p10": p10,
        "p25": safe_percentile(values, 25),
        "p75": safe_percentile(values, 75),
        "p90": p90,
        "p95": safe_percentile(values, 95),
        "gini": gini(values),
        "top_10_share": sum(non_negative[-top_count:]) / total if total and top_count else math.nan,
        "bottom_50_share": sum(non_negative[:bottom_count]) / total if total and bottom_count else 0.0,
        "p90_p10_ratio": p90 / p10 if p10 > 0 else math.nan,
        "positive_share": sum(value > 0 for value in values) / len(values) if values else math.nan,
        "zero_share": sum(abs(value) <= 1e-12 for value in values) / len(values) if values else math.nan,
        "negative_share": sum(value < 0 for value in values) / len(values) if values else math.nan,
    }
