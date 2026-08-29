"""Deterministic, checkpoint-safe review-phase helpers."""


_PROCESS_CODES = {
    "production": 0,
    "investment": 3,
}


def deterministic_review_phase(firm_id, cadence, process):
    """Return a stable phase without Python hash or runtime RNG draws."""
    interval = max(1, int(cadence))
    code = _PROCESS_CODES.get(str(process), 0)
    return (int(firm_id) * 7 + code) % interval


def review_due(week, cadence, phase):
    interval = max(1, int(cadence))
    return (int(week) - int(phase)) % interval == 0
