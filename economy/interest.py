"""Small, behaviorally explicit interest-settlement primitives."""

import math


def annual_to_weekly_rate(annual_rate):
    annual_rate = max(0.0, float(annual_rate or 0.0))
    if annual_rate == 0.0:
        return 0.0
    return (1.0 + annual_rate) ** (1.0 / 52.0) - 1.0


def settle_interest(
    opening_principal,
    opening_arrears,
    cash_before_interest,
    operating_liquidity_floor,
    annual_rate,
    current_interest_due=None,
    opening_legacy_arrears=0.0,
):
    """Settle legacy arrears, revolving arrears, and current interest.

    ``opening_legacy_arrears`` is normally zero.  When Step 13.8F is active,
    it preserves the existing arrears-first cash priority after a snapshot
    term-out without putting that claim back into revolving utilization.
    """
    principal = max(0.0, float(opening_principal))
    arrears = max(0.0, float(opening_arrears))
    legacy_arrears = max(0.0, float(opening_legacy_arrears))
    cash = max(0.0, float(cash_before_interest))
    floor = max(0.0, float(operating_liquidity_floor))
    weekly_rate = annual_to_weekly_rate(annual_rate)
    current_due = (
        principal * weekly_rate
        if current_interest_due is None
        else max(0.0, float(current_interest_due))
    )
    obligation = legacy_arrears + arrears + current_due
    service_cash = max(0.0, cash - floor)
    paid = min(obligation, service_cash)
    paid_legacy = min(legacy_arrears, paid)
    paid_old = min(arrears, max(0.0, paid - paid_legacy))
    paid_current = min(
        current_due,
        max(0.0, paid - paid_legacy - paid_old),
    )
    unpaid_current = max(0.0, current_due - paid_current)
    closing_arrears = max(0.0, arrears - paid_old + unpaid_current)
    closing_legacy_arrears = max(0.0, legacy_arrears - paid_legacy)
    return {
        "weekly_rate": weekly_rate,
        "current_interest_due": current_due,
        "total_interest_obligation": obligation,
        "interest_service_cash": service_cash,
        "interest_paid": paid,
        "interest_paid_to_legacy_term_claim": paid_legacy,
        "interest_paid_to_opening_arrears": paid_old,
        "interest_paid_to_current_due": paid_current,
        "current_interest_unpaid": unpaid_current,
        "closing_interest_arrears": closing_arrears,
        "closing_legacy_arrears": closing_legacy_arrears,
        "cash_after_interest": cash - paid,
    }
