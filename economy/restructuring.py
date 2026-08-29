"""Passive state helpers for the Step 13.8B restructuring experiment."""

from __future__ import annotations


def ensure_restructuring_state(firm):
    """Backfill only state needed for one relief attempt per contract Default."""
    defaults = {
        "restructuring_used_this_default": False,
        "restructuring_active": False,
        "restructuring_start_this_week": False,
        "restructuring_weeks_used": 0,
        "restructuring_default_weeks": 0,
        "weekly_interest_relief_amount": 0.0,
        "cumulative_interest_relief_amount": 0.0,
        "gross_current_interest_due": 0.0,
        "net_current_interest_due": 0.0,
        "interest_relief_amount": 0.0,
        # Step 13.8F: one snapshot term-out attempt per contractual Default.
        # The claim remains payable and is never merged back automatically.
        "termout_used_this_default": False,
        "termout_start_this_week": False,
        "termout_snapshot_amount": 0.0,
        "legacy_arrears_term_claim": 0.0,
        "opening_legacy_arrears_term_claim": 0.0,
        "closing_legacy_arrears_term_claim": 0.0,
        "legacy_term_claim_payment": 0.0,
        "post_termout_arrears_payment": 0.0,
        "termout_pre_snapshot_arrears": 0.0,
        "termout_claim_reclassification_gap": 0.0,
    }
    for name, value in defaults.items():
        if not hasattr(firm, name):
            setattr(firm, name, value)
    return firm


def begin_restructuring_week(
    firms,
    enabled,
    eligibility_weeks,
    max_duration_weeks,
    termout_enabled=False,
):
    """Activate an eligible relief contract before this week's settlement.

    The counter is finalized at the end of the preceding week.  Thus an event
    in week ``t`` counts weeks ``t`` through ``t+25`` and can first start
    relief in week ``t+26``.
    """
    eligibility_weeks = max(1, int(eligibility_weeks))
    max_duration_weeks = max(1, int(max_duration_weeks))
    for firm in firms:
        ensure_restructuring_state(firm)
        firm.restructuring_start_this_week = False
        firm.weekly_interest_relief_amount = 0.0
        firm.termout_start_this_week = False
        firm.termout_snapshot_amount = 0.0
        if not enabled:
            firm.restructuring_active = False

        if (
            termout_enabled
            and getattr(firm, "active_contract_default", False)
            and not getattr(firm, "termout_used_this_default", False)
            and firm.restructuring_default_weeks >= eligibility_weeks
        ):
            pre_snapshot_arrears = max(
                0.0, float(getattr(firm, "interest_arrears", 0.0))
            )
            firm.termout_pre_snapshot_arrears = pre_snapshot_arrears
            firm.termout_snapshot_amount = pre_snapshot_arrears
            firm.legacy_arrears_term_claim = max(
                0.0, float(getattr(firm, "legacy_arrears_term_claim", 0.0))
            ) + pre_snapshot_arrears
            # Only the revolving arrears bucket is cleared.  No cash, money,
            # principal, or lender claim is changed by this reclassification.
            firm.interest_arrears = 0.0
            firm.termout_start_this_week = True
            firm.termout_used_this_default = True

        if firm.restructuring_active:
            if firm.restructuring_weeks_used >= max_duration_weeks:
                firm.restructuring_active = False
            continue

        if (
            getattr(firm, "active_contract_default", False)
            and not firm.restructuring_used_this_default
            and firm.restructuring_default_weeks >= eligibility_weeks
        ):
            firm.restructuring_active = True
            firm.restructuring_start_this_week = True
            firm.restructuring_weeks_used = 0
            firm.restructuring_used_this_default = True


def finalize_restructuring_week(
    firms,
    enabled,
    eligibility_weeks,
    max_duration_weeks,
):
    """Advance counters after Default bookkeeping has observed this week."""
    max_duration_weeks = max(1, int(max_duration_weeks))
    for firm in firms:
        ensure_restructuring_state(firm)
        if getattr(firm, "default_event_this_week", False):
            # The event week is the first completed active-contract week.
            firm.restructuring_default_weeks = 1
            firm.restructuring_used_this_default = False
            firm.termout_used_this_default = False
        elif getattr(firm, "active_contract_default", False):
            firm.restructuring_default_weeks += 1
        else:
            # A genuine current-service cure closes this contract and permits
            # a future, separately triggered Default to receive one attempt.
            firm.restructuring_default_weeks = 0
            firm.restructuring_used_this_default = False
            firm.termout_used_this_default = False
            firm.restructuring_active = False
            firm.restructuring_weeks_used = 0

        if not enabled:
            firm.restructuring_active = False
            firm.restructuring_start_this_week = False
            continue

        if firm.restructuring_active:
            firm.restructuring_weeks_used += 1
            if (
                not getattr(firm, "active_contract_default", False)
                or firm.restructuring_weeks_used >= max_duration_weeks
            ):
                firm.restructuring_active = False
