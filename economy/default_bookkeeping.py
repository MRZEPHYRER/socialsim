"""Passive Default eligibility and event bookkeeping.

This module deliberately contains no economic action.  It only turns the
accepted Step 13.5E flow observations into a D3 label and advances the small
Step 13.6E episode state machine.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import fmean


DISTRESS_TOLERANCE = 1e-9
DISTRESS_WINDOW_WEEKS = 26
D3_TRIGGER_WEEKS = 4
# D3_REARM_WEEKS is retained only for historical one-argument replay helpers.
# Runtime contractual Default cure is based on current-interest service, not D3.
D3_REARM_WEEKS = 4


def _number(value, default=0.0):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if not math.isnan(number) else default


def _ratio(numerator, denominator):
    return numerator / denominator if denominator else 0.0


def _mean(values):
    return fmean(values) if values else 0.0


def _window(history, size=DISTRESS_WINDOW_WEEKS):
    return history[-size:]


def _get(source, name, default=0.0):
    if isinstance(source, dict):
        return source.get(name, default)
    return getattr(source, name, default)


def _boolean(value):
    if isinstance(value, str):
        return value.strip().lower() == "true"
    return bool(value)


def _service_state(observation):
    obligation = observation["total_interest_obligation"]
    paid = observation["interest_paid"]
    unpaid = observation["current_interest_unpaid"]
    if obligation <= DISTRESS_TOLERANCE:
        return "I0"
    if unpaid <= DISTRESS_TOLERANCE:
        return "I1"
    if paid > DISTRESS_TOLERANCE:
        return "I2"
    return "I3"


def firm_distress_observation(firm):
    """Build the exact current-flow inputs used by accepted Family A."""

    credit_limit = _number(_get(firm, "credit_limit", 0.0))
    opening_exposure = _number(
        _get(
            firm,
            "opening_lender_exposure",
            _get(firm, "loan_balance", 0.0)
            + _get(firm, "opening_interest_arrears", 0.0),
        )
    )
    utilization = (
        opening_exposure / credit_limit
        if credit_limit > DISTRESS_TOLERANCE and math.isfinite(credit_limit)
        else 0.0
    )
    requested_credit = _number(_get(firm, "requested_credit", 0.0))
    denied_credit = _number(_get(firm, "denied_credit", 0.0))
    operating_cash_flow = (
        _number(_get(firm, "sales_revenue", 0.0))
        - _number(
            _get(
                firm,
                "executed_wage_bill",
                _get(firm, "wage_payment", 0.0),
            )
        )
        + _number(_get(firm, "public_sector_cash_inflow", 0.0))
        - _number(_get(firm, "public_sector_cash_outflow", 0.0))
    )
    observation = {
        "interest_state": "",
        "total_interest_obligation": _number(
            _get(firm, "total_interest_obligation", 0.0)
        ),
        "interest_paid": _number(_get(firm, "interest_paid", 0.0)),
        "current_interest_unpaid": _number(
            _get(firm, "current_interest_unpaid", 0.0)
        ),
        "opening_interest_arrears": _number(
            _get(firm, "opening_interest_arrears", 0.0)
        ),
        "closing_interest_arrears": _number(
            _get(
                firm,
                "closing_interest_arrears",
                _get(firm, "interest_arrears", 0.0),
            )
        ),
        "credit_limit": credit_limit,
        "opening_lender_exposure": opening_exposure,
        "credit_headroom": _number(_get(firm, "credit_headroom", 0.0)),
        "credit_requested": requested_credit,
        "credit_denied": denied_credit,
        "credit_binding": _boolean(
            _get(firm, "credit_limit_binding", False)
        ),
        "exposure_utilization": utilization,
        "scheduled_wage_bill": _number(
            _get(
                firm,
                "scheduled_wage_bill",
                _get(firm, "wage_bill", 0.0),
            )
        ),
        "executed_wage_bill": _number(
            _get(
                firm,
                "executed_wage_bill",
                _get(firm, "wage_payment", 0.0),
            )
        ),
        "payroll_funding_ratio": _number(
            _get(firm, "payroll_funding_ratio", 1.0), 1.0
        ),
        "payroll_underfunding": False,
        "operating_cash_flow": operating_cash_flow,
    }
    observation["interest_state"] = _service_state(observation)
    observation["payroll_underfunding"] = (
        observation["payroll_funding_ratio"]
        < 1.0 - DISTRESS_TOLERANCE
    )
    return observation


def family_a_state(history):
    """Return the accepted Family-A state for a history including this week.

    The rolling definitions mirror the accepted Step 13.5E implementation:
    26-week trailing flow windows, current payroll impairment, and the
    four-payroll-week severe arrears burden.
    """

    if not history:
        return "D0", {}

    current = history[-1]
    window = _window(history)
    service_failures = [
        float(row["interest_state"] in {"I2", "I3"}) for row in window
    ]
    unpaid_flow = [
        float(row["current_interest_unpaid"] > DISTRESS_TOLERANCE)
        for row in window
    ]
    binding = [float(row["credit_binding"]) for row in window]
    denied = [
        float(row["credit_denied"] > DISTRESS_TOLERANCE) for row in window
    ]
    ocf = [row["operating_cash_flow"] for row in window]
    payroll = [float(row["payroll_underfunding"]) for row in window]
    trailing_wage = _mean(
        row["scheduled_wage_bill"]
        for row in history[-DISTRESS_WINDOW_WEEKS:]
    )
    arrears_payroll_weeks = _ratio(
        current["closing_interest_arrears"], trailing_wage
    ) if trailing_wage > DISTRESS_TOLERANCE else 0.0

    service_flow = (
        _mean(unpaid_flow) >= 0.5
        or _mean(
            _ratio(row["interest_paid"], row["total_interest_obligation"])
            for row in window
        )
        < 0.99
    )
    credit_flow = _mean(denied) > 0 or _mean(binding) >= 0.5
    ocf_flow = _mean(ocf) < 0
    payroll_acute = current["payroll_underfunding"]
    burden_severe = arrears_payroll_weeks >= 4.0
    votes = sum((service_flow, credit_flow, ocf_flow))
    d2 = votes >= 2
    d3 = d2 and (payroll_acute or (votes == 3 and burden_severe))
    current_pressure = (
        current["current_interest_unpaid"] > DISTRESS_TOLERANCE
        or current["credit_denied"] > DISTRESS_TOLERANCE
        or current["payroll_underfunding"]
        or current["operating_cash_flow"] < 0
    )
    state = "D3" if d3 else "D2" if d2 else "D1" if current_pressure else "D0"
    return state, {
        "service_flow": service_flow,
        "credit_flow": credit_flow,
        "ocf_flow": ocf_flow,
        "payroll_flow": _mean(payroll) >= 0.5,
        "payroll_acute": payroll_acute,
        "burden_severe": burden_severe,
        "arrears_payroll_weeks": arrears_payroll_weeks,
        "service_failure_share_26": _mean(service_failures),
        "unpaid_flow_share_26": _mean(unpaid_flow),
        "binding_share_26": _mean(binding),
        "denied_share_26": _mean(denied),
        "mean_ocf_26": _mean(ocf),
        "payroll_under_share_26": _mean(payroll),
        "near_exhausted_credit": (
            current["exposure_utilization"] >= 0.90
            or current["credit_headroom"] <= DISTRESS_TOLERANCE
        ),
        "votes": votes,
    }


@dataclass
class DefaultStateMachine:
    """Passive D3 trigger and contractual Default state machine.

    ``active_contract_default`` is deliberately separate from the acute D3
    phase.  A Firm remains in contractual Default until it has four
    consecutive weeks with no current-interest breach, even if D3 becomes
    false in the meantime.  The one-argument ``update`` compatibility path is
    used only by historical audits that replay the pre-13.7C state machine;
    production code always supplies ``technical_interest_breach``.
    """

    default_history_count: int = 0
    active_contract_default: bool = False
    default_event_this_week: bool = False
    consecutive_d3_weeks: int = 0
    consecutive_non_d3_weeks: int = 0
    consecutive_no_breach_weeks: int = 0
    contract_cure_this_week: bool = False
    acute_default_phase_exit_this_week: bool = False

    @property
    def active_default_episode(self):
        """Compatibility alias for pre-13.7C diagnostics and audit scripts."""
        return self.active_contract_default

    @active_default_episode.setter
    def active_default_episode(self, value):
        self.active_contract_default = bool(value)

    def _update_legacy_d3_rearm(self, is_d3):
        """Replay-only implementation of the accepted pre-13.7C rule."""
        self.default_event_this_week = False
        self.contract_cure_this_week = False
        self.acute_default_phase_exit_this_week = False
        if not self.active_contract_default:
            self.consecutive_non_d3_weeks = 0
            if is_d3:
                self.consecutive_d3_weeks += 1
                if self.consecutive_d3_weeks >= D3_TRIGGER_WEEKS:
                    self.active_contract_default = True
                    self.default_history_count += 1
                    self.default_event_this_week = True
            else:
                self.consecutive_d3_weeks = 0
            return self

        if is_d3:
            self.consecutive_d3_weeks += 1
            self.consecutive_non_d3_weeks = 0
            return self

        self.consecutive_d3_weeks = 0
        self.consecutive_non_d3_weeks += 1
        if self.consecutive_non_d3_weeks >= D3_REARM_WEEKS:
            self.active_contract_default = False
            self.consecutive_non_d3_weeks = 0
            self.consecutive_d3_weeks = 0
        return self

    def update(self, is_d3, technical_interest_breach=None):
        if technical_interest_breach is None:
            return self._update_legacy_d3_rearm(is_d3)

        was_active = self.active_contract_default
        self.default_event_this_week = False
        self.contract_cure_this_week = False
        self.acute_default_phase_exit_this_week = (
            self.consecutive_d3_weeks > 0 and not is_d3
        )

        if is_d3:
            self.consecutive_d3_weeks += 1
            self.consecutive_non_d3_weeks = 0
        else:
            self.consecutive_d3_weeks = 0
            self.consecutive_non_d3_weeks += 1

        if technical_interest_breach:
            self.consecutive_no_breach_weeks = 0
        else:
            self.consecutive_no_breach_weeks += 1

        if was_active:
            if self.consecutive_no_breach_weeks >= D3_TRIGGER_WEEKS:
                self.active_contract_default = False
                self.contract_cure_this_week = True
                self.consecutive_no_breach_weeks = 0
                self.consecutive_d3_weeks = 0
                self.consecutive_non_d3_weeks = 0
            return self

        if is_d3 and self.consecutive_d3_weeks >= D3_TRIGGER_WEEKS:
            self.active_contract_default = True
            self.default_history_count += 1
            self.default_event_this_week = True
        return self


def ensure_default_state(firm):
    """Backfill fields on Firms loaded from pre-13.6E checkpoints."""

    # Old checkpoints contain only the pre-13.7C active_default_episode flag.
    # Its historical meaning was D3 re-arm exposure, so it is intentionally
    # not promoted into an unknown contractual Default state.
    had_contract_state = hasattr(firm, "active_contract_default")
    defaults = {
        "default_history_count": 0,
        "active_contract_default": False,
        "active_default_episode": False,
        "default_event_this_week": False,
        "consecutive_d3_weeks": 0,
        "consecutive_non_d3_weeks": 0,
        "consecutive_no_breach_weeks": 0,
        "contract_cure_this_week": False,
        "acute_default_phase_exit_this_week": False,
        "default_distress_history": [],
        "default_distress_state": "D0",
        "default_last_event_step": None,
    }
    for name, value in defaults.items():
        if not hasattr(firm, name):
            setattr(firm, name, list(value) if isinstance(value, list) else value)
    if not had_contract_state:
        firm.active_contract_default = False
    firm.active_default_episode = bool(firm.active_contract_default)
    if firm.default_distress_history is None:
        firm.default_distress_history = []
    return firm


def update_firm_default_bookkeeping(firm, global_step):
    """Record the finalized current week and advance the passive state machine."""

    ensure_default_state(firm)
    history = list(firm.default_distress_history)
    observation = firm_distress_observation(firm)
    history.append(observation)
    history = history[-DISTRESS_WINDOW_WEEKS:]
    state, features = family_a_state(history)
    machine = DefaultStateMachine(
        default_history_count=int(firm.default_history_count),
        active_contract_default=bool(firm.active_contract_default),
        default_event_this_week=False,
        consecutive_d3_weeks=int(firm.consecutive_d3_weeks),
        consecutive_non_d3_weeks=int(firm.consecutive_non_d3_weeks),
        consecutive_no_breach_weeks=int(
            getattr(firm, "consecutive_no_breach_weeks", 0)
        ),
    )
    technical_interest_breach = (
        observation["current_interest_unpaid"] > DISTRESS_TOLERANCE
    )
    machine.update(state == "D3", technical_interest_breach)
    firm.default_distress_history = history
    firm.default_distress_state = state
    firm.default_history_count = machine.default_history_count
    firm.active_contract_default = machine.active_contract_default
    firm.active_default_episode = machine.active_contract_default
    firm.default_event_this_week = machine.default_event_this_week
    firm.consecutive_d3_weeks = machine.consecutive_d3_weeks
    firm.consecutive_non_d3_weeks = machine.consecutive_non_d3_weeks
    firm.consecutive_no_breach_weeks = machine.consecutive_no_breach_weeks
    firm.contract_cure_this_week = machine.contract_cure_this_week
    firm.acute_default_phase_exit_this_week = (
        machine.acute_default_phase_exit_this_week
    )
    if machine.default_event_this_week:
        firm.default_last_event_step = global_step
    firm.default_distress_features = features
    return state, features
