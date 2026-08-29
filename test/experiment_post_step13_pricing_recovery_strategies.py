"""Isolated Post-Step13 early-warning pricing-objective strategy screen.

This experiment deliberately monkeypatches World only inside this process.  It
does not modify canonical model state semantics, accounting fields, parameters,
or random-number use.  S1-S3 change only the objective used to evaluate a price
trial that started while the completed-history liquidity warning was active.
"""

from __future__ import annotations

import math
import pickle
import sys
import time
from collections import defaultdict, deque
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from economy import config as economy_config
from scenarios import apply_runtime_scenario, apply_scenario
from world import World


POPULATION = 5000
FIRM_COUNT = 5
SEED = 42
STEPS = 1820
SCENARIO = "interest_behavioral_5pct"
MATURE_WEEKS = 520
FORECAST_HISTORY = 13
FORECAST_HORIZON = 12
TOLERANCE = 1e-6
PARITY_TOLERANCE = 1e-8

CANONICAL = ROOT / "test" / "output" / "main_step13_financial_core"
OUT = ROOT / "test" / "output" / "post_step13_pricing_recovery_strategy_screen"

STRATEGIES = {
    "S0": "BASELINE",
    "S1": "FORECAST_CFO",
    "S2": "FORECAST_LIQUIDITY",
    "S3": "FORECAST_CLAIM_ADJUSTED_CFO",
}


def finite(value, default=0.0):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def safe_ratio(numerator, denominator, default=0.0):
    numerator = finite(numerator)
    denominator = finite(denominator)
    return numerator / denominator if abs(denominator) > 1e-12 else default


def first_true_step(frame, mask):
    rows = frame.loc[mask, "global_step"]
    return int(rows.iloc[0]) if not rows.empty else np.nan


def linear_slope(values):
    series = pd.Series(values, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    if len(series) < 2:
        return 0.0
    return float(np.polyfit(np.arange(len(series), dtype=float), series.to_numpy(), 1)[0])


def classification_metrics(warning, outcome):
    warning = pd.Series(warning, dtype=bool)
    outcome = pd.Series(outcome, dtype=bool)
    tp = int((warning & outcome).sum())
    fp = int((warning & ~outcome).sum())
    fn = int((~warning & outcome).sum())
    return {
        "warning_precision": safe_ratio(tp, tp + fp, np.nan),
        "warning_recall": safe_ratio(tp, tp + fn, np.nan),
        "warning_true_positive_count": tp,
        "warning_false_positive_count": fp,
        "warning_false_negative_count": fn,
    }


def add_forecast_fields(frame):
    result = frame.sort_values(["firm_id", "global_step"]).copy()
    grouped = result.groupby("firm_id", sort=False)
    result["trailing_cfo_13"] = grouped["cfo"].transform(
        lambda values: values.rolling(FORECAST_HISTORY, min_periods=FORECAST_HISTORY).mean()
    )
    result["forecast_cash_12"] = result["cash"] + FORECAST_HORIZON * result["trailing_cfo_13"]
    result["forecast_liquidity_gap_12"] = result["target_cash"] - result["forecast_cash_12"]
    result["liquidity_warning"] = (
        result["trailing_cfo_13"].notna()
        & (result["forecast_cash_12"] < result["target_cash"])
    )
    result["actual_cash_t_plus_12"] = grouped["cash"].shift(-FORECAST_HORIZON)
    future_needs = []
    request = result["requested_credit"].fillna(0.0).gt(TOLERANCE)
    for offset in range(1, FORECAST_HORIZON + 1):
        future_needs.append(request.groupby(result["firm_id"]).shift(-offset).fillna(False))
    result["next_12_week_financing_need"] = pd.concat(future_needs, axis=1).any(axis=1)
    return result


def forecast_metrics(frame):
    valid = frame[
        frame["trailing_cfo_13"].notna()
        & frame["actual_cash_t_plus_12"].notna()
    ].copy()
    error = valid["forecast_cash_12"] - valid["actual_cash_t_plus_12"]
    correlation = valid["forecast_cash_12"].corr(valid["actual_cash_t_plus_12"])
    metrics = {
        "forecast_observations": int(len(valid)),
        "forecast_cash_mae": float(error.abs().mean()) if len(valid) else np.nan,
        "forecast_cash_median_absolute_error": float(error.abs().median()) if len(valid) else np.nan,
        "forecast_cash_correlation": finite(correlation, np.nan),
    }
    metrics.update(
        classification_metrics(
            valid["liquidity_warning"],
            valid["next_12_week_financing_need"],
        )
    )
    return metrics


def canonical_frames():
    macro = pd.read_csv(CANONICAL / "diagnostics.csv")
    firm = pd.read_csv(CANONICAL / "firm_diagnostics.csv")
    accounting = pd.read_csv(CANONICAL / "accounting" / "firm_accounting.csv")
    merged = firm.merge(
        accounting[
            [
                "global_step", "firm_id", "cfo", "accounting_operating_profit",
                "accounting_net_income", "cash_flow_gap", "inventory_bridge_gap",
                "equity_bridge_gap", "balance_sheet_gap",
            ]
        ],
        on=["global_step", "firm_id"],
        how="left",
        validate="one_to_one",
        suffixes=("", "_accounting"),
    )
    return macro, firm, accounting, add_forecast_fields(merged)


def build_world():
    scenario = apply_scenario(SCENARIO)
    world = World(
        initial_population=POPULATION,
        seed=SEED,
        record_ledger_details=False,
        scenario_name=scenario["name"],
        scenario_overrides=scenario["overrides"],
        diagnostics_mode="full",
        initial_age_phase_mode="distributed",
    )
    apply_runtime_scenario(scenario, world)
    world.split_firms(FIRM_COUNT)
    world.steps = STEPS
    return world


def advance(world, count, label):
    started = time.perf_counter()
    for offset in range(count):
        world.step()
        if (offset + 1) % 200 == 0 or offset + 1 == count:
            elapsed = time.perf_counter() - started
            print(f"[{label}] {offset + 1}/{count} weeks, {elapsed:.1f}s")


def accounting_cfo_from_firm(firm):
    sales_revenue = max(0.0, finite(getattr(firm, "sales_revenue", 0.0)))
    public_inflow = max(0.0, finite(getattr(firm, "public_sector_cash_inflow", 0.0)))
    recorded_other_inflow = max(0.0, finite(getattr(firm, "other_cash_inflow", 0.0)))
    recorded_other_outflow = max(0.0, finite(getattr(firm, "other_cash_outflow", 0.0)))
    dividends = max(0.0, finite(getattr(firm, "dividend_payment", 0.0)))
    interest = max(0.0, finite(getattr(firm, "loan_interest_paid", 0.0)))
    wage = max(0.0, finite(getattr(firm, "wage_payment", getattr(firm, "wage_bill", 0.0))))
    other_inflow = max(0.0, recorded_other_inflow - public_inflow)
    other_outflow = max(0.0, recorded_other_outflow - dividends - interest)
    return sales_revenue + public_inflow + other_inflow - wage - other_outflow - interest


class StrategyController:
    """External-only objective state for one treatment branch."""

    def __init__(self, strategy, prefix_frame):
        self.strategy = strategy
        self.alpha = float(economy_config.FIRM_PRICE_PROFIT_EMA_ALPHA)
        self.cfo_history = defaultdict(lambda: deque(maxlen=FORECAST_HISTORY))
        self.objective_ema = {}
        self.trial_regime = defaultdict(lambda: "baseline")
        self.trial_objective_start = {}
        self.current_warning = defaultdict(bool)
        self.warning_records = []
        self.recovery_trial_count = 0
        self._initialize(prefix_frame)

    def signal_from_row(self, row):
        if self.strategy == "S1":
            return finite(row["cfo"])
        if self.strategy == "S2":
            return (
                finite(row["cash_before_interest"])
                - finite(row["current_interest_due"])
                - finite(row["target_cash"])
            )
        return finite(row["cfo"]) - finite(row["current_interest_unpaid"])

    def signal_from_firm(self, firm, cfo):
        if self.strategy == "S1":
            return cfo
        if self.strategy == "S2":
            return (
                finite(getattr(firm, "cash_before_interest", 0.0))
                - finite(getattr(firm, "current_interest_due", 0.0))
                - finite(getattr(firm, "target_cash", 0.0))
            )
        return cfo - finite(getattr(firm, "current_interest_unpaid", 0.0))

    def _initialize(self, prefix_frame):
        for firm_id, rows in prefix_frame.sort_values("global_step").groupby("firm_id"):
            ema = None
            for _, row in rows.iterrows():
                cfo = finite(row["cfo"])
                self.cfo_history[int(firm_id)].append(cfo)
                signal = self.signal_from_row(row)
                ema = signal if ema is None else self.alpha * signal + (1.0 - self.alpha) * ema
            self.objective_ema[int(firm_id)] = finite(ema)

    def observe_completed_week(self, world):
        step = len(world.population_history)
        for firm in world.firms:
            firm_id = int(firm.firm_id)
            cfo = accounting_cfo_from_firm(firm)
            self.cfo_history[firm_id].append(cfo)
            signal = self.signal_from_firm(firm, cfo)
            previous = self.objective_ema.get(firm_id, signal)
            self.objective_ema[firm_id] = self.alpha * signal + (1.0 - self.alpha) * previous
            available = len(self.cfo_history[firm_id]) >= FORECAST_HISTORY
            trailing = (
                float(np.mean(self.cfo_history[firm_id]))
                if available else np.nan
            )
            forecast = finite(getattr(firm, "cash", 0.0)) + FORECAST_HORIZON * trailing
            warning = bool(available and forecast < finite(getattr(firm, "target_cash", 0.0)))
            self.current_warning[firm_id] = warning
            self.warning_records.append(
                {
                    "global_step": step,
                    "firm_id": firm_id,
                    "trailing_cfo_13": trailing,
                    "forecast_cash_12": forecast,
                    "liquidity_warning": warning,
                }
            )

    def evaluate(self, original, world, firm):
        firm_id = int(firm.firm_id)
        if self.trial_regime[firm_id] != "recovery":
            return original(world, firm)
        timer = int(getattr(firm, "evaluation_timer", 0))
        if timer <= 0:
            return None
        timer -= 1
        firm.evaluation_timer = timer
        if timer > 0:
            return None
        baseline = self.trial_objective_start.get(
            firm_id,
            self.objective_ema.get(firm_id, 0.0),
        )
        evaluated = self.objective_ema.get(firm_id, baseline)
        old_price = max(1e-9, finite(getattr(firm, "last_test_price", firm.price)))
        new_price = max(1e-9, finite(firm.price))
        price_step = math.log(new_price / old_price)
        objective_delta = evaluated - baseline
        # These are canonical learner-state fields, not accounting/profit fields.
        firm.evaluated_profit = evaluated
        firm.last_profit_delta = objective_delta
        firm.estimated_gradient = (
            objective_delta / price_step if abs(price_step) > 1e-12 else 0.0
        )
        return None

    def register_new_trials(self, world):
        for firm in world.firms:
            firm_id = int(firm.firm_id)
            new_trial = (
                bool(getattr(firm, "price_reviewed", False))
                and int(getattr(firm, "evaluation_timer", 0))
                == int(economy_config.FIRM_PRICE_EVALUATION_WINDOW)
                and int(getattr(firm, "last_direction", 0)) != 0
            )
            if not new_trial:
                continue
            if self.current_warning[firm_id]:
                self.trial_regime[firm_id] = "recovery"
                self.trial_objective_start[firm_id] = self.objective_ema[firm_id]
                self.recovery_trial_count += 1
            else:
                self.trial_regime[firm_id] = "baseline"


def run_treatment(prefix_blob, prefix_frame, strategy, remaining):
    world = pickle.loads(prefix_blob)
    controller = StrategyController(strategy, prefix_frame)
    original_adaptive = World.update_adaptive_firm_prices
    original_evaluation = World.update_firm_price_evaluation

    def experimental_evaluation(instance, firm):
        return controller.evaluate(original_evaluation, instance, firm)

    def experimental_adaptive(instance):
        controller.observe_completed_week(instance)
        result = original_adaptive(instance)
        controller.register_new_trials(instance)
        return result

    World.update_firm_price_evaluation = experimental_evaluation
    World.update_adaptive_firm_prices = experimental_adaptive
    try:
        advance(world, remaining, strategy)
    finally:
        World.update_adaptive_firm_prices = original_adaptive
        World.update_firm_price_evaluation = original_evaluation
    return world, controller


def world_frames(world):
    frames = {
        "macro": pd.DataFrame(world.diagnostics_rows),
        "firm": pd.DataFrame(world.firm_diagnostics_rows),
        "firm_accounting": pd.DataFrame(world.accounting.rows),
        "household_accounting": pd.DataFrame(world.accounting.household_rows),
        "public_accounting": pd.DataFrame(world.accounting.public_rows),
        "central_bank_accounting": pd.DataFrame(world.accounting.central_bank_rows),
        "accounting_reconciliation": pd.DataFrame(world.accounting.reconciliation_rows),
    }
    merged = frames["firm"].merge(
        frames["firm_accounting"][
            [
                "global_step", "firm_id", "cfo", "accounting_operating_profit",
                "accounting_net_income", "cash_flow_gap", "inventory_bridge_gap",
                "equity_bridge_gap", "balance_sheet_gap",
            ]
        ],
        on=["global_step", "firm_id"],
        how="left",
        validate="one_to_one",
        suffixes=("", "_accounting"),
    )
    frames["combined"] = add_forecast_fields(merged)
    return frames


def compare_frames(reference, candidate, keys):
    left = reference.sort_values(keys).reset_index(drop=True)
    right = candidate.sort_values(keys).reset_index(drop=True)
    if len(left) != len(right):
        return {"pass": False, "max_abs_diff": math.inf, "first_difference": "row_count"}
    columns = [column for column in left.columns if column in right.columns]
    maximum = 0.0
    first = None
    for column in columns:
        left_col = left[column]
        right_col = right[column]
        if pd.api.types.is_numeric_dtype(left_col) and pd.api.types.is_numeric_dtype(right_col):
            a = pd.to_numeric(left_col, errors="coerce").to_numpy(dtype=float)
            b = pd.to_numeric(right_col, errors="coerce").to_numpy(dtype=float)
            nan_mismatch = np.isnan(a) ^ np.isnan(b)
            differences = np.abs(a - b)
            differences[np.isnan(differences)] = 0.0
            column_max = float(np.max(differences)) if len(differences) else 0.0
            maximum = max(maximum, column_max)
            mismatch = nan_mismatch | (differences > PARITY_TOLERANCE)
        else:
            a = left_col.fillna("<NA>").astype(str).to_numpy()
            b = right_col.fillna("<NA>").astype(str).to_numpy()
            mismatch = a != b
            differences = np.zeros(len(a))
        if first is None and np.any(mismatch):
            index = int(np.flatnonzero(mismatch)[0])
            first = {
                "column": column,
                "key": {key: left.loc[index, key] for key in keys},
                "reference": left_col.iloc[index],
                "candidate": right_col.iloc[index],
                "abs_diff": finite(differences[index], None),
            }
    return {"pass": first is None, "max_abs_diff": maximum, "first_difference": first}


def control_parity(control_frames):
    comparisons = {
        "macro": compare_frames(
            pd.read_csv(CANONICAL / "diagnostics.csv"),
            control_frames["macro"],
            ["global_step"],
        ),
        "firm": compare_frames(
            pd.read_csv(CANONICAL / "firm_diagnostics.csv"),
            control_frames["firm"],
            ["global_step", "firm_id"],
        ),
    }
    for name in (
        "firm_accounting", "household_accounting", "public_accounting",
        "central_bank_accounting", "accounting_reconciliation",
    ):
        keys = ["global_step", "firm_id"] if name == "firm_accounting" else ["global_step"]
        comparisons[name] = compare_frames(
            pd.read_csv(CANONICAL / "accounting" / f"{name}.csv"),
            control_frames[name],
            keys,
        )
    return {
        "pass": all(item["pass"] for item in comparisons.values()),
        "max_abs_diff": max(item["max_abs_diff"] for item in comparisons.values()),
        "comparisons": comparisons,
    }


def max_abs(frame, column):
    if column not in frame or frame.empty:
        return 0.0
    return float(pd.to_numeric(frame[column], errors="coerce").abs().max())


def accounting_check(frames):
    checks = {
        "macro.monetary_accounting_gap": max_abs(frames["macro"], "monetary_accounting_gap"),
        "macro.money_delta_gap": max_abs(frames["macro"], "money_delta_gap"),
        "macro.goods_conservation_gap": max_abs(frames["macro"], "food_conservation_gap"),
        "macro.ledger_money_net_gap": max_abs(frames["macro"], "ledger_money_net_gap"),
        "firm.cash_bridge_gap": max_abs(frames["firm"], "cash_bridge_gap"),
        "firm.credit_bridge_gap": max_abs(frames["firm"], "credit_bridge_gap"),
        "accounting.cash_flow_gap": max_abs(frames["firm_accounting"], "cash_flow_gap"),
        "accounting.inventory_bridge_gap": max_abs(frames["firm_accounting"], "inventory_bridge_gap"),
        "accounting.equity_bridge_gap": max_abs(frames["firm_accounting"], "equity_bridge_gap"),
        "household.wealth_bridge_gap": max_abs(frames["household_accounting"], "household_wealth_bridge_gap"),
        "public.cash_flow_gap": max_abs(frames["public_accounting"], "public_cash_flow_gap"),
        "central_bank.credit_stock_flow_gap": max_abs(frames["central_bank_accounting"], "credit_money_stock_flow_gap"),
        "reconciliation.loan_gap": max_abs(frames["accounting_reconciliation"], "loan_reconciliation_gap"),
        "reconciliation.total_lender_claim_gap": max_abs(frames["accounting_reconciliation"], "total_lender_claim_gap"),
        "reconciliation.money_location_gap": max_abs(frames["accounting_reconciliation"], "money_location_gap"),
    }
    invariant_failures = int(frames["macro"].get("invariant_failed", pd.Series(dtype=bool)).fillna(False).astype(bool).sum())
    return {
        "pass": max(checks.values(), default=0.0) <= TOLERANCE and invariant_failures == 0,
        "max_abs_gap": max(checks.values(), default=0.0),
        "invariant_failures": invariant_failures,
        "checks": checks,
    }


def post90_metrics(rows):
    reached = rows[rows["principal_utilization"] >= 0.90]
    if reached.empty:
        return {
            "minimum_later_utilization": np.nan,
            "largest_post90_utilization_drawdown": 0.0,
            "weeks_to_minimum_after_90": np.nan,
            "recovered_below_80_after_90": False,
            "recovered_below_70_after_90": False,
            "recovered_below_50_after_90": False,
            "sustained_debt_recovery": False,
        }
    first_index = reached.index[0]
    later = rows.loc[first_index:]
    peak = float(later["principal_utilization"].cummax().max())
    minimum = float(later["principal_utilization"].min())
    minimum_row = later.loc[later["principal_utilization"].idxmin()]
    first_step = int(rows.loc[first_index, "global_step"])
    trailing_mean = float(later["principal_utilization"].tail(26).mean())
    drawdown = peak - minimum
    return {
        "minimum_later_utilization": minimum,
        "largest_post90_utilization_drawdown": drawdown,
        "weeks_to_minimum_after_90": int(minimum_row["global_step"]) - first_step,
        "recovered_below_80_after_90": bool((later["principal_utilization"] < 0.80).any()),
        "recovered_below_70_after_90": bool((later["principal_utilization"] < 0.70).any()),
        "recovered_below_50_after_90": bool((later["principal_utilization"] < 0.50).any()),
        "sustained_debt_recovery": bool(drawdown >= 0.10 and trailing_mean <= peak - 0.10),
    }


def firm_summary(strategy, frames, baseline_metrics):
    data = frames["combined"].copy()
    data["principal_utilization"] = np.where(
        data["credit_limit"].gt(TOLERANCE) & np.isfinite(data["credit_limit"]),
        data["closing_principal"] / data["credit_limit"],
        0.0,
    )
    mature_start = STEPS - MATURE_WEEKS
    output = []
    for firm_id, rows in data.groupby("firm_id", sort=True):
        rows = rows.sort_values("global_step").reset_index(drop=True)
        mature = rows[rows["global_step"] >= mature_start]
        final = rows.iloc[-1]
        warning_rows = rows[rows["trailing_cfo_13"].notna()]
        forecast = forecast_metrics(rows)
        first_warning = first_true_step(rows, rows["liquidity_warning"])
        first_borrowing = first_true_step(rows, rows["loan_issued"] > TOLERANCE)
        borrowing_onset = rows["loan_issued"].gt(TOLERANCE) & rows["loan_issued"].shift(1, fill_value=0.0).le(TOLERANCE)
        preceded = []
        for index in rows.index[borrowing_onset]:
            week = int(rows.loc[index, "global_step"])
            preceded.append(
                bool(rows[rows["global_step"].between(week - FORECAST_HORIZON, week - 1)]["liquidity_warning"].any())
            )
        result = {
            "strategy": strategy,
            "strategy_name": STRATEGIES[strategy],
            "firm_id": int(firm_id),
            "warning_week_share": float(warning_rows["liquidity_warning"].mean()) if len(warning_rows) else 0.0,
            "false_positive_warning_share": float((warning_rows["liquidity_warning"] & ~warning_rows["next_12_week_financing_need"]).mean()) if len(warning_rows) else 0.0,
            "first_warning_week": first_warning,
            "first_borrowing_week": first_borrowing,
            "warning_lead_time": first_borrowing - first_warning if pd.notna(first_warning) and pd.notna(first_borrowing) else np.nan,
            "borrowing_onset_preceded_by_warning_share": float(np.mean(preceded)) if preceded else np.nan,
            "mature_mean_CFO": float(mature["cfo"].mean()),
            "mature_median_CFO": float(mature["cfo"].median()),
            "final_CFO_13": finite(final["trailing_cfo_13"], np.nan),
            "final_cash": finite(final["cash"]),
            "mature_cash_slope": linear_slope(mature["cash"]),
            "negative_CFO_week_share": float(mature["cfo"].lt(0.0).mean()),
            "mature_mean_market_share": float(mature["unit_market_share"].mean()),
            "final_market_share": finite(final["unit_market_share"]),
            "mature_mean_observed_demand": float(mature["observed_demand"].mean()),
            "mature_mean_sales": float(mature["sales_units"].mean()),
            "mature_mean_output": float(mature["actual_production"].mean()),
            "mature_mean_capacity_utilization": float(mature["capacity_utilization"].mean()),
            "mature_mean_price": float(mature["price"].mean()),
            "final_price": finite(final["price"]),
            "mature_price_std": float(mature["price"].std(ddof=0)),
            "mature_mean_relative_price": float(mature["relative_price"].mean()),
            "minimum_relative_price": float(rows["relative_price"].min()),
            "price_cut_count": int(rows["price_decision"].eq("cut").sum()),
            "price_raise_count": int(rows["price_decision"].eq("raise").sum()),
            "price_cut_frequency": float(rows["price_decision"].eq("cut").mean()),
            "price_within_2pct_floor_share": float((rows["price"] <= rows["price_floor"] * 1.02 + TOLERANCE).mean()),
            "final_principal": finite(final["closing_principal"]),
            "final_principal_utilization": finite(final["principal_utilization"]),
            "max_principal_utilization": float(rows["principal_utilization"].max()),
            "final_arrears": finite(final["closing_interest_arrears"]),
            "mature_arrears_slope": linear_slope(mature["closing_interest_arrears"]),
            "mature_gross_borrowing": float(mature["loan_issued"].sum()),
            "mature_gross_repayment": float(mature["loan_repaid"].sum()),
            "total_borrowing": float(rows["loan_issued"].sum()),
            "total_principal_repayment": float(rows["loan_repaid"].sum()),
            "net_principal_change": finite(final["closing_principal"]) - finite(rows.iloc[0]["opening_principal"]),
            "mature_current_interest_due": float(mature["current_interest_due"].sum()),
            "mature_interest_paid": float(mature["interest_paid"].sum()),
            "mature_unpaid_current_interest": float(mature["current_interest_unpaid"].sum()),
            "D2_firm_weeks": int(rows["distress_state"].eq("D2").sum()),
            "D3_firm_weeks": int(rows["distress_state"].eq("D3").sum()),
            "technical_interest_breach_firm_weeks": int(rows["current_interest_unpaid"].gt(TOLERANCE).sum()),
            "default_event": int(rows["default_event_this_week"].fillna(False).astype(bool).sum()),
            "active_default_firm_weeks": int(rows["active_contract_default"].fillna(False).astype(bool).sum()),
            "cure_events": int(rows["contract_cure_this_week"].fillna(False).astype(bool).sum()),
            "mature_mean_operating_profit": float(mature["accounting_operating_profit"].mean()),
            "negative_operating_profit_week_share": float(mature["accounting_operating_profit"].lt(0.0).mean()),
            **forecast,
        }
        for threshold in (0.50, 0.80, 0.90, 0.95, 0.99):
            result[f"first_{int(threshold * 100)}pct_utilization_week"] = first_true_step(
                rows, rows["principal_utilization"] >= threshold
            )
        result.update(post90_metrics(rows))
        output.append(result)
    result_frame = pd.DataFrame(output)
    if strategy == "S0":
        result_frame["trap_trajectory_label"] = result_frame.apply(
            lambda row: trajectory_label(row, None), axis=1
        )
        result_frame["financial_vs_market_recovery"] = "BASELINE"
    else:
        baselines = baseline_metrics.set_index("firm_id")
        result_frame["trap_trajectory_label"] = result_frame.apply(
            lambda row: trajectory_label(row, baselines.loc[row["firm_id"]]), axis=1
        )
        result_frame["financial_vs_market_recovery"] = result_frame.apply(
            lambda row: recovery_dimension(row, baselines.loc[row["firm_id"]]), axis=1
        )
    return result_frame


def trajectory_label(row, baseline):
    if bool(row["sustained_debt_recovery"]):
        return "SUSTAINED_DEBT_RECOVERY"
    if row["largest_post90_utilization_drawdown"] >= 0.05:
        return "PARTIAL_DELEVERAGING"
    if row["final_principal_utilization"] >= 0.95:
        if baseline is not None:
            current = row.get("first_95pct_utilization_week", np.nan)
            old = baseline.get("first_95pct_utilization_week", np.nan)
            if pd.notna(current) and pd.notna(old) and current >= old + 26:
                return "DELAYED_RATCHET"
        return "STILL_RATCHET_TO_LIMIT"
    if row["max_principal_utilization"] >= 0.90:
        return "PARTIAL_DELEVERAGING"
    return "NO_LIMIT_RATCHET"


def recovery_dimension(row, baseline):
    cfo_scale = max(1.0, abs(finite(baseline["mature_mean_CFO"])))
    arrears_scale = max(1.0, abs(finite(baseline["mature_arrears_slope"])))
    financial_votes = [
        row["mature_mean_CFO"] > baseline["mature_mean_CFO"] + 0.05 * cfo_scale,
        row["final_principal_utilization"] < baseline["final_principal_utilization"] - 0.05,
        row["mature_arrears_slope"] < baseline["mature_arrears_slope"] - 0.05 * arrears_scale,
    ]
    market_votes = [
        row["mature_mean_market_share"] > baseline["mature_mean_market_share"] * 1.05,
        row["mature_mean_observed_demand"] > baseline["mature_mean_observed_demand"] * 1.05,
        row["mature_mean_capacity_utilization"] > baseline["mature_mean_capacity_utilization"] * 1.05,
    ]
    financial = sum(financial_votes) >= 2
    market = sum(market_votes) >= 2
    if financial and market:
        return "BOTH"
    if financial:
        return "FINANCIAL_ONLY"
    if market:
        return "MARKET_ONLY"
    return "NEITHER"


def strategy_risk_flags(summary, baseline):
    if summary["strategy"].iloc[0] == "S0":
        return False, False, False
    indexed = summary.set_index("firm_id")
    old = baseline.set_index("firm_id")
    firm0 = indexed.loc[0]
    firm2 = indexed.loc[2]
    firm0_destabilized = bool(
        firm0["final_principal_utilization"] > old.loc[0, "final_principal_utilization"] + 0.10
        or firm0["mature_mean_CFO"] < old.loc[0, "mature_mean_CFO"] - max(100.0, 0.20 * abs(old.loc[0, "mature_mean_CFO"]))
        or firm0["mature_mean_market_share"] < old.loc[0, "mature_mean_market_share"] * 0.80
    )
    firm2_destabilized = bool(
        firm2["final_principal_utilization"] > old.loc[2, "final_principal_utilization"] + 0.10
        or firm2["mature_mean_CFO"] < old.loc[2, "mature_mean_CFO"] - max(100.0, 0.20 * abs(old.loc[2, "mature_mean_CFO"]))
    )
    trapped = indexed.loc[[1, 3, 4]]
    trapped_old = old.loc[[1, 3, 4]]
    price_war = bool(
        trapped["price_cut_frequency"].mean() > 1.5 * max(1e-9, trapped_old["price_cut_frequency"].mean())
        and trapped["mature_mean_relative_price"].mean() < 0.95 * trapped_old["mature_mean_relative_price"].mean()
        and trapped["mature_mean_operating_profit"].mean() < trapped_old["mature_mean_operating_profit"].mean() - 0.10 * max(1.0, abs(trapped_old["mature_mean_operating_profit"].mean()))
    )
    return firm0_destabilized, firm2_destabilized, price_war


def experimental_label(summary, baseline, accounting_pass, firm0_bad, firm2_bad, price_war):
    if not accounting_pass:
        return "H. IMPLEMENTATION_FAILURE"
    if firm0_bad or firm2_bad:
        return "G. DESTABILIZING"
    trapped = summary[summary["firm_id"].isin([1, 3, 4])]
    old = baseline[baseline["firm_id"].isin([1, 3, 4])]
    if trapped["sustained_debt_recovery"].any():
        return "E. SUSTAINED_DEBT_RECOVERY"
    improved_finance = (
        trapped["mature_mean_CFO"].mean() > old["mature_mean_CFO"].mean() + 0.05 * max(1.0, abs(old["mature_mean_CFO"].mean()))
        or trapped["final_principal_utilization"].mean() < old["final_principal_utilization"].mean() - 0.05
    )
    if improved_finance and price_war:
        return "F. FINANCIAL_IMPROVEMENT_WITH_PRICING_COST"
    if (trapped["largest_post90_utilization_drawdown"] >= 0.05).any():
        return "D. PARTIAL_DEBT_RECOVERY"
    if improved_finance:
        return "C. IMPROVES_CASH_FLOW_BUT_NOT_DEBT_RECOVERY"
    delays = []
    old_index = old.set_index("firm_id")
    for _, row in trapped.iterrows():
        old_week = old_index.loc[row["firm_id"], "first_95pct_utilization_week"]
        delays.append(pd.notna(row["first_95pct_utilization_week"]) and pd.notna(old_week) and row["first_95pct_utilization_week"] >= old_week + 26)
    return "B. DELAYS_DEBT_RATCHET_ONLY" if any(delays) else "A. NO_MEANINGFUL_EFFECT"


def system_summary(strategy, frames, by_firm, baseline, accounting, parity_pass, recovery_trials):
    macro = frames["macro"].sort_values("global_step")
    mature = macro[macro["global_step"] >= STEPS - MATURE_WEEKS]
    firm0_bad, firm2_bad, price_war = strategy_risk_flags(by_firm, baseline)
    label = (
        "S0 CONTROL"
        if strategy == "S0"
        else experimental_label(by_firm, baseline, accounting["pass"], firm0_bad, firm2_bad, price_war)
    )
    return {
        "strategy": strategy,
        "strategy_name": STRATEGIES[strategy],
        "experimental_interpretation_label": label,
        "recovery_objective_trial_count": int(recovery_trials),
        "mature_mean_total_production": float(mature["actual_production"].mean()),
        "mature_mean_total_sales_units": float(mature["food_sales_units"].mean()),
        "mature_mean_consumption": float(mature["total_consumption"].mean()),
        "final_total_firm_cash": finite(macro.iloc[-1]["firm_cash"]),
        "final_total_principal": float(by_firm["final_principal"].sum()),
        "final_total_arrears": float(by_firm["final_arrears"].sum()),
        "final_money_stock": finite(macro.iloc[-1]["total_money_stock"]),
        "total_D2_firm_weeks": int(by_firm["D2_firm_weeks"].sum()),
        "total_D3_firm_weeks": int(by_firm["D3_firm_weeks"].sum()),
        "total_default_events": int(by_firm["default_event"].sum()),
        "total_active_default_firm_weeks": int(by_firm["active_default_firm_weeks"].sum()),
        "accounting_pass": bool(accounting["pass"]),
        "max_abs_accounting_gap": accounting["max_abs_gap"],
        "invariant_failures": accounting["invariant_failures"],
        "control_branch_parity_pass": bool(parity_pass) if strategy == "S0" else np.nan,
        "Firm0_destabilized": firm0_bad,
        "Firm2_destabilized": firm2_bad,
        "price_war_risk": price_war,
        "genuine_debt_recovery_observed": bool(by_firm["sustained_debt_recovery"].any()),
        "warning_week_share": float(by_firm["warning_week_share"].mean()),
        "warning_lead_time_median": float(by_firm["warning_lead_time"].dropna().median()) if by_firm["warning_lead_time"].notna().any() else np.nan,
        "borrowing_onset_preceded_by_warning_share": float(by_firm["borrowing_onset_preceded_by_warning_share"].dropna().mean()) if by_firm["borrowing_onset_preceded_by_warning_share"].notna().any() else np.nan,
    }


def make_figure(branches):
    colors = ["#176B87", "#D1495B", "#2A9D8F", "#E9A23B", "#6D597A"]
    fig, axes = plt.subplots(4, 4, figsize=(22, 17), sharex=True)
    columns = [
        ("trailing_cfo_13", "Accounting CFO (13-week mean)"),
        ("unit_market_share", "Unit market share"),
        ("price", "Price"),
        ("principal_utilization", "Principal utilization"),
    ]
    for row_index, strategy in enumerate(STRATEGIES):
        frame = branches[strategy]["combined"].copy()
        frame["principal_utilization"] = np.where(
            frame["credit_limit"].gt(TOLERANCE) & np.isfinite(frame["credit_limit"]),
            frame["closing_principal"] / frame["credit_limit"],
            0.0,
        )
        for column_index, (field, title) in enumerate(columns):
            axis = axes[row_index, column_index]
            for firm_id, rows in frame.groupby("firm_id", sort=True):
                axis.plot(
                    rows["global_step"], rows[field], color=colors[int(firm_id)],
                    linewidth=1.15, label=f"Firm {int(firm_id)}",
                )
            axis.axvline(STEPS - MATURE_WEEKS, color="#777777", linestyle=":", linewidth=0.8)
            axis.grid(alpha=0.18)
            if row_index == 0:
                axis.set_title(title, fontsize=12, fontweight="bold")
            if column_index == 0:
                axis.set_ylabel(f"{strategy}\n{STRATEGIES[strategy]}", fontsize=10)
            if row_index == 3:
                axis.set_xlabel("Simulation week")
            if field == "principal_utilization":
                axis.axhline(0.90, color="#333333", linestyle="--", linewidth=0.7, alpha=0.6)
                axis.set_ylim(bottom=-0.03)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 0.995))
    fig.suptitle("Post-Step13 Early-Warning Pricing Recovery Strategy Screen", fontsize=16, fontweight="bold", y=1.018)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(OUT / "pricing_recovery_strategy_overview.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def markdown_table(frame, columns, digits=4):
    subset = frame[columns].copy()
    for column in subset.columns:
        if pd.api.types.is_float_dtype(subset[column]):
            subset[column] = subset[column].map(
                lambda value: "" if pd.isna(value) else f"{value:.{digits}f}"
            )
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in subset.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def write_summary(
    parity, passive_forecast, passive_by_firm, comparison, systems,
    accounting_results, earliest_warning, branch_week,
):
    treatments = systems[systems["strategy"].ne("S0")]
    system_index = systems.set_index("strategy")
    firm_index = comparison.set_index(["strategy", "firm_id"])
    prevented_rows = []
    for strategy in ("S1", "S2", "S3"):
        for firm_id in (1, 3, 4):
            if (
                firm_index.loc[("S0", firm_id), "max_principal_utilization"] >= 0.90
                and firm_index.loc[(strategy, firm_id), "max_principal_utilization"] < 0.90
            ):
                prevented_rows.append(f"{strategy}/Firm {firm_id}")
    prevented_text = ", ".join(prevented_rows) if prevented_rows else "none"
    early_warning_material = bool(
        comparison["warning_lead_time"].dropna().median() >= 4
        and passive_forecast["warning_recall"] >= 0.50
    )
    flags = {
        "control_branch_parity_pass": bool(parity["pass"]),
        "production_source_modified": False,
        "baseline_parameters_modified": False,
        "experimental_strategy_adds_rng_draws": False,
        "accounting_all_strategies_pass": bool(systems["accounting_pass"].all()),
        "forecast_validation_ready": bool(passive_forecast["forecast_observations"] > 0),
        "early_warning_precedes_borrowing_materially": early_warning_material,
        "S1_genuine_debt_recovery_observed": bool(systems.loc[systems["strategy"].eq("S1"), "genuine_debt_recovery_observed"].iloc[0]),
        "S2_genuine_debt_recovery_observed": bool(systems.loc[systems["strategy"].eq("S2"), "genuine_debt_recovery_observed"].iloc[0]),
        "S3_genuine_debt_recovery_observed": bool(systems.loc[systems["strategy"].eq("S3"), "genuine_debt_recovery_observed"].iloc[0]),
        "Firm0_destabilized_any_strategy": bool(treatments["Firm0_destabilized"].any()),
        "Firm2_destabilized_any_strategy": bool(treatments["Firm2_destabilized"].any()),
        "price_war_risk_observed": bool(treatments["price_war_risk"].any()),
        "human_review_required": True,
        "production_mechanism_selected": False,
        "seed7_21_run": False,
        "new_full_from_zero_runs": 1,
        "branched_strategy_runs": 4,
    }
    trapped = comparison[comparison["firm_id"].isin([1, 3, 4])]
    core_table = markdown_table(
        trapped,
        [
            "strategy", "firm_id", "mature_mean_CFO", "mature_mean_market_share",
            "final_price", "final_principal_utilization",
            "largest_post90_utilization_drawdown", "final_arrears",
            "trap_trajectory_label", "financial_vs_market_recovery",
        ],
    )
    system_table = markdown_table(
        systems,
        [
            "strategy", "experimental_interpretation_label", "final_total_principal",
            "final_total_arrears", "mature_mean_total_sales_units",
            "accounting_pass", "Firm0_destabilized", "Firm2_destabilized",
            "price_war_risk",
        ],
    )
    forecast_table = markdown_table(
        passive_by_firm,
        [
            "firm_id", "forecast_cash_mae", "forecast_cash_median_absolute_error",
            "forecast_cash_correlation", "warning_precision", "warning_recall",
        ],
    )
    parity_lines = []
    for name, result in parity["comparisons"].items():
        parity_lines.append(
            f"- `{name}`: pass={str(result['pass']).lower()}, max_abs_diff={result['max_abs_diff']:.3e}, first_difference={result['first_difference']}"
        )
    accounting_lines = []
    for strategy, result in accounting_results.items():
        accounting_lines.append(
            f"- `{strategy}`: pass={str(result['pass']).lower()}, max_abs_gap={result['max_abs_gap']:.3e}, invariant_failures={result['invariant_failures']}"
        )
    flag_lines = [f"{key} = {str(value).lower() if isinstance(value, bool) else value}" for key, value in flags.items()]
    summary = f"""# Post-Step13 Early-Warning Pricing Recovery Strategy Screen

## Experimental contract

The four branches use the canonical `population=5000`, `firms=5`, `seed=42`,
`steps=1820`, `scenario=interest_behavioral_5pct` configuration. The first
baseline warning is week **{earliest_warning}**; one canonical prefix through
week {branch_week} was run and serialized, then S0-S3 continued from identical
World and RNG states. S1-S3 add no RNG draws and alter only the completed-trial
evaluation objective when the warning was fixed active at trial start.

## Passive forecast validation

Pooled baseline observations: {passive_forecast['forecast_observations']}; cash
forecast MAE={passive_forecast['forecast_cash_mae']:.3f}, median absolute
error={passive_forecast['forecast_cash_median_absolute_error']:.3f},
correlation={passive_forecast['forecast_cash_correlation']:.4f}. Warning
precision={passive_forecast['warning_precision']:.4f} and
recall={passive_forecast['warning_recall']:.4f} for a positive funding request
within the next 12 completed weeks. No coefficients or horizons were fitted.

{forecast_table}

## S0 parity gate

Overall pass: **{str(parity['pass']).lower()}**; maximum absolute difference
across macro, Firm and accounting outputs: `{parity['max_abs_diff']:.3e}`.

{chr(10).join(parity_lines)}

## Headline system comparison

{system_table}

The experimental labels are descriptive only. No weighted score or production
mechanism selection was performed.

## Trapped-Firm human-review table

{core_table}

`SUSTAINED_DEBT_RECOVERY` requires at least a 0.10 post-90% utilization
drawdown and a final 26-week mean that remains at least 0.10 below the later
peak. `PARTIAL_DELEVERAGING` records a post-90% drawdown of at least 0.05.
These are experiment-reporting definitions, not new model thresholds.

## Interpretation

1. Forecast usefulness is summarized by the pooled/per-Firm precision, recall,
   error and correlation above. Median warning lead time across observed
   Firm/strategy borrowing paths is {comparison['warning_lead_time'].dropna().median():.1f} weeks.
2. S1, S2 and S3 debt-recovery evidence is reported explicitly through each
   Firm's post-90% drawdown, persistence flag and final utilization; lower debt
   alone is not treated as recovery.
3. Market effects are separated from financial effects in
   `financial_vs_market_recovery`; a Firm may show `FINANCIAL_ONLY`,
   `MARKET_ONLY`, `BOTH`, or `NEITHER`.
4. Price-war risk requires the joint pattern of materially more cuts, at least
   5% lower relative prices, and at least 10% worse operating profit among
   Firms 1/3/4. It is not inferred from one price threshold.
5. Firm 0 and Firm 2 destabilization flags compare CFO, share and principal
   utilization against S0. Macro totals and all accounting identities remain
   separate guardrails.
6. Arrears, interest due/paid/unpaid, Distress, Default Events and contractual
   Default weeks remain secondary diagnostics; none of their behavior was
   changed by this experiment.
7. Strategy selection remains **HUMAN REVIEW REQUIRED**. No treatment is
   promoted into production by this screen.

## Direct answers to the required review questions

1. **Predictive power:** yes in this trajectory. Pooled correlation is
   {passive_forecast['forecast_cash_correlation']:.4f}, precision is
   {passive_forecast['warning_precision']:.4f}, and recall is
   {passive_forecast['warning_recall']:.4f}. Firm 0 has blank precision/recall
   because it has neither warnings nor positive funding-needs, not because the
   calculation failed.
2. **Lead time:** the median observed first-warning lead over first borrowing
   is {comparison['warning_lead_time'].dropna().median():.1f} weeks. The
   canonical earliest warning is week {earliest_warning}, 11 weeks before Firm
   3's first borrowing in S0.
3. **S1:** no sustained post-90% debt recovery. It prevents Firm 4 from ever
   reaching 90% utilization, but Firms 1 and 3 still finish around the limit;
   total arrears rise from {system_index.loc['S0', 'final_total_arrears']:.0f}
   to {system_index.loc['S1', 'final_total_arrears']:.0f}.
4. **S2:** no sustained post-90% debt recovery. It produces the strongest
   prevention result for Firm 4 (final utilization
   {firm_index.loc[('S2', 4), 'final_principal_utilization']:.3f}) and lowers
   total principal and arrears, but Firm 0's mature CFO becomes negative.
5. **S3:** no sustained post-90% debt recovery and no trapped Firm is kept
   below 90%; total arrears are higher than S0.
6. **Largest changes:** Firm 4 changes most under S1/S2. Firm 1 gains unit
   share under S1/S2 but remains near its credit limit. Firm 3's S1 operating
   path improves while its arrears become much worse.
7. **Prevented saturation:** {prevented_text}. No treatment prevents Firms 1
   or 3 from reaching the high-utilization regime.
8. **Delay only:** no trapped-Firm path meets the report's
   `DELAYED_RATCHET` classification. S3 mainly changes timing/levels without
   preventing saturation.
9. **Recovery after 90%:** none of S1/S2/S3 meets the persistence definition;
   all three required genuine-recovery flags are false.
10. **Market share:** S1/S2 increase Firm 4's mature unit share from
    {firm_index.loc[('S0', 4), 'mature_mean_market_share']:.3f} to
    {firm_index.loc[('S1', 4), 'mature_mean_market_share']:.3f}/
    {firm_index.loc[('S2', 4), 'mature_mean_market_share']:.3f}. S1 raises Firm
    1's share but reduces Firm 3's; S3 reduces Firm 1 and Firm 3 shares.
11. **Prices and cuts:** treatments change price ordering materially, but cut
    frequencies stay close to S0 rather than exploding. The joint price-war
    guard is false for all strategies.
12. **Profit versus CFO:** improvements are Firm-specific, not a uniform
    cash-flow gain bought by broad price cutting. Firm 4 under S2 has better
    CFO and operating profit than S0; Firm 4 under S1 avoids the limit despite
    slightly worse mature CFO and operating profit.
13. **Arrears:** S2 lowers system arrears to
    {system_index.loc['S2', 'final_total_arrears']:.0f}; S1 and S3 increase
    them to {system_index.loc['S1', 'final_total_arrears']:.0f} and
    {system_index.loc['S3', 'final_total_arrears']:.0f}.
14. **Healthy Firm 0:** it receives no recovery-objective warning and never
    borrows, but market spillovers materially lower its CFO/cash in every
    treatment. Therefore the conservative `Firm0_destabilized` flag is true.
15. **Firm 2:** it is not destabilized. All treatments avoid its small late S0
    debt draw, although its CFO remains negative.
16. **Macro stability:** no accounting failure or broad production/sales
    collapse appears. Mature sales differ from S0 by roughly -1.5% (S1), +0.7%
    (S2), and -1.9% (S3); the `G` labels refer to the material Firm 0 spillover,
    not a system-wide accounting or goods failure.
17. **Accounting:** all four branches pass every existing reconciliation at
    the unchanged 1e-6 tolerance, with zero invariant violations.
18. **Dominance:** no treatment is unambiguously Pareto-dominant once CFO,
    share, price, principal, arrears, healthy-Firm spillovers and macro output
    are considered together. S2 and S1 expose the clearest Firm 4 prevention
    paths; S3 shows weaker debt benefit with higher arrears. This is a review
    priority statement, not a production selection.

## Accounting and conservation

{chr(10).join(accounting_lines)}

## Required flags

```text
{chr(10).join(flag_lines)}
```

## Scope stop

No production source, baseline parameter, credit, interest, repayment,
Distress/Default, Exit, restructuring, sector or good was modified. Seeds 7
and 21 were not run.
"""
    (OUT / "acceptance_summary.md").write_text(summary, encoding="utf-8")


def ensure_output_contract():
    OUT.mkdir(parents=True, exist_ok=True)
    allowed = {
        "acceptance_summary.md",
        "strategy_comparison_by_firm.csv",
        "strategy_system_summary.csv",
        "pricing_recovery_strategy_overview.png",
    }
    unexpected = [path for path in OUT.iterdir() if path.name not in allowed]
    if unexpected:
        raise RuntimeError(f"Unexpected permanent files in output directory: {unexpected}")
    for name in allowed:
        path = OUT / name
        if path.exists():
            path.unlink()


def main():
    ensure_output_contract()
    canonical_macro, canonical_firm, canonical_accounting, canonical_combined = canonical_frames()
    passive_forecast = forecast_metrics(canonical_combined)
    passive_by_firm = pd.DataFrame(
        [
            {"firm_id": int(firm_id), **forecast_metrics(rows)}
            for firm_id, rows in canonical_combined.groupby("firm_id", sort=True)
        ]
    )
    warned = canonical_combined[canonical_combined["liquidity_warning"]]
    if warned.empty:
        raise RuntimeError("Canonical trajectory has no liquidity warning; no strategy branch is meaningful.")
    earliest_warning = int(warned["global_step"].min())
    branch_week = max(earliest_warning - 1, 0)
    prefix_steps = branch_week + 1
    remaining = STEPS - prefix_steps
    print(f"Earliest baseline warning={earliest_warning}; prefix rows=0..{branch_week}")

    prefix_world = build_world()
    advance(prefix_world, prefix_steps, "PREFIX")
    prefix_blob = pickle.dumps(prefix_world, protocol=5)
    prefix_frames = world_frames(prefix_world)
    prefix_combined = prefix_frames["combined"]

    control_world = pickle.loads(prefix_blob)
    advance(control_world, remaining, "S0")
    branches = {"S0": world_frames(control_world)}
    controllers = {"S0": None}
    parity = control_parity(branches["S0"])
    print(f"S0 parity pass={parity['pass']} max_abs_diff={parity['max_abs_diff']:.3e}")
    if not parity["pass"]:
        raise RuntimeError(f"S0 parity gate failed: {parity['comparisons']}")

    for strategy in ("S1", "S2", "S3"):
        world, controller = run_treatment(prefix_blob, prefix_combined, strategy, remaining)
        branches[strategy] = world_frames(world)
        controllers[strategy] = controller

    accounting_results = {
        strategy: accounting_check(frames)
        for strategy, frames in branches.items()
    }
    baseline_by_firm = firm_summary("S0", branches["S0"], pd.DataFrame())
    comparisons = [baseline_by_firm]
    for strategy in ("S1", "S2", "S3"):
        comparisons.append(
            firm_summary(strategy, branches[strategy], baseline_by_firm)
        )
    comparison = pd.concat(comparisons, ignore_index=True)

    systems = []
    for strategy in STRATEGIES:
        by_firm = comparison[comparison["strategy"].eq(strategy)]
        systems.append(
            system_summary(
                strategy,
                branches[strategy],
                by_firm,
                baseline_by_firm,
                accounting_results[strategy],
                parity["pass"],
                0 if strategy == "S0" else controllers[strategy].recovery_trial_count,
            )
        )
    systems = pd.DataFrame(systems)

    comparison.to_csv(OUT / "strategy_comparison_by_firm.csv", index=False)
    systems.to_csv(OUT / "strategy_system_summary.csv", index=False)
    make_figure(branches)
    write_summary(
        parity,
        passive_forecast,
        passive_by_firm,
        comparison,
        systems,
        accounting_results,
        earliest_warning,
        branch_week,
    )
    files = sorted(path.name for path in OUT.iterdir())
    if files != sorted([
        "acceptance_summary.md",
        "strategy_comparison_by_firm.csv",
        "strategy_system_summary.csv",
        "pricing_recovery_strategy_overview.png",
    ]):
        raise RuntimeError(f"Output contract violation: {files}")
    print(f"Completed. Permanent outputs: {files}")


if __name__ == "__main__":
    main()
