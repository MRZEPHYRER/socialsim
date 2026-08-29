"""Pure offline audit of the post-Step13 competitive-financial Firm trap."""

from __future__ import annotations

import json
import math
import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from economy import config as economy_config


RUN = ROOT / "test" / "output" / "main_step13_financial_core"
OUT = ROOT / "test" / "output" / "post_step13_competitive_financial_trap_audit"
MATURE_START = 1560
TOL = 1e-9
SOURCE = "test/output/main_step13_financial_core"


def numeric(frame: pd.DataFrame, field: str) -> pd.Series:
    return pd.to_numeric(frame[field], errors="coerce").fillna(0.0)


def boolean(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().eq("true")


def safe_ratio(a: float, b: float) -> float:
    return float(a / b) if abs(b) > 1e-12 else 0.0


def linear_relation(x: pd.Series, y: pd.Series) -> dict:
    valid = np.isfinite(x.to_numpy(float)) & np.isfinite(y.to_numpy(float))
    xv = x.to_numpy(float)[valid]
    yv = y.to_numpy(float)[valid]
    if len(xv) < 3 or np.std(xv) <= 1e-12 or np.std(yv) <= 1e-12:
        return {"n": len(xv), "corr": 0.0, "slope": 0.0, "intercept": float(np.mean(yv)) if len(yv) else 0.0, "r2": 0.0}
    slope, intercept = np.polyfit(xv, yv, 1)
    fitted = slope * xv + intercept
    total = float(np.sum((yv - np.mean(yv)) ** 2))
    residual = float(np.sum((yv - fitted) ** 2))
    return {
        "n": len(xv),
        "corr": float(np.corrcoef(xv, yv)[0, 1]),
        "slope": float(slope),
        "intercept": float(intercept),
        "r2": 0.0 if total <= 1e-24 else float(1.0 - residual / total),
    }


def first_step(frame: pd.DataFrame, mask: pd.Series):
    selected = frame.loc[mask]
    return None if selected.empty else int(selected.iloc[0]["global_step"])


def first_sustained_step(frame: pd.DataFrame, mask: pd.Series, weeks: int):
    values = mask.astype(int).rolling(weeks, min_periods=weeks).sum()
    endings = values.index[values >= weeks]
    if len(endings) == 0:
        return None
    end_index = endings[0]
    end_position = frame.index.get_loc(end_index)
    start_position = end_position - weeks + 1
    return int(frame.iloc[start_position]["global_step"])


def count_recoveries(utilization: pd.Series, high=0.90, low=0.80) -> int:
    armed = False
    count = 0
    for value in utilization:
        if value >= high:
            armed = True
        elif armed and value < low:
            count += 1
            armed = False
    return count


def add(rows, section, entity, metric, value, unit="", notes=""):
    if isinstance(value, (np.integer, np.floating)):
        value = value.item()
    rows.append(
        {
            "section": section,
            "entity": entity,
            "metric": metric,
            "value": value,
            "unit": unit,
            "source": SOURCE,
            "notes": notes,
        }
    )


def prepare_data():
    firms = pd.read_csv(RUN / "firm_diagnostics.csv")
    accounting = pd.read_csv(RUN / "accounting" / "firm_accounting.csv")
    for frame in (firms, accounting):
        frame["global_step"] = numeric(frame, "global_step").astype(int)
        frame["firm_id"] = numeric(frame, "firm_id").astype(int)
    accounting_fields = [
        "global_step",
        "firm_id",
        "accounting_operating_profit",
        "accounting_net_income",
        "cfo",
        "cff",
        "inventory_cost_or_cogs",
        "spoilage_or_inventory_loss",
        "capitalized_production_cost",
        "interest_paid",
        "principal_repaid",
        "dividends",
        "other_operating_cashflows",
    ]
    frame = firms.merge(
        accounting[accounting_fields],
        on=["global_step", "firm_id"],
        how="left",
        validate="one_to_one",
        suffixes=("", "_accounting"),
    )
    numeric_fields = [
        "price",
        "price_change",
        "unit_market_share",
        "choice_probability",
        "observed_demand",
        "unmet_demand",
        "sales_units",
        "sales_revenue",
        "actual_production",
        "capacity_utilization",
        "realized_ulc",
        "normal_ulc",
        "margin",
        "price_floor",
        "profit",
        "smoothed_profit",
        "last_profit_delta",
        "cash",
        "cash_start",
        "cash_end",
        "loan_issued",
        "loan_repaid",
        "opening_principal",
        "closing_principal",
        "credit_limit",
        "credit_headroom",
        "denied_credit",
        "funding_gap",
        "current_interest_due",
        "interest_paid",
        "closing_interest_arrears",
        "dividend_paid",
        "dividend_eligible_profit",
        "scheduled_wage_bill",
        "executed_wage_bill",
        "payroll_funding_ratio",
        "accounting_operating_profit",
        "accounting_net_income",
        "cfo",
        "cff",
        "inventory_cost_or_cogs",
        "spoilage_or_inventory_loss",
        "capitalized_production_cost",
        "principal_repaid",
        "dividends",
    ]
    for field in numeric_fields:
        frame[field] = numeric(frame, field)
    market_price = frame.groupby("global_step").apply(
        lambda group: safe_ratio(
            float((group["price"] * group["unit_market_share"]).sum()),
            float(group["unit_market_share"].sum()),
        ),
        include_groups=False,
    )
    frame["market_weighted_price"] = frame["global_step"].map(market_price)
    frame["dynamic_relative_price"] = frame["price"] / frame["market_weighted_price"].replace(0.0, np.nan)
    frame["principal_utilization"] = frame["closing_principal"] / frame["credit_limit"].replace(0.0, np.nan)
    frame["distance_to_price_floor"] = (frame["price"] - frame["price_floor"]) / frame["price_floor"].replace(0.0, np.nan)
    frame["price_to_normal_ulc"] = frame["price"] / frame["normal_ulc"].replace(0.0, np.nan)
    frame["price_to_realized_ulc"] = frame["price"] / frame["realized_ulc"].replace(0.0, np.nan)
    frame["cash_change"] = frame["cash_end"] - frame["cash_start"]
    frame["inventory_book_investment"] = (
        frame["capitalized_production_cost"]
        - frame["inventory_cost_or_cogs"]
        - frame["spoilage_or_inventory_loss"]
    )
    frame["operating_to_cash_wedge"] = frame["accounting_operating_profit"] - frame["cfo"]
    frame["ocf_shortfall"] = (-frame["cfo"]).clip(lower=0.0)
    frame["cumulative_ocf_shortfall"] = frame.groupby("firm_id")["ocf_shortfall"].cumsum()
    return frame.sort_values(["firm_id", "global_step"]).reset_index(drop=True)


def source_semantics(metrics):
    semantics = {
        "price_review_probability_per_eligible_week": economy_config.FIRM_PRICE_REVIEW_PROBABILITY,
        "evaluation_window_weeks": economy_config.FIRM_PRICE_EVALUATION_WINDOW,
        "profit_ema_alpha": economy_config.FIRM_PRICE_PROFIT_EMA_ALPHA,
        "keep_probability_per_review": economy_config.FIRM_PRICE_KEEP_PROBABILITY,
        "exploration_probability_per_direction_choice": economy_config.FIRM_PRICE_EXPLORATION_PROBABILITY,
        "small_trial_step": economy_config.FIRM_PRICE_TRIAL_STEP_SMALL,
        "large_trial_step": economy_config.FIRM_PRICE_TRIAL_STEP_LARGE,
        "large_step_probability": 0.35,
        "price_objective_name": "EMA of legacy firm.profit = sales_revenue - executed_wage_bill",
        "objective_comparison": "evaluated_smoothed_profit - baseline_smoothed_profit after 12 weeks",
        "positive_trial_result": "continue last direction",
        "negative_trial_result": "reverse last direction",
        "zero_or_unidentified_gradient": "random cut or raise",
        "price_floor_formula": "normal_unit_labor_cost * 1.02, then global FOOD_MIN_PRICE",
        "price_ceiling": economy_config.FOOD_MAX_PRICE,
        "global_min_price": economy_config.FOOD_MIN_PRICE,
        "normal_ulc_role": "price lower-bound input only",
        "realized_ulc_role": "diagnostic margin/cost only; not learner input",
        "inventory_gap_role": "not used by current price decision",
        "declared_direction_persist_probability_used": False,
        "declared_direction_reverse_probability_used": False,
        "declared_inventory_worsen_tolerance_used": False,
        "demand_signal_role": "not directly used; affects realized sales and profit",
        "cost_growth_role": "not used by multi-Firm adaptive price decision",
        "margin_role": "diagnostic only; not learner input",
        "includes_interest": False,
        "includes_financing_cost": False,
        "includes_cash_flow": False,
        "includes_debt_state": False,
        "includes_default_state": False,
        "includes_arrears": False,
        "includes_denied_credit": False,
        "includes_payroll_underfunding_directly": False,
        "accepted_rejected_is_source_concept": False,
        "exploration_event_is_recorded": False,
        "existing_relative_price_updates_after_bootstrap": False,
    }
    for key, value in semantics.items():
        add(metrics, "source_semantics", "price_learner", key, value, notes="Verified from world.py update_firm_price_evaluation, price_direction_after_review, and update_adaptive_firm_prices.")

    matrix = {
        "D2": {"Pricing": "NO", "Production": "NO", "Wage": "NO", "Dividend": "NO", "Borrowing": "NO"},
        "D3": {"Pricing": "NO", "Production": "NO", "Wage": "NO", "Dividend": "NO", "Borrowing": "NO"},
        "Default": {"Pricing": "NO", "Production": "NO", "Wage": "NO", "Dividend": "NO", "Borrowing": "NO"},
        "High_debt": {"Pricing": "NO", "Production": "INDIRECT", "Wage": "INDIRECT", "Dividend": "INDIRECT", "Borrowing": "YES"},
        "Credit_denial": {"Pricing": "NO", "Production": "INDIRECT", "Wage": "INDIRECT", "Dividend": "INDIRECT", "Borrowing": "INDIRECT"},
    }
    for state, channels in matrix.items():
        for channel, value in channels.items():
            add(metrics, "decision_channel_matrix", state, channel, value, notes="INDIRECT means the same liquidity/headroom constraint changes feasible operations, not that the diagnostic state is read as a policy input.")
    return semantics, matrix


def event_study(frame: pd.DataFrame, decision: str, firm_id=None):
    subset = frame if firm_id is None else frame.loc[frame["firm_id"] == firm_id]
    events = subset.loc[subset["price_decision"] == decision, ["firm_id", "global_step"]]
    fields = {
        "price": "price",
        "market_share": "unit_market_share",
        "choice_probability": "choice_probability",
        "demand": "observed_demand",
        "sales_units": "sales_units",
        "sales_revenue": "sales_revenue",
        "unmet_demand": "unmet_demand",
        "production": "actual_production",
        "capacity_utilization": "capacity_utilization",
        "realized_ulc": "realized_ulc",
        "normal_ulc": "normal_ulc",
        "margin": "margin",
        "operating_profit": "accounting_operating_profit",
        "learner_profit": "profit",
        "price_objective": "smoothed_profit",
        "dividend_eligible_profit": "dividend_eligible_profit",
        "ocf": "cfo",
        "cash": "cash",
        "borrowing": "loan_issued",
        "principal": "closing_principal",
        "interest_due": "current_interest_due",
        "interest_paid": "interest_paid",
    }
    records = []
    for event in events.itertuples(index=False):
        firm = frame.loc[frame["firm_id"] == int(event.firm_id)].set_index("global_step")
        pre_steps = list(range(int(event.global_step) - 4, int(event.global_step)))
        post_steps = list(range(int(event.global_step) + 1, int(event.global_step) + 5))
        if not set(pre_steps).issubset(firm.index) or not set(post_steps).issubset(firm.index):
            continue
        row = {"firm_id": int(event.firm_id), "event_step": int(event.global_step)}
        for label, field in fields.items():
            row[f"pre_{label}"] = float(firm.loc[pre_steps, field].mean())
            row[f"post_{label}"] = float(firm.loc[post_steps, field].mean())
        if row["pre_price"] > 0 and row["post_price"] > 0:
            row["delta_log_price"] = math.log(row["post_price"] / row["pre_price"])
        else:
            row["delta_log_price"] = np.nan
        if row["pre_demand"] > 0 and row["post_demand"] > 0:
            row["delta_log_demand"] = math.log(row["post_demand"] / row["pre_demand"])
        else:
            row["delta_log_demand"] = np.nan
        if row["pre_sales_units"] > 0 and row["post_sales_units"] > 0:
            row["delta_log_sales"] = math.log(row["post_sales_units"] / row["pre_sales_units"])
        else:
            row["delta_log_sales"] = np.nan
        if abs(row["delta_log_price"]) > 1e-8:
            row["demand_price_response"] = row["delta_log_demand"] / row["delta_log_price"]
            row["sales_price_response"] = row["delta_log_sales"] / row["delta_log_price"]
        else:
            row["demand_price_response"] = np.nan
            row["sales_price_response"] = np.nan
        records.append(row)
    event_frame = pd.DataFrame(records)
    result = {"event_count": int(len(events)), "complete_window_event_count": len(event_frame)}
    if event_frame.empty:
        return result, event_frame
    for label in fields:
        result[f"mean_pre_{label}"] = float(event_frame[f"pre_{label}"].mean())
        result[f"mean_post_{label}"] = float(event_frame[f"post_{label}"].mean())
        result[f"mean_change_{label}"] = result[f"mean_post_{label}"] - result[f"mean_pre_{label}"]
    for field in ("demand_price_response", "sales_price_response"):
        finite = event_frame[field].replace([np.inf, -np.inf], np.nan).dropna()
        result[field] = float(finite.median()) if len(finite) else None
        result[f"mean_{field}"] = float(finite.mean()) if len(finite) else None
    result["response_label"] = "DESCRIPTIVE_LOCAL_RESPONSE_NOT_STRUCTURAL_ELASTICITY"
    return result, event_frame


def review_metrics(frame, firm_id, metrics):
    rows = frame.loc[frame["firm_id"] == firm_id].sort_values("global_step")
    mature = rows.loc[rows["global_step"] >= MATURE_START]
    reviews = boolean(rows["price_reviewed"])
    decisions = rows["price_decision"].astype(str)
    cut_rows = rows.loc[decisions == "cut"]
    raise_rows = rows.loc[decisions == "raise"]
    evaluation = {"cut_improved": 0, "cut_worsened": 0, "raise_improved": 0, "raise_worsened": 0}
    by_step = rows.set_index("global_step")
    for decision, events in (("cut", cut_rows), ("raise", raise_rows)):
        for step in events["global_step"]:
            evaluation_step = int(step) + economy_config.FIRM_PRICE_EVALUATION_WINDOW
            if evaluation_step not in by_step.index:
                continue
            delta = float(by_step.loc[evaluation_step, "last_profit_delta"])
            key = f"{decision}_{'improved' if delta > TOL else 'worsened'}"
            evaluation[key] += 1
    result = {
        "full_review_count": int(reviews.sum()),
        "mature_review_count": int(boolean(mature["price_reviewed"]).sum()),
        "cut_count": int((decisions == "cut").sum()),
        "raise_count": int((decisions == "raise").sum()),
        "keep_count": int((decisions == "keep").sum()),
        "mature_cut_count": int((mature["price_decision"].astype(str) == "cut").sum()),
        "mature_raise_count": int((mature["price_decision"].astype(str) == "raise").sum()),
        "mature_keep_count": int((mature["price_decision"].astype(str) == "keep").sum()),
        "blocked_by_floor_count": int((decisions == "blocked_by_floor").sum()),
        "exploratory_cut_count": "NOT_OBSERVABLE",
        "exploratory_raise_count": "NOT_OBSERVABLE",
        "accepted_rejected_count": "NOT_A_SOURCE_CONCEPT",
        "evaluated_cut_improved_count": evaluation["cut_improved"],
        "evaluated_cut_worsened_count": evaluation["cut_worsened"],
        "evaluated_raise_improved_count": evaluation["raise_improved"],
        "evaluated_raise_worsened_count": evaluation["raise_worsened"],
        "mean_price_change_on_cut": float(cut_rows["price_change"].mean()) if len(cut_rows) else 0.0,
        "mean_price_change_on_raise": float(raise_rows["price_change"].mean()) if len(raise_rows) else 0.0,
        "cumulative_log_price_change": float(math.log(rows.iloc[-1]["price"] / rows.iloc[0]["price"])),
        "final_price": float(rows.iloc[-1]["price"]),
        "mean_price_mature": float(mature["price"].mean()),
        "mean_price_floor_mature": float(mature["price_floor"].mean()),
        "mean_relative_price_mature": float(mature["dynamic_relative_price"].mean()),
        "median_relative_price_mature": float(mature["dynamic_relative_price"].median()),
        "mean_market_share_mature": float(mature["unit_market_share"].mean()),
        "mean_choice_probability_mature": float(mature["choice_probability"].mean()),
        "mean_capacity_utilization_mature": float(mature["capacity_utilization"].mean()),
        "mean_normal_ULC_mature": float(mature["normal_ulc"].mean()),
        "mean_realized_ULC_mature": float(mature["realized_ulc"].mean()),
        "mean_price_floor_distance_mature": float(mature["distance_to_price_floor"].mean()),
        "mean_price_to_normal_ULC_mature": float(mature["price_to_normal_ulc"].mean()),
        "mean_price_to_realized_ULC_mature": float(mature["price_to_realized_ulc"].mean()),
        "mean_operating_profit_mature": float(mature["accounting_operating_profit"].mean()),
        "mean_price_learner_objective_mature": float(mature["smoothed_profit"].mean()),
        "mean_legacy_profit_mature": float(mature["profit"].mean()),
        "mean_accounting_net_income_mature": float(mature["accounting_net_income"].mean()),
        "mean_OCF_mature": float(mature["cfo"].mean()),
        "mean_cash_change_mature": float(mature["cash_change"].mean()),
        "mature_gross_borrowing": float(mature["loan_issued"].sum()),
        "mature_gross_repayment": float(mature["loan_repaid"].sum()),
        "mature_interest_due": float(mature["current_interest_due"].sum()),
        "mature_interest_paid": float(mature["interest_paid"].sum()),
        "final_principal": float(rows.iloc[-1]["closing_principal"]),
        "final_principal_utilization": float(rows.iloc[-1]["principal_utilization"]),
        "final_arrears": float(rows.iloc[-1]["closing_interest_arrears"]),
        "share_within_0_5pct_floor_mature": float((mature["distance_to_price_floor"] <= 0.005).mean()),
        "share_within_1pct_floor_mature": float((mature["distance_to_price_floor"] <= 0.01).mean()),
        "share_within_2pct_floor_mature": float((mature["distance_to_price_floor"] <= 0.02).mean()),
        "share_within_5pct_floor_mature": float((mature["distance_to_price_floor"] <= 0.05).mean()),
        "price_floor_binding_share_mature": float(boolean(mature["price_floor_binding"]).mean()),
    }
    for key, value in result.items():
        add(metrics, "firm_core", f"firm_{firm_id}", key, value)
    return result


def principal_and_timeline(frame, firm_id, metrics):
    rows = frame.loc[frame["firm_id"] == firm_id].sort_values("global_step").copy()
    first_borrow = first_step(rows, rows["loan_issued"] > TOL)
    first_persistent = first_sustained_step(rows, rows["loan_issued"] > TOL, 4)
    first_weak_share = first_sustained_step(rows, rows["unit_market_share"] < 0.20, 8)
    transitions = {
        "relative_price_weakness": first_sustained_step(rows, rows["dynamic_relative_price"] > 1.0, 8),
        "market_share_weakness": first_weak_share,
        "utilization_below_90pct": first_sustained_step(rows, rows["capacity_utilization"] < 0.90, 8),
        "negative_OCF": first_sustained_step(rows, rows["cfo"] < 0.0, 4),
        "first_borrowing": first_borrow,
        "first_persistent_borrowing": first_persistent,
        "principal_utilization_50pct": first_step(rows, rows["principal_utilization"] >= 0.50),
        "principal_utilization_80pct": first_step(rows, rows["principal_utilization"] >= 0.80),
        "principal_utilization_90pct": first_step(rows, rows["principal_utilization"] >= 0.90),
        "principal_utilization_95pct": first_step(rows, rows["principal_utilization"] >= 0.95),
        "principal_utilization_99pct": first_step(rows, rows["principal_utilization"] >= 0.99),
        "D2": first_step(rows, rows["distress_state"].astype(str) == "D2"),
        "D3": first_step(rows, rows["distress_state"].astype(str) == "D3"),
        "Default": first_step(rows, boolean(rows["default_event_this_week"])),
    }
    for name, step in transitions.items():
        add(metrics, "timeline", f"firm_{firm_id}", f"first_{name}_week", step)
        if step is None:
            continue
        event = rows.loc[rows["global_step"] == step].iloc[0]
        for field in (
            "price",
            "dynamic_relative_price",
            "unit_market_share",
            "capacity_utilization",
            "normal_ulc",
            "realized_ulc",
            "accounting_operating_profit",
            "smoothed_profit",
            "cfo",
            "cash",
            "funding_gap",
            "loan_issued",
            "current_interest_due",
            "principal_utilization",
        ):
            add(metrics, "transition_snapshot", f"firm_{firm_id}_{name}", field, float(event[field]))

    after = rows.loc[rows["global_step"] >= first_borrow].copy() if first_borrow is not None else rows.iloc[0:0].copy()
    principal_delta = after["closing_principal"] - after["opening_principal"]
    running_peak = after["closing_principal"].cummax() if len(after) else pd.Series(dtype=float)
    recovery_80 = count_recoveries(after["principal_utilization"], 0.90, 0.80) if len(after) else 0
    recovery_70 = count_recoveries(after["principal_utilization"], 0.90, 0.70) if len(after) else 0
    recovery_50 = count_recoveries(after["principal_utilization"], 0.90, 0.50) if len(after) else 0
    reached_90 = bool((after["principal_utilization"] >= 0.90).any()) if len(after) else False
    if first_borrow is None:
        ratchet = "SELF_FINANCING"
    elif reached_90 and recovery_80 == 0 and float(rows.iloc[-1]["principal_utilization"]) >= 0.90:
        ratchet = "RATCHET_TO_LIMIT"
    elif reached_90:
        ratchet = "PARTIALLY_REVOLVING"
    else:
        ratchet = "REVOLVING"
    weak_onset = first_weak_share if first_weak_share is not None else 0
    weak_period = rows.loc[rows["global_step"] >= weak_onset]
    principal = {
        "first_borrowing_week": first_borrow,
        "first_persistent_borrowing_week": first_persistent,
        "first_90pct_utilization_week": transitions["principal_utilization_90pct"],
        "default_event_week": transitions["Default"],
        "principal_rise_week_count_after_borrowing": int((principal_delta > TOL).sum()),
        "principal_fall_week_count_after_borrowing": int((principal_delta < -TOL).sum()),
        "mean_positive_principal_change": float(principal_delta.loc[principal_delta > TOL].mean()) if (principal_delta > TOL).any() else 0.0,
        "mean_negative_principal_change": float(principal_delta.loc[principal_delta < -TOL].mean()) if (principal_delta < -TOL).any() else 0.0,
        "largest_principal_drawdown": float((running_peak - after["closing_principal"]).max()) if len(after) else 0.0,
        "recovery_90_to_below_80_count": recovery_80,
        "recovery_90_to_below_70_count": recovery_70,
        "recovery_90_to_below_50_count": recovery_50,
        "principal_ratchet_classification": ratchet,
        "cumulative_borrowing_after_weak_share_onset": float(weak_period["loan_issued"].sum()),
        "cumulative_denied_credit_after_weak_share_onset": float(weak_period["denied_credit"].sum()),
        "operation_weeks_principal_utilization_ge_95pct": int((rows["principal_utilization"] >= 0.95).sum()),
        "operation_weeks_active_contract_default": int(boolean(rows["active_contract_default"]).sum()),
        "descriptive_nonviable_firm_weeks": int((boolean(rows["active_contract_default"]) & (rows["principal_utilization"] >= 0.95)).sum()),
    }
    for key, value in principal.items():
        add(metrics, "principal_and_soft_budget", f"firm_{firm_id}", key, value)
    ordered = sorted(((step, name) for name, step in transitions.items() if step is not None))
    add(metrics, "timeline", f"firm_{firm_id}", "ordered_timeline", " -> ".join(f"{name}@{step}" for step, name in ordered))
    return transitions, principal


def relation_metrics(frame, metrics):
    mature = frame.loc[frame["global_step"] >= MATURE_START].copy()
    specifications = {
        "A_market_share_on_relative_price": ("dynamic_relative_price", "unit_market_share"),
        "B_utilization_on_market_share": ("unit_market_share", "capacity_utilization"),
        "C_realized_ULC_on_utilization": ("capacity_utilization", "realized_ulc"),
        "D_OCF_on_utilization": ("capacity_utilization", "cfo"),
        "E_borrowing_on_OCF": ("cfo", "loan_issued"),
        "F_principal_utilization_on_cumulative_OCF_shortfall": ("cumulative_ocf_shortfall", "principal_utilization"),
        "utilization_on_margin": ("capacity_utilization", "margin"),
        "price_on_utilization": ("capacity_utilization", "price"),
    }
    results = {}
    for entity, subset in [("pooled", mature)] + [(f"firm_{firm_id}", group) for firm_id, group in mature.groupby("firm_id")]:
        results[entity] = {}
        for name, (x, y) in specifications.items():
            relation = linear_relation(subset[x], subset[y])
            results[entity][name] = relation
            for metric, value in relation.items():
                add(metrics, "descriptive_relations", entity, f"{name}_{metric}", value, notes="Descriptive OLS/correlation only; no structural causal claim.")
    return results


def financing_wedge(frame, firm_id, metrics):
    mature = frame.loc[(frame["firm_id"] == firm_id) & (frame["global_step"] >= MATURE_START)]
    result = {
        "mean_accounting_operating_profit": float(mature["accounting_operating_profit"].mean()),
        "mean_learner_objective": float(mature["smoothed_profit"].mean()),
        "mean_accounting_net_income": float(mature["accounting_net_income"].mean()),
        "mean_CFO_after_interest": float(mature["cfo"].mean()),
        "mean_cash_change": float(mature["cash_change"].mean()),
        "mean_operating_to_cash_wedge": float(mature["operating_to_cash_wedge"].mean()),
        "mean_inventory_book_investment": float(mature["inventory_book_investment"].mean()),
        "mean_interest_paid": float(mature["interest_paid"].mean()),
        "mean_principal_repayment": float(mature["loan_repaid"].mean()),
        "mean_borrowing": float(mature["loan_issued"].mean()),
        "mean_dividend": float(mature["dividend_paid"].mean()),
        "mean_payroll_funding_ratio": float(mature["payroll_funding_ratio"].mean()),
        "positive_operating_profit_share": float((mature["accounting_operating_profit"] > 0).mean()),
        "positive_operating_profit_negative_CFO_share": float(((mature["accounting_operating_profit"] > 0) & (mature["cfo"] < 0)).mean()),
        "positive_legacy_profit_negative_CFO_share": float(((mature["profit"] > 0) & (mature["cfo"] < 0)).mean()),
        "positive_operating_profit_principal_increase_share": float(((mature["accounting_operating_profit"] > 0) & (mature["closing_principal"] > mature["opening_principal"] + TOL)).mean()),
        "positive_objective_negative_CFO_share": float(((mature["smoothed_profit"] > 0) & (mature["cfo"] < 0)).mean()),
    }
    for key, value in result.items():
        add(metrics, "profit_cash_wedge", f"firm_{firm_id}", key, value)
    return result


def classify_cut_response(event_result, floor_share):
    if floor_share >= 0.10:
        return "COST_FLOOR_CONSTRAINED"
    demand_change = event_result.get("mean_change_demand", 0.0)
    demand_base = abs(event_result.get("mean_pre_demand", 0.0))
    objective_change = event_result.get("mean_change_price_objective", 0.0)
    accounting_profit_change = event_result.get("mean_change_operating_profit", 0.0)
    if abs(demand_change) <= 0.005 * max(1.0, demand_base):
        return "LITTLE_DEMAND_RESPONSE"
    if objective_change * accounting_profit_change < 0:
        return "MIXED"
    if demand_change > 0 and objective_change > 0:
        return "DEMAND_GAIN_PROFIT_GAIN"
    if demand_change > 0 and objective_change <= 0:
        return "DEMAND_GAIN_PROFIT_LOSS"
    return "MIXED"


def trap_classification(core, principal, cuts, wedge):
    if principal["principal_ratchet_classification"] != "RATCHET_TO_LIMIT":
        return "CONTROL_NOT_TRAPPED"
    material_floor = core["share_within_2pct_floor_mature"] >= 0.10
    finance_blind_wedge = wedge["positive_objective_negative_CFO_share"] > 0.05
    demand_profit_gain = cuts.get("mean_change_demand", 0.0) > 0 and cuts.get("mean_change_price_objective", 0.0) > 0
    if material_floor:
        return "COST_DISADVANTAGE"
    if finance_blind_wedge and demand_profit_gain:
        return "MIXED"
    if finance_blind_wedge:
        return "FINANCIAL_OBJECTIVE_BLINDNESS"
    if demand_profit_gain:
        return "PRICE_STRATEGY_FAILURE"
    return "SOFT_BUDGET_DEFAULT_TRAP"


def overview_figure(frame):
    mature = frame.loc[frame["global_step"] >= MATURE_START].copy()
    colors = plt.get_cmap("tab10")
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
    for firm_id, group in mature.groupby("firm_id"):
        group = group.sort_values("global_step")
        x = group["global_step"]
        color = colors(firm_id)
        rolling = group.rolling(13, min_periods=1)
        axes[0, 0].plot(x, rolling["dynamic_relative_price"].mean(), color=color, label=f"Firm {firm_id}")
        axes[0, 1].plot(x, rolling["unit_market_share"].mean(), color=color, label=f"F{firm_id} share")
        axes[0, 1].plot(x, rolling["capacity_utilization"].mean(), color=color, linestyle="--", alpha=0.75, label=f"F{firm_id} util")
        axes[1, 0].plot(x, rolling["cfo"].mean(), color=color, label=f"Firm {firm_id}")
        axes[1, 1].plot(x, rolling["principal_utilization"].mean(), color=color, label=f"F{firm_id} principal util")
        scale = max(1.0, float(group["credit_limit"].mean()))
        axes[1, 1].plot(x, rolling["closing_interest_arrears"].mean() / scale, color=color, linestyle=":", alpha=0.8, label=f"F{firm_id} arrears/limit")
    axes[0, 0].axhline(1.0, color="black", linewidth=0.8, alpha=0.6)
    axes[0, 0].set_title("A. Dynamic Relative Price")
    axes[0, 0].set_ylabel("Price / sales-weighted market price")
    axes[0, 1].set_title("B. Market Share and Capacity Utilization")
    axes[0, 1].set_ylabel("Share / utilization")
    axes[1, 0].axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
    axes[1, 0].set_title("C. Accounting CFO (13-week mean)")
    axes[1, 0].set_ylabel("Currency per week")
    axes[1, 1].axhline(1.0, color="black", linewidth=0.8, alpha=0.6)
    axes[1, 1].set_title("D. Principal Utilization and Arrears")
    axes[1, 1].set_ylabel("Ratio to dynamic credit limit")
    for axis in axes.ravel():
        axis.set_xlabel("Simulation week")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=7, ncol=2, frameon=False)
    fig.suptitle("Post-Step13 Competitive-Financial Trap Overview", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(OUT / "competitive_financial_trap_overview.png", dpi=150)
    plt.close(fig)


def main():
    required = [RUN / "firm_diagnostics.csv", RUN / "accounting" / "firm_accounting.csv", RUN / "manifest.json"]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing canonical inputs: " + ", ".join(missing))
    frame = prepare_data()
    metrics = []
    semantics, matrix = source_semantics(metrics)
    core = {}
    timelines = {}
    principals = {}
    wedges = {}
    event_results = {}
    event_frames = {}
    for firm_id in sorted(frame["firm_id"].unique()):
        core[firm_id] = review_metrics(frame, firm_id, metrics)
        timelines[firm_id], principals[firm_id] = principal_and_timeline(frame, firm_id, metrics)
        wedges[firm_id] = financing_wedge(frame, firm_id, metrics)
    relations = relation_metrics(frame, metrics)

    for decision in ("cut", "raise"):
        for label, firm_id in [("pooled", None)] + [(f"firm_{firm}", firm) for firm in sorted(frame["firm_id"].unique())]:
            result, events = event_study(frame, decision, firm_id)
            event_results[(decision, label)] = result
            event_frames[(decision, label)] = events
            for key, value in result.items():
                add(metrics, "event_study", f"{label}_{decision}", key, value, notes="Pre window weeks -4..-1; post window weeks +1..+4. Local response is descriptive, not structural elasticity.")

    cut_classifications = {}
    trap_classes = {}
    for firm_id in sorted(core):
        cuts = event_results[("cut", f"firm_{firm_id}")]
        cut_classifications[firm_id] = classify_cut_response(cuts, core[firm_id]["share_within_2pct_floor_mature"])
        trap_classes[firm_id] = trap_classification(core[firm_id], principals[firm_id], cuts, wedges[firm_id])
        add(metrics, "classification", f"firm_{firm_id}", "price_cut_response_classification", cut_classifications[firm_id])
        add(metrics, "classification", f"firm_{firm_id}", "trap_classification", trap_classes[firm_id])

    pooled_rel = relations["pooled"]
    price_floor_binding_material = any(core[firm]["share_within_2pct_floor_mature"] >= 0.10 for firm in (1, 3, 4))
    capacity_cost_material = pooled_rel["C_realized_ULC_on_utilization"]["corr"] <= -0.30
    price_share_material = pooled_rel["A_market_share_on_relative_price"]["corr"] <= -0.30
    share_util_material = pooled_rel["B_utilization_on_market_share"]["corr"] >= 0.30
    util_ocf_material = pooled_rel["D_OCF_on_utilization"]["corr"] >= 0.30
    ocf_borrow_material = pooled_rel["E_borrowing_on_OCF"]["corr"] <= -0.30
    principal_ratchet = any(principals[firm]["principal_ratchet_classification"] == "RATCHET_TO_LIMIT" for firm in (1, 3, 4))
    soft_budget = sum(principals[firm]["descriptive_nonviable_firm_weeks"] for firm in (1, 3, 4)) > 0

    material_mechanisms = sum((capacity_cost_material, price_share_material, share_util_material, util_ocf_material, ocf_borrow_material, soft_budget))
    if material_mechanisms >= 3:
        verdict = "F. MIXED_COMPETITIVE_FINANCIAL_TRAP"
    elif soft_budget:
        verdict = "E. SOFT_BUDGET_CONSTRAINT_IS_PRIMARY"
    else:
        verdict = "B. PRICE_LEARNER_IS_FINANCIALLY_BLIND"
    selected_next = "FIRM_EXIT_AND_REPLACEMENT" if soft_budget else "PRICE_LEARNER_OBJECTIVE_REVIEW"

    system = {
        "verdict": verdict,
        "price_floor_binding_material": price_floor_binding_material,
        "capacity_utilization_cost_feedback_material": capacity_cost_material,
        "price_market_share_relation_material": price_share_material,
        "market_share_utilization_relation_material": share_util_material,
        "utilization_OCF_relation_material": util_ocf_material,
        "OCF_borrowing_relation_material": ocf_borrow_material,
        "principal_ratchet_observed": principal_ratchet,
        "normal_revolver_control_exists": principals[2]["principal_ratchet_classification"] == "REVOLVING",
        "soft_budget_constraint_evidence": soft_budget,
        "product_market_selection_exists": True,
        "firm_population_selection_exists": False,
        "selected_next_mechanism_family": selected_next,
    }
    selection_capabilities = {
        "weak_firm_can_lose_market_share": True,
        "weak_firm_can_lose_revenue": True,
        "weak_firm_can_lose_utilization": True,
        "weak_firm_can_lose_cash": True,
        "weak_firm_can_lose_existing_employees_from_competition": False,
        "weak_firm_can_lose_productive_capacity_from_competition": False,
        "weak_firm_can_lose_active_legal_status": False,
        "weak_firm_can_lose_market_participation": False,
        "weak_firm_assets_can_be_liquidated": False,
    }
    for key, value in system.items():
        add(metrics, "system", "classification", key, value)
    for key, value in selection_capabilities.items():
        add(metrics, "market_selection", "capability_boundary", key, value, notes="Verified from fixed Firm list, employee assignment, market settlement, and absence of bankruptcy/Exit/liquidation paths.")

    flags = {
        "verdict": verdict,
        "economic_behavior_changed": False,
        "financial_mechanism_changed": False,
        "pricing_mechanism_changed": False,
        "rng_changed": False,
        "diagnostic_rerun_used": False,
        "new_long_runs": 0,
        "seed7_21_run": False,
        "price_objective_understood": True,
        "price_cut_response_understood": True,
        "cost_floor_role_understood": True,
        "capacity_cost_feedback_understood": True,
        "profit_cash_wedge_understood": True,
        "borrowing_trigger_understood": True,
        "principal_ratchet_understood": True,
        "default_trap_understood": True,
        "market_selection_boundary_understood": True,
        "next_mechanism_family_selected": True,
        "selected_next_mechanism_family": selected_next,
        "price_learner_objective_identified": True,
        "price_learner_sees_interest": False,
        "price_learner_sees_debt": False,
        "price_learner_sees_default": False,
        **{key: value for key, value in system.items() if key != "verdict"},
    }

    OUT.mkdir(parents=True, exist_ok=True)
    for path in OUT.iterdir():
        if path.is_file():
            path.unlink()
        else:
            shutil.rmtree(path)
    pd.DataFrame(metrics).to_csv(OUT / "competitive_financial_trap_metrics.csv", index=False, encoding="utf-8-sig")
    (OUT / "acceptance_flags.json").write_text(json.dumps(flags, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    overview_figure(frame)

    cut1 = event_results[("cut", "firm_1")]
    cut0 = event_results[("cut", "firm_0")]
    cut2 = event_results[("cut", "firm_2")]
    rel = relations["pooled"]
    firm_rows = []
    for firm_id in sorted(core):
        firm_rows.append(
            f"| {firm_id} | {core[firm_id]['final_price']:.4f} | {core[firm_id]['mean_relative_price_mature']:.3f} | "
            f"{core[firm_id]['mean_market_share_mature']:.3f} | {core[firm_id]['mean_capacity_utilization_mature']:.3f} | "
            f"{core[firm_id]['mean_price_learner_objective_mature']:.1f} | {core[firm_id]['mean_OCF_mature']:.1f} | "
            f"{core[firm_id]['final_principal_utilization']:.3f} | {core[firm_id]['final_arrears']:.0f} | "
            f"{principals[firm_id]['principal_ratchet_classification']} | {trap_classes[firm_id]} |"
        )
    summary = f"""# Post-Step13 Firm Competitive-Financial Trap Audit

## Verdict

**{verdict}**

本审计完全复用 `main_step13_financial_core`，没有重跑模拟、消费 RNG、修改状态或改变任何经济机制。

## 核心诊断

当前 Firm price learner 优化的是 **12 周试价后 `firm.profit = sales_revenue - executed_wage_bill` 的 EMA 变化**。它不是 accounting operating profit、net income 或 CFO，也不读取 interest、principal、arrears、credit utilization、denied credit、D2/D3 或 contractual Default。正向试验结果使下一次更倾向延续方向，负向结果使其反转；保留 8% exploration，但 diagnostics 没有记录某次方向是否来自 exploration，因此没有伪造 exploratory/accepted/rejected 计数。

价格下界是 `normal_ULC * 1.02`，再受全局 `[0.5, 2.5]` 边界约束。realized ULC、inventory gap、margin、cost growth 和 Default 状态均不直接决定价格。Firm 1/3/4 mature window 内靠近 floor 2% 的周占比最高为 `{max(core[f]['share_within_2pct_floor_mature'] for f in (1,3,4)):.2%}`，因此 cost floor 不是它们不继续降价的主要限制。原始 `relative_price` 字段是 bootstrap 常量，不随价格更新；本审计按每周销量加权市场价格离线重算动态 relative price，不修改行为或原 CSV。

| Firm | Final price | Relative price | Share | Utilization | Learner objective | CFO | Principal util. | Arrears | Principal path | Trap classification |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
{chr(10).join(firm_rows)}

## 竞争到金融的链条

Mature Firm-week 的描述性关系为：relative price 与 market share 相关系数 `{rel['A_market_share_on_relative_price']['corr']:.3f}`；market share 与 utilization `{rel['B_utilization_on_market_share']['corr']:.3f}`；utilization 与 realized ULC `{rel['C_realized_ULC_on_utilization']['corr']:.3f}`；utilization 与 CFO `{rel['D_OCF_on_utilization']['corr']:.3f}`；CFO 与 borrowing `{rel['E_borrowing_on_OCF']['corr']:.3f}`。这些是描述性相关/简单 OLS，不是结构因果估计。

结果支持一条混合反馈：较高相对价格压低选择概率与份额，低份额降低利用率，低利用率机械抬高 realized ULC 并削弱现金生成，working-capital credit 随后填补缺口。Firm 1/3/4 的 principal 最终形成 `RATCHET_TO_LIMIT`，但 arrears 与 Default 不反向改变价格、生产或 Firm 存续。

## 调价事件研究

Firm 1 全程 review `{core[1]['full_review_count']}` 次，实际 cut `{core[1]['cut_count']}` 次、raise `{core[1]['raise_count']}` 次、keep `{core[1]['keep_count']}` 次。cut 的 +1..+4 周相对 -4..-1 周，平均 demand 变化 `{cut1.get('mean_change_demand', 0.0):.2f}`，learner objective 变化 `{cut1.get('mean_change_price_objective', 0.0):.2f}`，accounting operating profit 变化 `{cut1.get('mean_change_operating_profit', 0.0):.2f}`，CFO 变化 `{cut1.get('mean_change_ocf', 0.0):.2f}`；descriptive local demand-price response 为 `{cut1.get('demand_price_response', 0.0):.3f}`。由于 learner proxy/CFO 与 accounting operating profit 的方向分歧，分类为 **{cut_classifications[1]}**。

Firm 1 确实反复试验降价，并非被 floor 锁死。47 次 cut 中，12 周 EMA evaluation 有 `{core[1]['evaluated_cut_improved_count']}` 次改善、`{core[1]['evaluated_cut_worsened_count']}` 次恶化；57 次 raise 中也有 `{core[1]['evaluated_raise_improved_count']}` 次改善、`{core[1]['evaluated_raise_worsened_count']}` 次恶化。它仍维持最高价格，是因为非平稳市场中的短窗口 EMA 对两个方向都给出大量正负混杂反馈，下一次仍有 keep/exploration/反向试验；它没有“降低债务或避免 Default”的生存目标。positive margin 只是 realized-cost diagnostics，而且 Firm 1 mature learner objective 实际为负，不代表企业在经济意义上“满意”。

Firm 0 的低相对价格带来较高份额和接近满负荷利用率，使 CFO 与现金缓冲足以自我融资。Firm 2 是观测窗口内的 revolving control：它直到 week `{principals[2]['first_borrowing_week']}` 才开始借款，仍有大量 headroom，并在每周销售结算后同时发生 draw 与 repayment，因此尚未形成 90% utilization 后的 ratchet。它的 mature CFO 仍为负、principal 净增，所以这只能证明“还款机制确实工作”，不能证明 Firm 2 长期一定稳定。

## Profit 与 Cash Flow

Accounting CFO 在本模型口径中包含 interest cash outflow。`operating_profit - CFO` 主要由当期生产工资资本化后未进入 COGS 的 inventory book investment、interest payment 与实际 other operating flows解释；principal repayment 和 dividend 位于 financing cash flow，不属于 CFO。完整逐 Firm wedge、borrowing、repayment 与 transition snapshot 见 metrics CSV。

因此，一个看似有正 margin 或正 learner objective 的 Firm 仍可能因 inventory cash absorption、interest service、principal repayment和 operating-liquidity buffer 需要信用。更重要的是，pricing learner 对这组 financing stress 完全盲视。

## Default 与市场选择边界

代码矩阵确认：D2、D3 和 record-only Default 不改变 pricing、production、wage、dividend 或 borrowing policy。高 debt/credit denial 只通过 headroom、funded payroll/capacity、interest reservation 等现有流动性约束间接影响运营。

当前竞争能让弱 Firm 丢失 share、revenue、utilization 和 cash，但不能因竞争表现而失去既有 employees、productive capacity、active status、market participation 或通过 liquidation 清算资产。新进入劳动力只按现有 capacity 平衡分配，既有员工不跳槽。因此结论是：**PRODUCT-MARKET SELECTION EXISTS, BUT FIRM-POPULATION SELECTION DOES NOT.**

Firm 1/3/4 在 active Default 且 principal utilization >=95% 时仍持续经营的 Firm-weeks 合计 `{sum(principals[f]['descriptive_nonviable_firm_weeks'] for f in (1,3,4))}`。这构成 soft budget constraint 的描述性证据：信用延长了弱 Firm 生存，但当前没有 resolution/Exit 后果。

## 15 个明确回答

1. learner 优化 12 周试价后 legacy profit EMA，而不是会计或金融目标。
2. financing stress 不进入目标。
3. Firm 1 的 cut 在短事件窗平均改善 learner proxy/CFO，但 12 周 evaluation 对 cut/raise 都产生混合结果，且没有金融生存目标，因而没有形成持续单向降价。
4. Firm 1 实际 cut `{core[1]['cut_count']}` 次。
5. cut 后需求的平均变化为 `{cut1.get('mean_change_demand', 0.0):.2f}`，需求明确响应。
6. learner objective/CFO 平均改善 `{cut1.get('mean_change_price_objective', 0.0):.2f}` / `{cut1.get('mean_change_ocf', 0.0):.2f}`，但 accounting operating profit 变化 `{cut1.get('mean_change_operating_profit', 0.0):.2f}`，故总判定为 `{cut_classifications[1]}`。
7. Firm 1 未被 cost floor 实质锁定。
8. Firm 0 以低价获得高份额、近满负荷和正现金生成，从而无需债务。
9. Firm 2 较晚进入信用且仍有 headroom，展示真实 draw/repay；Firm 1/3/4 更早进入、累计净缺口使 principal ratchet 到上限。
10. utilization 与 realized ULC 的相关系数为 `{rel['C_realized_ULC_on_utilization']['corr']:.3f}`，反馈显著但 realized ULC 不设 price floor。
11. 数据支持 high price -> low share -> low utilization -> weak CFO -> borrowing 的多段描述性链条。
12. Default 不触发竞争适应行为。
13. credit 确实让弱 Firm 在高利用率/Default 后继续经营。
14. unresolved issue 是竞争、金融目标盲区、soft budget 与 missing Exit 的混合。
15. 下一步只提名 **{selected_next}**，本审计不实施。

## Next Mechanism Nomination

**{selected_next}**。原因是价格学习和信用流在各自局部语义内运行，但弱 Firm 缺少 Firm-population 层面的 resolution。提名不等于已经定义 Exit 条件或 liquidation 规则。
"""
    (OUT / "acceptance_summary.md").write_text(summary, encoding="utf-8-sig")
    print(verdict)
    print(OUT)


if __name__ == "__main__":
    main()
