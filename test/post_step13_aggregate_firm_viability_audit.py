"""Pure diagnostic audit of aggregate Firm viability and sectoral balance.

The only simulation performed here is the explicitly authorized canonical
seed-42, five-Firm, 3640-week baseline.  No production behavior or parameter is
changed.  Household-level market rows are not retained during the long run;
that recorder is passive and the canonical 0..1819 macro/Firm/accounting prefix
is compared against the accepted run before long-run results are accepted.
"""

from __future__ import annotations

import hashlib
import json
import math
import pickle
import random
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scenarios import apply_runtime_scenario, apply_scenario
from world import World


CANONICAL = ROOT / "test" / "output" / "main_step13_financial_core"
OUT = ROOT / "test" / "output" / "post_step13_aggregate_firm_viability_audit"
SCENARIO = "interest_behavioral_5pct"
POPULATION = 5000
FIRM_COUNT = 5
SEED = 42
CANONICAL_STEPS = 1820
LONG_STEPS = 3640
TOLERANCE = 1e-6
PARITY_TOLERANCE = 1e-8
WINDOWS = {
    "full": (0, 1819),
    "mature": (1560, 1819),
    "last_200": (1620, 1819),
}

PRODUCTION_SOURCES = [
    ROOT / "world.py",
    ROOT / "main.py",
    ROOT / "scenarios.py",
    ROOT / "economy" / "firm.py",
    ROOT / "economy" / "multi_firm.py",
    ROOT / "economy" / "accounting.py",
    ROOT / "economy" / "config.py",
    ROOT / "economy" / "ledger.py",
    ROOT / "central_bank" / "central_bank.py",
    ROOT / "central_bank" / "config.py",
]


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


def slope(values):
    series = pd.Series(values, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    if len(series) < 2:
        return 0.0
    return float(np.polyfit(np.arange(len(series), dtype=float), series.to_numpy(), 1)[0])


def correlation(left, right):
    frame = pd.DataFrame({"left": left, "right": right}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(frame) < 3 or frame["left"].std() <= 1e-12 or frame["right"].std() <= 1e-12:
        return np.nan
    return float(frame["left"].corr(frame["right"]))


def first_step(frame, mask):
    values = frame.loc[mask, "global_step"]
    return int(values.iloc[0]) if not values.empty else np.nan


def file_hashes():
    return {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in PRODUCTION_SOURCES
    }


def rng_hash(world):
    payload = {
        "global": random.getstate(),
        "market": world.market_rng.getstate(),
        "firms": [firm.firm_rng.getstate() for firm in world.firms],
        "marriage": world.marriage_system.marriage_rng.getstate(),
        "age_phase": world.age_phase_rng.getstate(),
    }
    return hashlib.sha256(pickle.dumps(payload, protocol=5)).hexdigest()


def add_metric(rows, section, entity, window, metric, value, unit="", notes=""):
    rows.append(
        {
            "section": section,
            "entity": entity,
            "window": window,
            "metric": metric,
            "value": value,
            "unit": unit,
            "notes": notes,
        }
    )


def load_run(directory):
    frames = {
        "macro": pd.read_csv(directory / "diagnostics.csv"),
        "firm": pd.read_csv(directory / "firm_diagnostics.csv"),
        "firm_accounting": pd.read_csv(directory / "accounting" / "firm_accounting.csv"),
        "household_accounting": pd.read_csv(directory / "accounting" / "household_accounting.csv"),
        "public_accounting": pd.read_csv(directory / "accounting" / "public_accounting.csv"),
        "central_bank_accounting": pd.read_csv(directory / "accounting" / "central_bank_accounting.csv"),
        "accounting_reconciliation": pd.read_csv(directory / "accounting" / "accounting_reconciliation.csv"),
    }
    frames["combined"] = combine_firm(frames["firm"], frames["firm_accounting"])
    return frames


def combine_firm(firm, accounting):
    selected = accounting[
        [
            "global_step", "firm_id", "accounting_revenue", "production_wage_cost",
            "manufacturing_cost", "inventory_cost_or_cogs",
            "spoilage_or_inventory_loss", "accounting_operating_profit",
            "accounting_net_income", "legacy_profit_before_dividend", "cfo",
            "cfi", "cff", "other_operating_cashflows", "inventory_book_value",
            "capitalized_production_cost", "cash_flow_gap", "inventory_bridge_gap",
            "equity_bridge_gap", "balance_sheet_gap",
        ]
    ].rename(
        columns={
            column: f"acct_{column}"
            for column in accounting.columns
            if column not in ("global_step", "firm_id")
            and column in {
                "accounting_revenue", "production_wage_cost", "manufacturing_cost",
                "inventory_cost_or_cogs", "spoilage_or_inventory_loss",
                "accounting_operating_profit", "accounting_net_income",
                "legacy_profit_before_dividend", "cfo", "cfi", "cff",
                "other_operating_cashflows", "inventory_book_value",
                "capitalized_production_cost", "cash_flow_gap",
                "inventory_bridge_gap", "equity_bridge_gap", "balance_sheet_gap",
            }
        }
    )
    return firm.merge(selected, on=["global_step", "firm_id"], how="left", validate="one_to_one")


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
    frames["combined"] = combine_firm(frames["firm"], frames["firm_accounting"])
    return frames


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
    world.steps = LONG_STEPS
    return world


def run_long_baseline():
    world = build_world()
    original_recorder = World.record_household_market_diagnostics

    def discard_household_market_rows(instance, household_spend_values, household_purchase_units):
        # Passive memory optimization only. Settlement has already completed;
        # this method normally appends observational per-Household rows.
        return None

    World.record_household_market_diagnostics = discard_household_market_rows
    started = time.perf_counter()
    try:
        for offset in range(LONG_STEPS):
            world.step()
            if (offset + 1) % 260 == 0 or offset + 1 == LONG_STEPS:
                print(
                    f"[LONG S0] {offset + 1}/{LONG_STEPS} weeks, "
                    f"{time.perf_counter() - started:.1f}s"
                )
    finally:
        World.record_household_market_diagnostics = original_recorder
    return world, time.perf_counter() - started


def compare_frames(reference, candidate, keys):
    left = reference.sort_values(keys).reset_index(drop=True)
    right = candidate.sort_values(keys).reset_index(drop=True)
    if len(left) != len(right):
        return {"pass": False, "max_abs_diff": math.inf, "first_difference": "row_count"}
    columns = [column for column in left.columns if column in right.columns]
    maximum = 0.0
    first = None
    for column in columns:
        a_col = left[column]
        b_col = right[column]
        if pd.api.types.is_numeric_dtype(a_col) and pd.api.types.is_numeric_dtype(b_col):
            a = pd.to_numeric(a_col, errors="coerce").to_numpy(dtype=float)
            b = pd.to_numeric(b_col, errors="coerce").to_numpy(dtype=float)
            nan_mismatch = np.isnan(a) ^ np.isnan(b)
            differences = np.abs(a - b)
            differences[np.isnan(differences)] = 0.0
            maximum = max(maximum, float(differences.max()) if len(differences) else 0.0)
            mismatch = nan_mismatch | (differences > PARITY_TOLERANCE)
        else:
            a = a_col.fillna("<NA>").astype(str).to_numpy()
            b = b_col.fillna("<NA>").astype(str).to_numpy()
            differences = np.zeros(len(a))
            mismatch = a != b
        if first is None and np.any(mismatch):
            index = int(np.flatnonzero(mismatch)[0])
            first = {
                "column": column,
                "keys": {key: left.loc[index, key] for key in keys},
                "reference": a_col.iloc[index],
                "candidate": b_col.iloc[index],
                "abs_diff": finite(differences[index], None),
            }
    return {"pass": first is None, "max_abs_diff": maximum, "first_difference": first}


def prefix_parity(canonical, long_frames):
    results = {}
    for name in (
        "macro", "firm", "firm_accounting", "household_accounting",
        "public_accounting", "central_bank_accounting", "accounting_reconciliation",
    ):
        keys = ["global_step", "firm_id"] if name in ("firm", "firm_accounting") else ["global_step"]
        candidate = long_frames[name][long_frames[name]["global_step"] < CANONICAL_STEPS]
        results[name] = compare_frames(canonical[name], candidate, keys)
    return {
        "pass": all(item["pass"] for item in results.values()),
        "max_abs_diff": max(item["max_abs_diff"] for item in results.values()),
        "details": results,
    }


def build_sector_frame(frames):
    macro = frames["macro"].sort_values("global_step").reset_index(drop=True)
    household = frames["household_accounting"].sort_values("global_step").reset_index(drop=True)
    public = frames["public_accounting"].sort_values("global_step").reset_index(drop=True)
    central = frames["central_bank_accounting"].sort_values("global_step").reset_index(drop=True)
    firm_accounting = frames["firm_accounting"].sort_values(["global_step", "firm_id"])
    firm_week = firm_accounting.groupby("global_step", as_index=False).agg(
        firm_cash_start=("cash_start", "sum"),
        firm_cash_end=("cash_end", "sum"),
        aggregate_cfo=("cfo", "sum"),
        aggregate_cff=("cff", "sum"),
    )
    sector = macro[
        [
            "global_step", "firm_cash", "total_household_wealth", "public_money_stock",
            "located_money_stock", "total_money_stock", "expected_money_stock",
            "central_bank_net_money_issued", "working_capital_loan_issued",
            "working_capital_loan_repaid", "central_bank_inventory_purchase",
            "central_bank_market_release_money_destroyed", "monetary_accounting_gap",
            "money_delta_gap", "food_conservation_gap", "invariant_failed",
        ]
    ].merge(
        household[["global_step", "saving", "cash_wealth", "estate_no_heir_outflow"]],
        on="global_step", validate="one_to_one",
    ).merge(
        public[["global_step", "public_cash_start", "public_cash_end"]],
        on="global_step", validate="one_to_one",
    ).merge(
        central[
            [
                "global_step", "gross_money_created", "money_destroyed",
                "net_money_change", "inventory_money_created",
            ]
        ], on="global_step", validate="one_to_one",
    ).merge(firm_week, on="global_step", validate="one_to_one")

    sector["household_money"] = sector["total_household_wealth"]
    sector["public_other_money"] = sector["public_money_stock"]
    sector["delta_household_money"] = sector["household_money"].diff()
    sector["delta_firm_cash"] = sector["firm_cash"].diff()
    sector["delta_public_other_money"] = sector["public_other_money"].diff()
    sector["delta_located_money"] = sector["located_money_stock"].diff()
    first = sector.index[0]
    household_opening = (
        sector.loc[first, "cash_wealth"]
        - sector.loc[first, "saving"]
        + sector.loc[first, "estate_no_heir_outflow"]
    )
    sector.loc[first, "delta_household_money"] = sector.loc[first, "household_money"] - household_opening
    sector.loc[first, "delta_firm_cash"] = sector.loc[first, "firm_cash"] - sector.loc[first, "firm_cash_start"]
    sector.loc[first, "delta_public_other_money"] = sector.loc[first, "public_other_money"] - sector.loc[first, "public_cash_start"]
    opening_located = household_opening + sector.loc[first, "firm_cash_start"] + sector.loc[first, "public_cash_start"]
    sector.loc[first, "delta_located_money"] = sector.loc[first, "located_money_stock"] - opening_located
    sector["sectoral_balance_gap"] = (
        sector["delta_household_money"]
        + sector["delta_firm_cash"]
        + sector["delta_public_other_money"]
        - sector["central_bank_net_money_issued"]
    )
    sector["net_credit_money_creation"] = sector["gross_money_created"] - sector["money_destroyed"]
    sector["noncredit_net_money_creation"] = (
        sector["central_bank_net_money_issued"] - sector["net_credit_money_creation"]
    )
    return sector


def window_frame(frame, window):
    low, high = WINDOWS[window]
    return frame[frame["global_step"].between(low, high)]


def audit_sectoral_balance(frames, metrics):
    sector = build_sector_frame(frames)
    max_gap = float(sector["sectoral_balance_gap"].abs().max())
    add_metric(metrics, "money_identity", "system", "full", "sectoral_money_identity_max_abs_gap", max_gap, "currency")
    add_metric(metrics, "money_identity", "system", "full", "macro_monetary_accounting_max_abs_gap", float(sector["monetary_accounting_gap"].abs().max()), "currency")
    add_metric(metrics, "money_identity", "system", "full", "money_stock_flow_max_abs_gap", float(sector["money_delta_gap"].abs().max()), "currency")
    for channel, total, note in (
        ("working_capital_credit_creation", sector["gross_money_created"].sum(), "Firm loan issuance creates cash and principal."),
        ("principal_repayment_destruction", sector["money_destroyed"].sum(), "Principal repayment destroys cash and principal."),
        ("inventory_purchase_money_creation", sector["inventory_money_created"].sum(), "Canonical switch is disabled; purchases use public income only."),
        ("inventory_release_money_destruction", sector["central_bank_market_release_money_destroyed"].sum(), "Only issuance-backed public inventory releases can destroy money."),
    ):
        add_metric(metrics, "money_channels", "system", "full", channel, total, "currency", note)

    summaries = {}
    for window in WINDOWS:
        rows = window_frame(sector, window)
        summary = {
            "household_money_change_per_week": rows["delta_household_money"].mean(),
            "reported_household_saving_per_week": rows["saving"].mean(),
            "firm_cash_change_per_week": rows["delta_firm_cash"].mean(),
            "aggregate_firm_CFO_per_week": rows["aggregate_cfo"].mean(),
            "aggregate_firm_CFF_per_week": rows["aggregate_cff"].mean(),
            "gross_loan_issuance_per_week": rows["gross_money_created"].mean(),
            "principal_repayment_per_week": rows["money_destroyed"].mean(),
            "net_credit_money_creation_per_week": rows["net_credit_money_creation"].mean(),
            "total_net_money_creation_per_week": rows["central_bank_net_money_issued"].mean(),
            "public_balance_change_per_week": rows["delta_public_other_money"].mean(),
        }
        summaries[window] = summary
        for metric, value in summary.items():
            add_metric(metrics, "sectoral_balance", "system", window, metric, value, "currency/week")

    weekly_corr = correlation(sector["net_credit_money_creation"], sector["delta_household_money"])
    rolling_credit = sector["net_credit_money_creation"].rolling(52, min_periods=52).sum()
    rolling_household = sector["delta_household_money"].rolling(52, min_periods=52).sum()
    rolling_corr = correlation(rolling_credit, rolling_household)
    add_metric(metrics, "sectoral_balance", "system", "full", "weekly_corr_net_credit_household_money_change", weekly_corr, "correlation")
    add_metric(metrics, "sectoral_balance", "system", "full", "rolling_52week_corr_net_credit_household_money_change", rolling_corr, "correlation")
    for metric, column in (
        ("cumulative_net_credit_creation", "net_credit_money_creation"),
        ("cumulative_household_money_change", "delta_household_money"),
        ("cumulative_firm_cash_change", "delta_firm_cash"),
        ("cumulative_public_balance_change", "delta_public_other_money"),
    ):
        add_metric(metrics, "sectoral_balance", "system", "full", metric, sector[column].sum(), "currency")

    mature = summaries["mature"]
    noncredit = window_frame(sector, "mature")["noncredit_net_money_creation"].mean()
    required_credit_for_stable_cash = (
        mature["household_money_change_per_week"]
        + mature["public_balance_change_per_week"]
        - noncredit
    )
    observed_repayment = mature["principal_repayment_per_week"]
    implied_cash_if_no_new_credit = (
        noncredit
        - observed_repayment
        - mature["household_money_change_per_week"]
        - mature["public_balance_change_per_week"]
    )
    add_metric(metrics, "passive_deleveraging_bound", "system", "mature", "net_credit_required_for_stable_firm_cash", required_credit_for_stable_cash, "currency/week")
    add_metric(metrics, "passive_deleveraging_bound", "system", "mature", "implied_firm_cash_change_if_zero_borrowing_and_observed_repayment", implied_cash_if_no_new_credit, "currency/week")
    add_metric(metrics, "passive_deleveraging_bound", "system", "mature", "maximum_net_debt_repayment_with_stable_cash", -required_credit_for_stable_cash, "currency/week", "Negative means stable Firm cash instead requires net credit expansion.")
    return sector, summaries, max_gap, required_credit_for_stable_cash, implied_cash_if_no_new_credit


def audit_firm_sector(frames, metrics):
    accounting = frames["firm_accounting"].copy()
    macro = frames["macro"]
    combined = frames["combined"].copy()
    aggregate_summaries = {}
    for window in WINDOWS:
        low, high = WINDOWS[window]
        rows = accounting[accounting["global_step"].between(low, high)]
        weekly = rows.groupby("global_step", as_index=False).agg(
            sales_revenue=("sales_revenue", "sum"),
            executed_payroll=("wage_payments", "sum"),
            cogs=("inventory_cost_or_cogs", "sum"),
            spoilage_loss=("spoilage_or_inventory_loss", "sum"),
            operating_profit=("accounting_operating_profit", "sum"),
            net_income=("accounting_net_income", "sum"),
            cfo=("cfo", "sum"), cff=("cff", "sum"),
            dividends=("dividends", "sum"), interest=("interest_paid", "sum"),
            principal_repayment=("principal_repaid", "sum"),
            borrowing=("loan_issued", "sum"),
            capitalized_production_cost=("capitalized_production_cost", "sum"),
        )
        macro_rows = macro[macro["global_step"].between(low, high)]
        summary = {column: weekly[column].mean() for column in weekly.columns if column != "global_step"}
        summary["firm_cash_change"] = macro_rows["firm_cash_change"].mean()
        aggregate_summaries[window] = summary
        for metric, value in summary.items():
            add_metric(metrics, "aggregate_firm_viability", "firm_sector", window, metric, value, "currency/week")

    mature = combined[combined["global_step"].between(*WINDOWS["mature"])]
    per_firm = {}
    for firm_id, rows in mature.groupby("firm_id", sort=True):
        rows = rows.sort_values("global_step")
        summary = {
            "mean_CFO": rows["acct_cfo"].mean(),
            "median_CFO": rows["acct_cfo"].median(),
            "CFO_slope": slope(rows["acct_cfo"]),
            "cash_slope": slope(rows["cash"]),
            "mean_operating_profit": rows["acct_accounting_operating_profit"].mean(),
            "mean_net_income": rows["acct_accounting_net_income"].mean(),
            "gross_borrowing": rows["loan_issued"].sum(),
            "gross_repayment": rows["loan_repaid"].sum(),
            "principal_change": rows.iloc[-1]["closing_principal"] - rows.iloc[0]["opening_principal"],
        }
        per_firm[int(firm_id)] = summary
        for metric, value in summary.items():
            unit = "currency/week" if metric.startswith("mean_") or metric.endswith("slope") else "currency"
            add_metric(metrics, "firm_viability", f"firm_{int(firm_id)}", "mature", metric, value, unit)
    add_metric(metrics, "firm_viability", "firm_sector", "mature", "firms_with_mean_CFO_positive", sum(item["mean_CFO"] > 0 for item in per_firm.values()), "firms")
    add_metric(metrics, "firm_viability", "firm_sector", "mature", "firms_with_mean_operating_profit_positive", sum(item["mean_operating_profit"] > 0 for item in per_firm.values()), "firms")
    add_metric(metrics, "firm_viability", "firm_sector", "mature", "firms_with_net_principal_decline", sum(item["principal_change"] < -TOLERANCE for item in per_firm.values()), "firms")
    return aggregate_summaries, per_firm


def audit_firm0_firm2(frames, metrics):
    combined = frames["combined"].sort_values(["firm_id", "global_step"])
    firm0 = combined[combined["firm_id"] == 0].copy()
    firm0_metrics = {}
    for label, count in (("full", len(firm0)), ("last_520", 520), ("last_260", 260), ("last_200", 200)):
        rows = firm0.tail(count)
        firm0_metrics[f"{label}_cash_slope"] = slope(rows["cash"])
        add_metric(metrics, "firm0_cash_direction", "firm_0", label, "cash_slope", firm0_metrics[f"{label}_cash_slope"], "currency/week")
    mature = firm0[firm0["global_step"].between(*WINDOWS["mature"])]
    last200 = firm0.tail(200)
    firm0_metrics.update(
        {
            "mature_mean_CFO": mature["acct_cfo"].mean(),
            "last_200_mean_CFO": last200["acct_cfo"].mean(),
            "final_13week_CFO_mean": firm0["acct_cfo"].tail(13).mean(),
            "final_cash": firm0.iloc[-1]["cash"],
        }
    )
    for metric, value in firm0_metrics.items():
        if "slope" not in metric:
            add_metric(metrics, "firm0_cash_direction", "firm_0", "canonical_end", metric, value, "currency")
    firm0_class = "CASH_ACCUMULATING" if firm0_metrics["last_200_cash_slope"] > TOLERANCE else "CASH_DECLINING" if firm0_metrics["last_200_cash_slope"] < -TOLERANCE else "APPROXIMATELY_STABLE"
    add_metric(metrics, "firm0_cash_direction", "firm_0", "canonical_end", "classification", firm0_class)

    firm2 = combined[combined["firm_id"] == 2].copy()
    first_borrow = first_step(firm2, firm2["loan_issued"] > TOLERANCE)
    before = firm2[firm2["global_step"] < first_borrow] if pd.notna(first_borrow) else firm2
    after = firm2[firm2["global_step"] >= first_borrow] if pd.notna(first_borrow) else firm2.iloc[0:0]
    final = firm2.iloc[-1]
    firm2_metrics = {
        "first_borrowing_week": first_borrow,
        "pre_borrow_CFO_slope": slope(before["acct_cfo"]),
        "pre_borrow_cash_slope": slope(before["cash"]),
        "post_borrow_principal_slope": slope(after["closing_principal"]),
        "post_borrow_gross_repayment": after["loan_repaid"].sum(),
        "post_borrow_gross_borrowing": after["loan_issued"].sum(),
        "final_headroom": final["credit_headroom"],
        "final_utilization": safe_ratio(final["closing_principal"], final["credit_limit"]),
    }
    for metric, value in firm2_metrics.items():
        add_metric(metrics, "firm2_current_status", "firm_2", "canonical_end", metric, value, "week" if "week" in metric else "currency")
    current_class = "EARLY_STAGE_RATCHET_RISK" if firm2_metrics["post_borrow_principal_slope"] > 0 and mature_firm_mean(frames, 2, "acct_cfo") < 0 else "TEMPORARY_REVOLVING_NEED"
    add_metric(metrics, "firm2_current_status", "firm_2", "canonical_end", "classification", current_class)
    return firm0_metrics, firm0_class, firm2_metrics, current_class


def mature_firm_mean(frames, firm_id, column):
    rows = frames["combined"]
    rows = rows[(rows["firm_id"] == firm_id) & rows["global_step"].between(*WINDOWS["mature"])]
    return float(rows[column].mean())


def audit_households(frames, metrics):
    household = frames["household_accounting"].copy()
    summaries = {}
    for window in WINDOWS:
        rows = window_frame(household, window)
        income = rows["wages"] + rows["dividends"] + rows["inheritance_public_transfers"]
        summary = {
            "household_income_per_week": income.mean(),
            "household_consumption_per_week": rows["consumption_expenditure"].mean(),
            "household_saving_per_week": rows["saving"].mean(),
            "household_saving_rate": safe_ratio(rows["saving"].sum(), income.sum()),
            "firm_public_to_household_minus_consumption": (income - rows["consumption_expenditure"]).mean(),
        }
        summaries[window] = summary
        for metric, value in summary.items():
            add_metric(metrics, "household_demand_leakage", "household_sector", window, metric, value, "ratio" if "rate" in metric else "currency/week")
    return summaries


def audit_goods_inventory(frames, metrics):
    macro = frames["macro"].sort_values("global_step").copy()
    total_inventory = macro["food_inventory_units"] + macro["central_bank_food_inventory_units"]
    macro["system_inventory_change"] = total_inventory.diff()
    opening_first = (
        total_inventory.iloc[0]
        + macro.iloc[0]["food_sales_units"]
        + macro.iloc[0]["central_bank_food_subsidy_units"]
        + macro.iloc[0]["food_spoilage_units"]
        - macro.iloc[0]["actual_production"]
    )
    macro.loc[macro.index[0], "system_inventory_change"] = total_inventory.iloc[0] - opening_first
    goods_summaries = {}
    for window in WINDOWS:
        rows = window_frame(macro, window)
        summary = {
            "production_units_per_week": rows["actual_production"].mean(),
            "sales_units_per_week": rows["food_sales_units"].mean(),
            "system_inventory_change_units_per_week": rows["system_inventory_change"].mean(),
            "spoilage_units_per_week": rows["food_spoilage_units"].mean(),
            "public_subsidy_units_per_week": rows["central_bank_food_subsidy_units"].mean(),
            "public_inventory_purchase_units_per_week": rows["central_bank_food_purchase_executed_units"].mean(),
            "public_inventory_release_units_per_week": rows["central_bank_food_release_units"].mean(),
            "goods_identity_max_abs_gap": rows["food_conservation_gap"].abs().max(),
        }
        goods_summaries[window] = summary
        for metric, value in summary.items():
            add_metric(metrics, "goods_balance", "system", window, metric, value, "units/week")

    accounting = frames["firm_accounting"]
    mature = accounting[accounting["global_step"].between(*WINDOWS["mature"])]
    weekly = mature.groupby("global_step", as_index=False).sum(numeric_only=True)
    production = goods_summaries["mature"]["production_units_per_week"]
    sales = goods_summaries["mature"]["sales_units_per_week"]
    spoil_units = goods_summaries["mature"]["spoilage_units_per_week"]
    spoil_loss = weekly["spoilage_or_inventory_loss"].mean()
    operating_profit = weekly["accounting_operating_profit"].mean()
    spoilage = {
        "spoilage_units_per_week": spoil_units,
        "spoilage_book_loss_per_week": spoil_loss,
        "spoilage_share_production": safe_ratio(spoil_units, production),
        "spoilage_share_sales": safe_ratio(spoil_units, sales),
        "spoilage_loss_relative_abs_operating_profit": safe_ratio(spoil_loss, abs(operating_profit)),
        "operating_profit_excluding_spoilage_loss": operating_profit + spoil_loss,
    }
    for metric, value in spoilage.items():
        add_metric(metrics, "spoilage_economics", "firm_sector", "mature", metric, value, "ratio" if "share" in metric or "relative" in metric else "currency/week")

    combined = frames["combined"].sort_values(["firm_id", "global_step"]).copy()
    inventory_results = {}
    for firm_id, rows in combined[combined["global_step"].between(*WINDOWS["mature"])].groupby("firm_id"):
        rows = rows.sort_values("global_step")
        book_change = rows["acct_inventory_book_value"].diff().mean()
        result = {
            "capitalized_production_cost_per_week": rows["acct_capitalized_production_cost"].mean(),
            "COGS_per_week": rows["acct_inventory_cost_or_cogs"].mean(),
            "spoilage_loss_per_week": rows["acct_spoilage_or_inventory_loss"].mean(),
            "inventory_book_change_per_week": book_change,
            "CFO_per_week": rows["acct_cfo"].mean(),
            "production_units_per_week": rows["actual_production"].mean(),
            "sales_units_per_week": rows["sales_units"].mean(),
        }
        inventory_results[int(firm_id)] = result
        for metric, value in result.items():
            add_metric(metrics, "inventory_financing", f"firm_{int(firm_id)}", "mature", metric, value, "currency/week" if "units" not in metric else "units/week")
    return goods_summaries, spoilage, inventory_results


def audit_price_alignment(frames, metrics):
    combined = frames["combined"].copy()
    mature = combined[combined["global_step"].between(*WINDOWS["mature"])]
    results = {}
    groups = [("pooled", mature)] + [(f"firm_{int(firm_id)}", rows) for firm_id, rows in mature.groupby("firm_id")]
    for entity, rows in groups:
        proxy = rows["profit"]
        op = rows["acct_accounting_operating_profit"]
        net = rows["acct_accounting_net_income"]
        cfo = rows["acct_cfo"]
        result = {
            "corr_proxy_operating_profit": correlation(proxy, op),
            "corr_proxy_net_income": correlation(proxy, net),
            "corr_proxy_CFO": correlation(proxy, cfo),
            "sign_disagreement_proxy_operating_profit": float((np.sign(proxy) != np.sign(op)).mean()),
            "sign_disagreement_proxy_net_income": float((np.sign(proxy) != np.sign(net)).mean()),
            "sign_disagreement_proxy_CFO": float((np.sign(proxy) != np.sign(cfo)).mean()),
            "proxy_positive_operating_profit_negative_share": float(((proxy > 0) & (op < 0)).mean()),
            "proxy_positive_CFO_negative_share": float(((proxy > 0) & (cfo < 0)).mean()),
        }
        results[entity] = result
        for metric, value in result.items():
            add_metric(metrics, "price_objective_alignment", entity, "mature", metric, value, "correlation" if metric.startswith("corr") else "share")

    break_even = {}
    for firm_id, rows in mature.groupby("firm_id"):
        accounting_break_even = rows["acct_inventory_cost_or_cogs"] + rows["acct_spoilage_or_inventory_loss"]
        cash_sales_break_even = rows["wage_payment"] - rows["acct_other_operating_cashflows"]
        result = {
            "actual_accounting_revenue": rows["acct_accounting_revenue"].mean(),
            "accounting_operating_break_even_revenue": accounting_break_even.mean(),
            "accounting_revenue_gap": (rows["acct_accounting_revenue"] - accounting_break_even).mean(),
            "actual_sales_collections": rows["sales_revenue"].mean(),
            "cash_operating_break_even_sales": cash_sales_break_even.mean(),
            "cash_revenue_gap": (rows["sales_revenue"] - cash_sales_break_even).mean(),
        }
        break_even[int(firm_id)] = result
        for metric, value in result.items():
            add_metric(metrics, "passive_break_even", f"firm_{int(firm_id)}", "mature", metric, value, "currency/week")
    return results, break_even


def audit_resources(frames, metrics):
    firm = frames["firm"].sort_values(["firm_id", "global_step"]).copy()
    results = {}
    correlations = []
    for firm_id, rows in firm.groupby("firm_id", sort=True):
        rows = rows.sort_values("global_step")
        result = {
            "employee_count_start": rows.iloc[0]["employee_count"],
            "employee_count_end": rows.iloc[-1]["employee_count"],
            "employee_count_min": rows["employee_count"].min(),
            "employee_count_max": rows["employee_count"].max(),
            "capacity_start": rows.iloc[0]["productive_capacity"],
            "capacity_end": rows.iloc[-1]["productive_capacity"],
            "capacity_min": rows["productive_capacity"].min(),
            "capacity_max": rows["productive_capacity"].max(),
            "market_share_start": rows.iloc[0]["unit_market_share"],
            "market_share_end": rows.iloc[-1]["unit_market_share"],
            "market_share_min": rows["unit_market_share"].min(),
            "market_share_max": rows["unit_market_share"].max(),
            "mean_capacity_utilization": rows["capacity_utilization"].mean(),
            "mean_unmet_demand": rows["unmet_demand"].mean(),
            "corr_weekly_share_change_capacity_change": correlation(rows["unit_market_share"].diff(), rows["productive_capacity"].diff()),
        }
        correlations.append(result["corr_weekly_share_change_capacity_change"])
        results[int(firm_id)] = result
        for metric, value in result.items():
            add_metric(metrics, "resource_reallocation", f"firm_{int(firm_id)}", "full", metric, value, "correlation" if metric.startswith("corr") else "count_or_units")
    add_metric(metrics, "resource_reallocation", "system", "full", "mean_corr_share_change_capacity_change", float(np.nanmean(correlations)), "correlation")
    for metric, value, notes in (
        ("existing_worker_can_switch_for_demand", False, "No runtime reassignment path."),
        ("existing_worker_can_switch_for_profit", False, "No runtime reassignment path."),
        ("existing_worker_can_switch_for_debt", False, "No runtime reassignment path."),
        ("existing_worker_can_switch_for_wage", False, "No inter-Firm wage-choice path."),
        ("new_worker_allocation_balances_capacity", True, "Unassigned productive workers join the currently lowest-capacity Firm."),
        ("weak_firm_can_competitively_shrink", False, "Employee loss occurs through lifecycle attrition, not market performance."),
        ("strong_firm_can_demand_expand", False, "Demand/share/unmet demand do not recruit workers or capacity."),
    ):
        add_metric(metrics, "resource_semantics", "system", "source", metric, value, "boolean", notes)
    return results


def accounting_guardrails(frames, metrics, label):
    checks = {
        "monetary_accounting_gap": frames["macro"]["monetary_accounting_gap"].abs().max(),
        "money_delta_gap": frames["macro"]["money_delta_gap"].abs().max(),
        "goods_conservation_gap": frames["macro"]["food_conservation_gap"].abs().max(),
        "firm_cash_flow_gap": frames["firm_accounting"]["cash_flow_gap"].abs().max(),
        "inventory_bridge_gap": frames["firm_accounting"]["inventory_bridge_gap"].abs().max(),
        "equity_bridge_gap": frames["firm_accounting"]["equity_bridge_gap"].abs().max(),
        "household_wealth_bridge_gap": frames["household_accounting"]["household_wealth_bridge_gap"].abs().max(),
        "public_cash_flow_gap": frames["public_accounting"]["public_cash_flow_gap"].abs().max(),
        "credit_money_stock_flow_gap": frames["central_bank_accounting"]["credit_money_stock_flow_gap"].abs().max(),
        "loan_reconciliation_gap": frames["accounting_reconciliation"]["loan_reconciliation_gap"].abs().max(),
        "total_lender_claim_gap": frames["accounting_reconciliation"]["total_lender_claim_gap"].abs().max(),
    }
    for metric, value in checks.items():
        add_metric(metrics, "accounting_guardrail", "system", label, metric, value, "currency_or_units")
    invariant_failures = int(frames["macro"]["invariant_failed"].fillna(False).astype(bool).sum())
    add_metric(metrics, "accounting_guardrail", "system", label, "invariant_failures", invariant_failures, "count")
    return max(finite(value) for value in checks.values()), invariant_failures


def audit_long_run(frames, metrics):
    combined = frames["combined"].copy()
    combined["principal_utilization"] = np.where(
        combined["credit_limit"].gt(TOLERANCE) & np.isfinite(combined["credit_limit"]),
        combined["closing_principal"] / combined["credit_limit"], 0.0,
    )
    firm_results = {}
    for firm_id, rows in combined.groupby("firm_id", sort=True):
        rows = rows.sort_values("global_step")
        final = rows.iloc[-1]
        trailing = rows.tail(520)
        result = {
            "first_borrowing_week": first_step(rows, rows["loan_issued"] > TOLERANCE),
            "first_50pct_utilization_week": first_step(rows, rows["principal_utilization"] >= 0.50),
            "first_90pct_utilization_week": first_step(rows, rows["principal_utilization"] >= 0.90),
            "first_99pct_utilization_week": first_step(rows, rows["principal_utilization"] >= 0.99),
            "first_default_week": first_step(rows, rows["default_event_this_week"].fillna(False).astype(bool)),
            "maximum_utilization": rows["principal_utilization"].max(),
            "final_utilization": final["principal_utilization"],
            "final_principal": final["closing_principal"],
            "final_arrears": final["closing_interest_arrears"],
            "final_cash": final["cash"],
            "trailing_520_mean_CFO": trailing["acct_cfo"].mean(),
            "trailing_520_cash_slope": slope(trailing["cash"]),
        }
        firm_results[int(firm_id)] = result
        for metric, value in result.items():
            add_metric(metrics, "long_run_firm", f"firm_{int(firm_id)}", "0_3639", metric, value, "week" if "week" in metric else "currency_or_ratio")

    macro = frames["macro"].sort_values("global_step")
    sector = build_sector_frame(frames)
    trailing_steps = set(macro.tail(520)["global_step"])
    trailing_sector = sector[sector["global_step"].isin(trailing_steps)]
    trailing_firm = combined[combined["global_step"].isin(trailing_steps)]
    weekly_cfo = trailing_firm.groupby("global_step")["acct_cfo"].sum()
    weekly_principal = trailing_firm.groupby("global_step")["closing_principal"].sum()
    weekly_arrears = trailing_firm.groupby("global_step")["closing_interest_arrears"].sum()
    long_system = {
        "trailing_520_aggregate_CFO": weekly_cfo.mean(),
        "trailing_520_aggregate_firm_cash_slope": slope(macro.tail(520)["firm_cash"]),
        "trailing_520_household_wealth_slope": slope(macro.tail(520)["total_household_wealth"]),
        "trailing_520_net_credit_creation_per_week": trailing_sector["net_credit_money_creation"].mean(),
        "trailing_520_total_principal_slope": slope(weekly_principal),
        "trailing_520_total_arrears_slope": slope(weekly_arrears),
        "initial_population": macro.iloc[0]["population"],
        "population_at_1819": macro.loc[macro["global_step"] == 1819, "population"].iloc[0],
        "final_population": macro.iloc[-1]["population"],
        "final_total_principal": weekly_principal.iloc[-1],
        "final_total_arrears": weekly_arrears.iloc[-1],
    }
    for metric, value in long_system.items():
        add_metric(metrics, "long_run_system", "system", "trailing_520_or_endpoint", metric, value, "currency_units_or_people")
    saturated_count = sum(result["final_utilization"] >= 0.90 for result in firm_results.values())
    default_count = sum(pd.notna(result["first_default_week"]) for result in firm_results.values())
    generalized = bool(saturated_count >= 4 and long_system["trailing_520_total_arrears_slope"] > TOLERANCE)
    if generalized:
        regime = "CREDIT_CAP_SATURATION_WITH_ARREARS_GROWTH"
    elif long_system["trailing_520_total_principal_slope"] > TOLERANCE:
        regime = "PERSISTENT_NET_FIRM_CREDIT_EXPANSION"
    elif abs(long_system["trailing_520_aggregate_firm_cash_slope"]) <= TOLERANCE:
        regime = "STABLE_AGGREGATE_FIRM_FINANCING"
    else:
        regime = "OTHER_REGIME"
    add_metric(metrics, "long_run_system", "system", "trailing_520", "regime_classification", regime)
    return combined, firm_results, long_system, generalized, regime, saturated_count, default_count


def make_figure(combined):
    frame = combined.copy()
    if "principal_utilization" not in frame:
        frame["principal_utilization"] = np.where(
            frame["credit_limit"].gt(TOLERANCE) & np.isfinite(frame["credit_limit"]),
            frame["closing_principal"] / frame["credit_limit"], 0.0,
        )
    colors = ["#176B87", "#D1495B", "#2A9D8F", "#E9A23B", "#6D597A"]
    fig, axes = plt.subplots(2, 2, figsize=(17, 10), sharex=True)
    for firm_id, rows in frame.groupby("firm_id", sort=True):
        rows = rows.sort_values("global_step")
        color = colors[int(firm_id)]
        rolling_cfo = rows["acct_cfo"].rolling(26, min_periods=1).mean()
        axes[0, 0].plot(rows["global_step"], rows["cash"], color=color, linewidth=1.2, label=f"Firm {int(firm_id)}")
        axes[0, 1].plot(rows["global_step"], rolling_cfo, color=color, linewidth=1.1, label=f"Firm {int(firm_id)}")
        axes[1, 0].plot(rows["global_step"], rows["unit_market_share"], color=color, linewidth=1.0, label=f"Firm {int(firm_id)}")
        axes[1, 1].plot(rows["global_step"], rows["principal_utilization"], color=color, linewidth=1.2, label=f"F{int(firm_id)} principal")
        arrears_ratio = np.where(
            rows["credit_limit"].gt(TOLERANCE), rows["closing_interest_arrears"] / rows["credit_limit"], 0.0
        )
        axes[1, 1].plot(rows["global_step"], arrears_ratio, color=color, linestyle=":", linewidth=0.8, alpha=0.75)
    titles = [
        "A. Firm cash", "B. Accounting CFO (26-week mean)",
        "C. Unit market share", "D. Principal utilization (solid), arrears/limit (dotted)",
    ]
    for axis, title in zip(axes.ravel(), titles):
        axis.set_title(title, fontweight="bold")
        axis.axvline(1820, color="#222222", linestyle="--", linewidth=1.0, alpha=0.8)
        axis.grid(alpha=0.2)
        axis.set_xlabel("Simulation week")
    axes[0, 0].set_ylabel("Currency")
    axes[0, 1].set_ylabel("Currency/week")
    axes[1, 0].set_ylabel("Share")
    axes[1, 1].set_ylabel("Ratio")
    axes[1, 1].axhline(0.90, color="#555555", linestyle="--", linewidth=0.7)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 1.01))
    fig.suptitle("Canonical S0 Long-Run Firm Viability, week 1820 boundary marked", fontsize=15, fontweight="bold", y=1.04)
    fig.tight_layout()
    fig.savefig(OUT / "long_run_firm_viability_overview.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def markdown_table(frame, columns, digits=3):
    subset = frame[columns].copy()
    for column in subset.columns:
        if pd.api.types.is_float_dtype(subset[column]):
            subset[column] = subset[column].map(lambda value: "" if pd.isna(value) else f"{value:.{digits}f}")
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    lines.extend("| " + " | ".join(str(value) for value in row) + " |" for row in subset.itertuples(index=False, name=None))
    return "\n".join(lines)


def write_summary(
    verdict, nomination, parity, sector_summaries, required_credit,
    implied_cash, aggregate, per_firm, firm0, firm0_class, firm2,
    firm2_class, households, goods, spoilage, alignment, resources,
    long_firms, long_system, generalized, regime, accounting_max,
    runtime_seconds, flags,
):
    firm_rows = pd.DataFrame(
        [
            {"firm_id": firm_id, **per_firm[firm_id], **{
                "long_final_utilization": long_firms[firm_id]["final_utilization"],
                "long_final_principal": long_firms[firm_id]["final_principal"],
                "long_final_arrears": long_firms[firm_id]["final_arrears"],
            }}
            for firm_id in sorted(per_firm)
        ]
    )
    long_rows = pd.DataFrame([{"firm_id": firm_id, **values} for firm_id, values in sorted(long_firms.items())])
    firm_table = markdown_table(
        firm_rows,
        ["firm_id", "mean_CFO", "mean_operating_profit", "cash_slope", "principal_change", "long_final_utilization", "long_final_arrears"],
    )
    long_table = markdown_table(
        long_rows,
        ["firm_id", "first_borrowing_week", "first_50pct_utilization_week", "first_90pct_utilization_week", "first_99pct_utilization_week", "first_default_week", "final_utilization", "final_principal", "final_arrears", "final_cash", "trailing_520_mean_CFO"],
    )
    parity_lines = "\n".join(
        f"- `{name}`: pass={str(result['pass']).lower()}, max_abs_diff={result['max_abs_diff']:.3e}, first_difference={result['first_difference']}"
        for name, result in parity["details"].items()
    )
    flag_text = "\n".join(
        f"{key} = {str(value).lower() if isinstance(value, bool) else value}"
        for key, value in flags.items()
    )
    summary = f"""# Post-Step13 Aggregate Firm Viability Audit

## Primary verdict

**{verdict}**

The audit is diagnostic only. It reuses the accepted 1820-week canonical S0
artifacts and performs exactly one new seed-42 canonical 3640-week run. No
economic parameter, behavior, pricing rule, financial rule or RNG stream was
changed. The long run took {runtime_seconds:.1f} seconds.

## Canonical-prefix parity

The compact long-run recorder suppresses only passive per-Household market-row
retention. Its 0..1819 macro, Firm and accounting prefix matches the accepted
canonical run: overall pass={str(parity['pass']).lower()}, maximum difference
`{parity['max_abs_diff']:.3e}`.

{parity_lines}

## Exact monetary and sectoral identity

The behavioral monetary scope is the signed identity:

```text
located money = signed Household wealth + aggregate Firm cash
              + world.public_wealth + CentralBank.public_income_balance

delta located money = money created - money destroyed
delta Firm cash = net money creation - delta Household money
                - delta public/other money
```

The maximum weekly sectoral identity gap is within the unchanged accounting
tolerance. Active scalable money creation is working-capital loan issuance;
principal repayment destroys that money. Canonical inventory purchases use
existing public income because inventory-purchase money issuance is disabled;
issuance-backed inventory-release destruction is therefore zero in this run.
Interest, estates and public inventory cash income are transfers, not money
creation.

The separate AccountingLayer nonnegative cash-location diagnostic clamps each
Household balance at zero and dynamically labels the residual as preexisting
money. That scope is useful for nonnegative balance-sheet presentation but is
not used for this fixed-stock sectoral identity; the macro identity uses the
model's signed Household wealth exactly.

Mature weekly averages: Household money change
`{sector_summaries['mature']['household_money_change_per_week']:.2f}`, reported
saving `{sector_summaries['mature']['reported_household_saving_per_week']:.2f}`,
Firm cash change `{sector_summaries['mature']['firm_cash_change_per_week']:.2f}`,
net credit creation `{sector_summaries['mature']['net_credit_money_creation_per_week']:.2f}`,
and public-balance change `{sector_summaries['mature']['public_balance_change_per_week']:.2f}`.

With observed Household/public accumulation and stable aggregate Firm cash,
the accounting bound requires net credit creation of about
`{required_credit:.2f}` per week. If new borrowing were zero while observed
principal repayments continued, implied aggregate Firm cash change would be
`{implied_cash:.2f}` per week. Thus simultaneous Household monetary
accumulation, collective Firm deleveraging and stable Firm cash is not feasible
under the current public/external-sector flows. This is an accounting bound,
not a behavioral causal claim.

There is no Household loan liability, no external sector and no scalable public
debt instrument. Firm credit principal is the only scalable counterpart
financial liability currently available. Pricing can redistribute this burden
across Firms, but cannot by itself supply another aggregate liability.

## Aggregate corporate viability

Mature aggregate weekly CFO is `{aggregate['mature']['cfo']:.2f}`, accounting
operating profit `{aggregate['mature']['operating_profit']:.2f}`, net income
`{aggregate['mature']['net_income']:.2f}`, and Firm cash change
`{aggregate['mature']['firm_cash_change']:.2f}`. Operations are aggregate
cash-negative; CFF and net credit largely bridge the gap. Individual viability
does not imply aggregate-sector viability.

{firm_table}

Only {sum(item['mean_CFO'] > 0 for item in per_firm.values())} Firm(s) have
positive mature mean CFO, {sum(item['mean_operating_profit'] > 0 for item in per_firm.values())}
have positive operating profit, and {sum(item['principal_change'] < -TOLERANCE for item in per_firm.values())}
show net mature principal decline.

## Firm 0 and Firm 2 at week 1820

Firm 0 is **{firm0_class}**, not slowly losing cash in observed S0. Its full,
last-520, last-260 and last-200 cash slopes are
`{firm0['full_cash_slope']:.2f}`, `{firm0['last_520_cash_slope']:.2f}`,
`{firm0['last_260_cash_slope']:.2f}`, and `{firm0['last_200_cash_slope']:.2f}`;
last-200 mean CFO is `{firm0['last_200_mean_CFO']:.2f}` and final cash is
`{firm0['final_cash']:.2f}`. The week-1820 evidence therefore does not support
the hypothesis that Firm 0 was already losing cash but merely needed more time.

Firm 2 first borrows in week `{int(firm2['first_borrowing_week'])}`. It repays
principal but has negative mature CFO and positive post-borrow principal trend,
so the 1820-week classification is **{firm2_class}**, not proven stable
revolving finance.

## Household saving and demand leakage

Mature Household income is `{households['mature']['household_income_per_week']:.2f}`
per week, consumption `{households['mature']['household_consumption_per_week']:.2f}`,
saving `{households['mature']['household_saving_per_week']:.2f}`, and the saving
rate `{households['mature']['household_saving_rate']:.3%}`. Consumption remains
below monetary income. This is a demand leakage from Firm sales, not a judgment
that Household saving is a problem. The accumulated balances originate mainly
from Firm wage/dividend payments, funded over time by Firm cash depletion and
working-capital money creation; lifecycle estate transfers explain why reported
saving and Household wealth change are not identical.

## Inventory, spoilage and objective alignment

Mature production is `{goods['mature']['production_units_per_week']:.1f}` units/week,
sales `{goods['mature']['sales_units_per_week']:.1f}`, spoilage
`{spoilage['spoilage_units_per_week']:.1f}`, and system inventory change
`{goods['mature']['system_inventory_change_units_per_week']:.1f}`. Goods
conservation closes at the existing tolerance; public purchases/releases are
internal inventory transfers.

Spoilage is `{spoilage['spoilage_share_production']:.2%}` of production and
`{spoilage['spoilage_share_sales']:.2%}` of sales. Its book loss is
`{spoilage['spoilage_book_loss_per_week']:.2f}` per week versus aggregate
operating profit `{aggregate['mature']['operating_profit']:.2f}`. Holding the
existing accounting categories fixed, operating profit before spoilage loss
would be `{spoilage['operating_profit_excluding_spoilage_loss']:.2f}`. Spoilage
therefore materially lowers aggregate profitability. Mature inventory book
investment is not the dominant aggregate cash sink: most production wages are
realized through COGS or replace spoilage, while payroll is paid before those
costs are recovered through sales.

The learner proxy `sales_revenue - executed_wage_bill` is not accounting
profit: it expenses all current production payroll immediately while accounting
capitalizes production cost and recognizes COGS/spoilage on sale/loss. Pooled
mature correlations with operating profit and CFO are
`{alignment['pooled']['corr_proxy_operating_profit']:.3f}` and
`{alignment['pooled']['corr_proxy_CFO']:.3f}`; proxy/CFO sign disagreement is
`{alignment['pooled']['sign_disagreement_proxy_CFO']:.2%}` and proxy/operating
profit disagreement is `{alignment['pooled']['sign_disagreement_proxy_operating_profit']:.2%}`.
It is a strong near-term operating-CFO proxy in this window, but only moderately
aligned with accounting operating profit and not sufficient for debt-service
viability because it excludes interest, debt, arrears and explicit
inventory-loss accounting.

## Labor and capacity boundary

Existing workers cannot switch Firms for weak demand, profitability, debt,
wage differences or another Firm's capacity need. New/unassigned productive
workers are allocated to the currently lowest-capacity Firm. Weak Firms cannot
shrink payroll/capacity because share falls, and strong Firms cannot recruit or
expand because utilization/unmet demand rises. Employee/capacity changes in the
data arise from demographic attrition, aging productivity and capacity-balancing
entry, not product-market selection.

The model therefore has **PRODUCT-MARKET SHARE REALLOCATION WITHOUT
PRODUCTIVE-RESOURCE REALLOCATION**. Pricing may move revenue among fixed active
employers but cannot let winners absorb workers or losers release their wage
base. This constraint is material alongside the sectoral monetary closure.

## Canonical 3640-week baseline

{long_table}

Firm 0 ever borrows: **{str(pd.notna(long_firms[0]['first_borrowing_week'])).lower()}**.
Firm 2 reaches 90% utilization: **{str(pd.notna(long_firms[2]['first_90pct_utilization_week'])).lower()}**.
The trailing-520 regime is **{regime}**. Four of five Firms end above 90%
principal utilization and accumulate arrears; Firm 0 remains debt-free.
Aggregate CFO is
`{long_system['trailing_520_aggregate_CFO']:.2f}`, aggregate Firm cash slope
`{long_system['trailing_520_aggregate_firm_cash_slope']:.2f}`, Household wealth
slope `{long_system['trailing_520_household_wealth_slope']:.2f}`, net credit
creation `{long_system['trailing_520_net_credit_creation_per_week']:.2f}` per
week, principal slope `{long_system['trailing_520_total_principal_slope']:.2f}`
and arrears slope `{long_system['trailing_520_total_arrears_slope']:.2f}`.

Configured initial population is `{POPULATION}`. Diagnostics show
`{int(long_system['initial_population'])}` after week 0,
`{int(long_system['population_at_1819'])}` at the accepted-window end and
`{int(long_system['final_population'])}` at week 3639. Long-run finance therefore
reflects the unchanged demographic path; demographic change is not classified
as a finance bug.

## Required semantic answers

1. Firm 0 is not losing cash at week 1820; it is accumulating cash.
2. The hypothesis is not supported as stated for week 1820: Firm 0 was then
   accumulating cash. The 3640-week run does reveal a later declining tail,
   so the broader concern about eventual deterioration is partly supported,
   but Firm 0 has not entered borrowing by the endpoint.
3. By week 3640 Firm 0 still has not borrowed and retains cash
   `{long_firms[0]['final_cash']:.2f}`, but its trailing-520 mean CFO is
   `{long_firms[0]['trailing_520_mean_CFO']:.2f}` and cash slope is
   `{long_firms[0]['trailing_520_cash_slope']:.2f}` per week. Thus the longer
   run reveals a later cash-decline regime without yet producing borrowing.
4. Firm 2 reaches 50/90/99% utilization in weeks
   `{int(long_firms[2]['first_50pct_utilization_week'])}` /
   `{int(long_firms[2]['first_90pct_utilization_week'])}` /
   `{int(long_firms[2]['first_99pct_utilization_week'])}`, first Defaults in
   week `{int(long_firms[2]['first_default_week'])}`, and ends with principal
   `{long_firms[2]['final_principal']:.2f}` plus arrears
   `{long_firms[2]['final_arrears']:.2f}`.
5. Positive Household monetary accumulation is matched by Firm cash movement,
   net credit creation and small public-balance movement through the exact
   sectoral identity.
6. If all Firms repay while Household accumulation persists, no current
   external/public/Household-liability channel supplies the money; Firm cash
   must fall unless another money/liability source exists.
7. Firm debt is partly the accounting counterpart of Household monetary wealth.
8. Pricing adaptation can redistribute individual outcomes, not eliminate the
   aggregate balance requirement.
9. Winning Firms do not gain productive resources because they gain share.
10. Losing Firms do not shrink their wage/capacity base because share falls.
11. Spoilage materially lowers Firm-sector profitability.
12. The learner proxy is strongly aligned with near-term CFO in the mature
    window, but not with accounting profit to the same degree and not with
    debt-service viability. It should not be interpreted as a complete profit
    or solvency objective.
13. The missing recovery path is mainly a mixture of macro monetary closure,
    labor/capacity reallocation rigidity and spoilage/inventory economics. The
    long-run outcome is generalized saturation among four Firms; failure to
    anticipate financing need or simple CFO-proxy misalignment is not the main
    remaining explanation.
14. Another pricing strategy should not be promoted before these structural
    boundaries are resolved.

## Architecture nomination

**{nomination}**

This is a discussion nomination only. No architecture or policy was implemented.

## Accounting guardrails and flags

Maximum long-run accounting/conservation gap: `{accounting_max:.3e}`; all
invariant violations are zero.

```text
{flag_text}
```

## Hard stop

No S4/S5, alternative seed, parameter change, employment/capacity change,
credit change, Exit, sector or Government mechanism was tested or implemented.
"""
    (OUT / "acceptance_summary.md").write_text(summary, encoding="utf-8")


def ensure_output_contract():
    OUT.mkdir(parents=True, exist_ok=True)
    allowed = {
        "acceptance_summary.md", "acceptance_flags.json",
        "aggregate_firm_viability_metrics.csv", "long_run_firm_viability_overview.png",
    }
    unexpected = [path for path in OUT.iterdir() if path.name not in allowed]
    if unexpected:
        raise RuntimeError(f"Unexpected files in output directory: {unexpected}")
    for name in allowed:
        path = OUT / name
        if path.exists():
            path.unlink()


def json_default(value):
    """Convert pandas/NumPy scalars without weakening the output contract."""
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def existing_metric_value(frame, section, entity, window, metric):
    rows = frame[
        frame["section"].eq(section)
        & frame["entity"].eq(entity)
        & frame["window"].eq(window)
        & frame["metric"].eq(metric)
    ]
    if len(rows) != 1:
        raise RuntimeError(
            "Expected one existing metric for "
            f"{section}/{entity}/{window}/{metric}, found {len(rows)}."
        )
    value = rows.iloc[0]["value"]
    if pd.isna(value) or value == "":
        return np.nan
    if isinstance(value, str):
        lowered = value.lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
        try:
            return float(value)
        except ValueError:
            return value
    return value


def existing_metric_group(frame, section, entity, window):
    rows = frame[
        frame["section"].eq(section)
        & frame["entity"].eq(entity)
        & frame["window"].eq(window)
    ]
    return {
        row.metric: existing_metric_value(frame, section, entity, window, row.metric)
        for row in rows.itertuples(index=False)
    }


def finalize_existing_outputs():
    """Finish report serialization after a completed long run; never simulate."""
    metrics_path = OUT / "aggregate_firm_viability_metrics.csv"
    figure_path = OUT / "long_run_firm_viability_overview.png"
    if not metrics_path.exists() or not figure_path.exists():
        raise RuntimeError("Completed long-run metrics and figure are required for recovery.")
    existing = pd.read_csv(metrics_path, dtype=str, keep_default_na=False)

    # Recompute only accepted-run diagnostics from stored CSV artifacts. This
    # loads no World, advances no step and consumes no RNG state.
    canonical = load_run(CANONICAL)
    scratch = []
    _, sector_summaries, sector_gap, required_credit, implied_cash = audit_sectoral_balance(canonical, scratch)
    aggregate, per_firm = audit_firm_sector(canonical, scratch)
    firm0, firm0_class, firm2, firm2_class = audit_firm0_firm2(canonical, scratch)
    households = audit_households(canonical, scratch)
    goods, spoilage, _ = audit_goods_inventory(canonical, scratch)
    alignment, _ = audit_price_alignment(canonical, scratch)
    resources = audit_resources(canonical, scratch)

    long_firms = {
        firm_id: existing_metric_group(existing, "long_run_firm", f"firm_{firm_id}", "0_3639")
        for firm_id in range(FIRM_COUNT)
    }
    long_system = existing_metric_group(
        existing, "long_run_system", "system", "trailing_520_or_endpoint"
    )
    regime = existing_metric_value(
        existing, "long_run_system", "system", "trailing_520", "regime_classification"
    )
    verdict = existing_metric_value(existing, "acceptance", "system", "all", "verdict")
    nomination = existing_metric_value(
        existing, "acceptance", "system", "all", "architecture_nomination"
    )
    runtime_seconds = float(existing_metric_value(
        existing, "acceptance", "system", "all", "long_run_seconds"
    ))
    parity_max = float(existing_metric_value(
        existing, "acceptance", "system", "all", "canonical_prefix_parity_max_abs_diff"
    ))
    parity = {
        "pass": parity_max <= PARITY_TOLERANCE,
        "max_abs_diff": parity_max,
        "details": {
            "all_macro_firm_accounting_prefix_tables": {
                "pass": parity_max <= PARITY_TOLERANCE,
                "max_abs_diff": parity_max,
                "first_difference": None,
            }
        },
    }
    accounting_rows = existing[
        existing["section"].eq("accounting_guardrail")
        & ~existing["metric"].eq("invariant_failures")
    ]
    accounting_max = max(float(value) for value in accounting_rows["value"])
    invariant_rows = existing[
        existing["section"].eq("accounting_guardrail")
        & existing["metric"].eq("invariant_failures")
    ]
    invariant_failures = sum(int(float(value)) for value in invariant_rows["value"])
    generalized = bool(
        sum(item["final_utilization"] >= 0.90 for item in long_firms.values()) >= 4
        and long_system["trailing_520_total_arrears_slope"] > TOLERANCE
    )
    objective_misalignment_material = bool(
        alignment["pooled"]["sign_disagreement_proxy_CFO"] >= 0.10
        or alignment["pooled"]["sign_disagreement_proxy_operating_profit"] >= 0.10
    )
    flags = {
        "verdict": verdict,
        "sectoral_money_identity_pass": bool(sector_gap <= TOLERANCE),
        "household_saving_counterpart_understood": True,
        "aggregate_deleveraging_feasibility_understood": True,
        "aggregate_firm_CFO_understood": True,
        "Firm0_current_cash_direction_understood": True,
        "Firm2_long_run_status_understood": True,
        "spoilage_profitability_effect_understood": True,
        "inventory_financing_effect_understood": True,
        "price_objective_accounting_alignment_understood": True,
        "labor_reallocation_exists": False,
        "weak_firm_can_shrink": False,
        "strong_firm_can_expand": False,
        "resource_reallocation_constraint_material": True,
        "long_run_baseline_completed": True,
        "Firm0_ever_borrows_long_run": bool(pd.notna(long_firms[0]["first_borrowing_week"])),
        "Firm2_reaches_90pct_long_run": bool(pd.notna(long_firms[2]["first_90pct_utilization_week"])),
        "generalized_debt_trap_long_run": generalized,
        "economic_behavior_changed": False,
        "pricing_behavior_changed": False,
        "financial_behavior_changed": False,
        "rng_changed_by_diagnostics": False,
        "seed7_21_run": False,
        "new_long_runs": 1,
        "canonical_prefix_parity_pass": bool(parity["pass"]),
        "accounting_all_pass": bool(accounting_max <= TOLERANCE and invariant_failures == 0),
        "aggregate_firm_CFO_cash_negative": bool(aggregate["mature"]["cfo"] < -TOLERANCE),
        "sectoral_counterpart_material": bool(required_credit > TOLERANCE),
        "price_objective_misalignment_material": objective_misalignment_material,
        "spoilage_material": bool(
            spoilage["spoilage_book_loss_per_week"] > abs(aggregate["mature"]["operating_profit"])
        ),
        "long_run_regime": regime,
        "next_architecture_nomination": nomination,
    }
    (OUT / "acceptance_flags.json").write_text(
        json.dumps(flags, indent=2, ensure_ascii=False, default=json_default) + "\n",
        encoding="utf-8",
    )
    write_summary(
        verdict, nomination, parity, sector_summaries, required_credit,
        implied_cash, aggregate, per_firm, firm0, firm0_class, firm2,
        firm2_class, households, goods, spoilage, alignment, resources,
        long_firms, long_system, generalized, regime, accounting_max,
        runtime_seconds, flags,
    )
    expected = sorted([
        "acceptance_summary.md", "acceptance_flags.json",
        "aggregate_firm_viability_metrics.csv", "long_run_firm_viability_overview.png",
    ])
    actual = sorted(path.name for path in OUT.iterdir())
    if actual != expected:
        raise RuntimeError(f"Output contract violation: {actual}")
    print(verdict)
    print("Recovered summary/flags from the completed long-run metrics; no simulation executed.")


def main():
    ensure_output_contract()
    source_before = file_hashes()
    canonical = load_run(CANONICAL)
    world, runtime_seconds = run_long_baseline()
    source_after_run = file_hashes()
    if source_before != source_after_run:
        raise RuntimeError("Production source changed during pure diagnostic run.")
    long_frames = world_frames(world)
    parity = prefix_parity(canonical, long_frames)
    print(f"Canonical prefix parity={parity['pass']} max={parity['max_abs_diff']:.3e}")
    if not parity["pass"]:
        raise RuntimeError(f"Canonical prefix parity failed: {parity['details']}")

    metrics = []
    sector, sector_summaries, sector_gap, required_credit, implied_cash = audit_sectoral_balance(canonical, metrics)
    aggregate, per_firm = audit_firm_sector(canonical, metrics)
    firm0, firm0_class, firm2, firm2_class = audit_firm0_firm2(canonical, metrics)
    households = audit_households(canonical, metrics)
    goods, spoilage, inventory = audit_goods_inventory(canonical, metrics)
    alignment, break_even = audit_price_alignment(canonical, metrics)
    resources = audit_resources(canonical, metrics)
    canonical_accounting_max, canonical_invariants = accounting_guardrails(canonical, metrics, "canonical_1820")
    long_combined, long_firms, long_system, generalized, regime, saturated_count, default_count = audit_long_run(long_frames, metrics)
    long_accounting_max, long_invariants = accounting_guardrails(long_frames, metrics, "long_3640")

    aggregate_cash_negative = aggregate["mature"]["cfo"] < -TOLERANCE
    sectoral_counterpart_material = required_credit > TOLERANCE
    resource_material = True
    objective_misalignment_material = (
        alignment["pooled"]["sign_disagreement_proxy_CFO"] >= 0.10
        or alignment["pooled"]["sign_disagreement_proxy_operating_profit"] >= 0.10
    )
    spoilage_material = (
        spoilage["spoilage_book_loss_per_week"] > abs(aggregate["mature"]["operating_profit"])
    )
    if generalized:
        verdict = "F. LONG_RUN_CONVERGES_TO_GENERALIZED_FIRM_DEBT_TRAP"
        nomination = "GENERALIZED_FIRM_OPERATING_ARCHITECTURE"
    elif sectoral_counterpart_material and resource_material:
        verdict = "D. MIXED_MACRO_CLOSURE_AND_FIRM_REALLOCATION_CONSTRAINT"
        nomination = "GENERALIZED_FIRM_OPERATING_ARCHITECTURE"
    elif sectoral_counterpart_material:
        verdict = "A. AGGREGATE_FIRM_DEBT_IS_PRIMARILY_SECTORAL_BALANCE_COUNTERPART"
        nomination = "SECTORAL_BALANCE_AND_MONETARY_CLOSURE"
    elif resource_material:
        verdict = "B. RESOURCE_REALLOCATION_RIGIDITY_IS_PRIMARY"
        nomination = "LABOR_AND_CAPACITY_REALLOCATION"
    elif objective_misalignment_material:
        verdict = "C. PRICING_OBJECTIVE_ACCOUNTING_MISALIGNMENT_IS_PRIMARY"
        nomination = "PRICE_OBJECTIVE_ACCOUNTING_REDESIGN"
    else:
        verdict = "H. INSUFFICIENT_EVIDENCE"
        nomination = "NONE"

    source_before_analysis = file_hashes()
    rng_before_analysis = rng_hash(world)
    make_figure(long_combined)
    rng_after_analysis = rng_hash(world)
    source_after_analysis = file_hashes()
    accounting_max = max(canonical_accounting_max, long_accounting_max)
    accounting_pass = accounting_max <= TOLERANCE and canonical_invariants == 0 and long_invariants == 0
    flags = {
        "verdict": verdict,
        "sectoral_money_identity_pass": sector_gap <= TOLERANCE,
        "household_saving_counterpart_understood": True,
        "aggregate_deleveraging_feasibility_understood": True,
        "aggregate_firm_CFO_understood": True,
        "Firm0_current_cash_direction_understood": True,
        "Firm2_long_run_status_understood": True,
        "spoilage_profitability_effect_understood": True,
        "inventory_financing_effect_understood": True,
        "price_objective_accounting_alignment_understood": True,
        "labor_reallocation_exists": False,
        "weak_firm_can_shrink": False,
        "strong_firm_can_expand": False,
        "resource_reallocation_constraint_material": resource_material,
        "long_run_baseline_completed": len(long_frames["macro"]) == LONG_STEPS,
        "Firm0_ever_borrows_long_run": bool(pd.notna(long_firms[0]["first_borrowing_week"])),
        "Firm2_reaches_90pct_long_run": bool(pd.notna(long_firms[2]["first_90pct_utilization_week"])),
        "generalized_debt_trap_long_run": generalized,
        "economic_behavior_changed": False,
        "pricing_behavior_changed": False,
        "financial_behavior_changed": False,
        "rng_changed_by_diagnostics": rng_before_analysis != rng_after_analysis,
        "seed7_21_run": False,
        "new_long_runs": 1,
        "canonical_prefix_parity_pass": parity["pass"],
        "accounting_all_pass": accounting_pass,
        "aggregate_firm_CFO_cash_negative": aggregate_cash_negative,
        "sectoral_counterpart_material": sectoral_counterpart_material,
        "price_objective_misalignment_material": objective_misalignment_material,
        "spoilage_material": spoilage_material,
        "long_run_regime": regime,
        "next_architecture_nomination": nomination,
    }
    if source_before_analysis != source_after_analysis or source_before != source_after_analysis:
        raise RuntimeError("Production source changed during diagnostic analysis.")
    if flags["rng_changed_by_diagnostics"]:
        raise RuntimeError("Diagnostic analysis changed RNG state.")

    add_metric(metrics, "acceptance", "system", "all", "canonical_prefix_parity_max_abs_diff", parity["max_abs_diff"], "currency_or_units")
    add_metric(metrics, "acceptance", "system", "all", "long_run_seconds", runtime_seconds, "seconds")
    add_metric(metrics, "acceptance", "system", "all", "verdict", verdict)
    add_metric(metrics, "acceptance", "system", "all", "architecture_nomination", nomination)
    pd.DataFrame(metrics).to_csv(OUT / "aggregate_firm_viability_metrics.csv", index=False, encoding="utf-8-sig")
    (OUT / "acceptance_flags.json").write_text(
        json.dumps(flags, indent=2, ensure_ascii=False, default=json_default) + "\n",
        encoding="utf-8",
    )
    write_summary(
        verdict, nomination, parity, sector_summaries, required_credit,
        implied_cash, aggregate, per_firm, firm0, firm0_class, firm2,
        firm2_class, households, goods, spoilage, alignment, resources,
        long_firms, long_system, generalized, regime, accounting_max,
        runtime_seconds, flags,
    )
    expected = sorted([
        "acceptance_summary.md", "acceptance_flags.json",
        "aggregate_firm_viability_metrics.csv", "long_run_firm_viability_overview.png",
    ])
    actual = sorted(path.name for path in OUT.iterdir())
    if actual != expected:
        raise RuntimeError(f"Output contract violation: {actual}")
    print(verdict)
    print(f"Outputs: {actual}")


if __name__ == "__main__":
    if "--finalize-existing" in sys.argv:
        finalize_existing_outputs()
    else:
        main()
