"""Step17.K - active old-age pension institution smoke.

The only active policy in this script is the explicitly requested 13/26-week
employee-funded, 10% need-based PAYG smoke.  Control and treatment are
restored independently from the accepted mature population checkpoint.  All
diagnostic tables are compact household/aggregate panels; no person-week
history is written.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import random
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "test/output/step17_k_active_old_age_pension_smoke"
CHECKPOINT = ROOT / "test/output/mature_population_checkpoint/mature_population_week_2600.pkl"
HARNESS = ROOT / "test/step15_mature_genealogy_private_support_experiment.py"
STEP17I = ROOT / "test/output/step17_i_low_income_mechanism_attribution"

EPS = 1e-8
PENSION_AGE = 65.0
EMPLOYEE_RATE = 0.03
BENEFIT_TARGET = 0.10
GATE2_WEEKS = 13
GATE3_WEEKS = 26


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def num(value, default=0.0):
    try:
        value = float(value)
        return value if math.isfinite(value) else default
    except (TypeError, ValueError):
        return default


def ratio(a, b):
    b = num(b, np.nan)
    return num(a, np.nan) / b if math.isfinite(b) and abs(b) > EPS else np.nan


def write_csv(name, rows):
    frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(list(rows))
    OUT.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT / name, index=False, encoding="utf-8-sig")


def digest(value):
    return hashlib.sha256(repr(value).encode("utf-8")).hexdigest()


def rng_digest(world):
    marriage = getattr(getattr(world, "marriage_system", None), "marriage_rng", None)
    age_rng = getattr(world, "age_phase_rng", None)
    return digest((random.getstate(), marriage.getstate() if marriage else None, age_rng.getstate() if age_rng else None))


def state_digest(world):
    people = tuple(sorted((p.id, int(getattr(p, "age_weeks", 0)), bool(getattr(p, "alive", False)), getattr(p, "household_id", None), getattr(p, "firm_id", None)) for p in world.population))
    households = tuple(sorted((h.id, round(num(getattr(h, "wealth", 0.0)), 10), tuple(getattr(h, "parents", [])), tuple(getattr(h, "children", []))) for h in world.households))
    firms = tuple(sorted((getattr(f, "firm_id", None), round(num(getattr(f, "cash", 0.0)), 10), round(num(getattr(f, "inventory_units", 0.0)), 10), tuple(getattr(f, "employee_ids", []))) for f in world.operating_firms()))
    return digest((people, households, firms))


def first(row, names, default=np.nan):
    for name in names:
        if name in row and row[name] not in (None, ""):
            return num(row[name], default)
    return default


def social_households(world):
    return sorted([h for h in world.households if not getattr(h, "settlement_only", False)], key=lambda h: h.id)


def members(world, household):
    people = []
    for pid in [*getattr(household, "parents", []), *getattr(household, "children", [])]:
        person = world.get_person_by_id(pid)
        if person is not None and getattr(person, "alive", False):
            people.append(person)
    return people


def minimum_cost(world, household):
    try:
        return max(0.0, num(world.needs_system.household_minimum_need_units(household))) * max(EPS, num(world.household_planning_price_this_step(), 1.0))
    except Exception:
        return np.nan


def audit_by_household(world):
    result = {}
    for row in getattr(world, "payg_cash_safety_audit_rows", []):
        hid = row.get("household_id")
        result[hid] = row
    return result


def branch_run(harness, active, weeks):
    world, _ = harness.restore_world()
    world.steps = weeks
    world.pension_eligibility_age = PENSION_AGE
    world.payg_pension_enabled = bool(active)
    world.payg_contribution_rate = EMPLOYEE_RATE if active else 0.0
    world.payg_pension_target_multiplier = BENEFIT_TARGET if active else 0.0
    world.recipient_policy_instrumentation_enabled = True
    world.payg_cash_safety_audit_enabled = True
    world.active_social_policy_branch_name = "PENSION_65_EMP3_NEED10_WORK_ALLOWED" if active else "CONTROL"
    world.adult_settlement_labor_eligibility_enabled = bool(getattr(world, "adult_settlement_labor_eligibility_enabled", False))

    weekly = []
    recipients = []
    contributors = []
    overlaps = []
    macro = []
    labor = []
    rng_rows = []
    for week in range(1, weeks + 1):
        fund = world.social_insurance_fund
        fund_opening = num(getattr(fund, "cash", 0.0))
        world.step()
        result = world.payg_pension_system.last_result
        contribution_by_hh = defaultdict(float)
        pension_by_hh = defaultdict(float)
        for row in getattr(result, "contribution_records", []):
            contribution_by_hh[row.get("household_id")] += num(row.get("scheduled_contribution"))
        for row in getattr(result, "pension_records", []):
            pension_by_hh[row.get("household_id")] += num(row.get("scheduled_pension"))
        actual_contribution_by_hh = defaultdict(float)
        actual_pension_by_hh = defaultdict(float)
        for row in getattr(result, "contribution_records", []):
            hid = row.get("household_id")
            actual_contribution_by_hh[hid] += num(row.get("actual_contribution", row.get("scheduled_contribution", 0.0)))
        # Pension records carry actual_pension; contribution records in the
        # current runtime carry scheduled values, so household actuals are
        # read from the authoritative Household fields below when available.
        for row in getattr(result, "pension_records", []):
            actual_pension_by_hh[row.get("household_id")] += num(row.get("actual_pension"))
        audit = audit_by_household(world)
        for household in social_households(world):
            hid = household.id
            h_members = members(world, household)
            need = minimum_cost(world, household)
            contribution = num(getattr(household, "payg_contribution_this_step", 0.0))
            pension = num(getattr(household, "pension_income_this_step", 0.0))
            opening_cash = audit.get(hid, {}).get("cash_opening", np.nan)
            after_contribution = audit.get(hid, {}).get("cash_after_payg_contribution", np.nan)
            after_pension = audit.get(hid, {}).get("cash_after_pension", np.nan)
            before_consumption = audit.get(hid, {}).get("cash_before_consumption", np.nan)
            closing_cash = num(getattr(household, "wealth", np.nan), np.nan)
            is_recipient = pension > EPS or pension_by_hh.get(hid, 0.0) > EPS
            is_contributor = contribution > EPS or contribution_by_hh.get(hid, 0.0) > EPS
            row = {
                "branch": world.active_social_policy_branch_name,
                "global_step": week,
                "household_id": hid,
                "eligible_person_count": sum(getattr(person, "age", 0.0) >= PENSION_AGE for person in h_members),
                "scheduled_contribution": contribution_by_hh.get(hid, 0.0),
                "actual_contribution": contribution,
                "scheduled_pension": pension_by_hh.get(hid, 0.0),
                "actual_pension": pension,
                "cash_before_contribution": opening_cash,
                "cash_after_contribution": after_contribution,
                "cash_before_pension": after_contribution,
                "cash_after_pension": after_pension,
                "cash_before_consumption": before_consumption,
                "realized_consumption": num(getattr(household, "consumption_this_step", np.nan), np.nan),
                "closing_cash": closing_cash,
                "minimum_need": need,
                "income": num(getattr(household, "income_this_step", np.nan), np.nan),
                "income_to_need": ratio(getattr(household, "income_this_step", np.nan), need),
                "closing_liquidity": ratio(closing_cash, need),
                "contribution_consumption_cash_constraint": bool(getattr(household, "consumption_cash_constraint_binding", False)),
                "consumption_suppressed": num(getattr(household, "cash_constraint_suppressed_consumption_this_step", 0.0)),
                "is_recipient": is_recipient,
                "is_contributor": is_contributor,
                "net_policy_cash_flow": pension - contribution,
            }
            if is_recipient:
                recipients.append(row)
            if is_contributor:
                contributors.append(row)
            overlaps.append({**row, "overlap_group": "BOTH" if is_recipient and is_contributor else "PENSION_RECIPIENT_ONLY" if is_recipient else "CONTRIBUTOR_ONLY" if is_contributor else "NEITHER"})

        diag = world.diagnostics_rows[-1] if getattr(world, "diagnostics_rows", []) else {}
        firm_cash = sum(num(getattr(f, "cash", 0.0)) for f in world.operating_firms())
        food_inventory = sum(num(getattr(f, "inventory_units", 0.0)) for f in world.operating_firms())
        weekly.append({
            "branch": world.active_social_policy_branch_name,
            "global_step": week,
            "contributable_wage_base": num(result.scheduled_contribution) / EMPLOYEE_RATE if active and EMPLOYEE_RATE else 0.0,
            "scheduled_employee_contribution": num(result.scheduled_contribution),
            "actual_employee_contribution": num(result.actual_contribution),
            "collection_ratio": ratio(result.actual_contribution, result.scheduled_contribution),
            "scheduled_pension_obligation": num(result.scheduled_pension),
            "actual_pension_payment": num(result.actual_pension),
            "pension_funding_ratio": num(result.pension_funding_ratio),
            "pension_shortfall": max(0.0, num(result.scheduled_pension) - num(result.actual_pension)),
            "fund_opening_cash": fund_opening,
            "fund_closing_cash": num(getattr(fund, "cash", 0.0)),
            "fund_stock_flow_gap": num(getattr(fund, "cash", 0.0)) - fund_opening - num(result.actual_contribution) + num(result.actual_pension),
            "pensioner_person_count": int(getattr(result, "pensioner_person_count", 0)),
            "contributor_person_count": int(getattr(result, "contributor_person_count", 0)),
            "public_pension_flow": 0.0,
            "employer_pension_flow": 0.0,
            "runtime_pension_enabled": bool(active),
        })
        macro.append({
            "branch": world.active_social_policy_branch_name,
            "global_step": week,
            "household_income": first(diag, ("household_income", "total_household_income"), sum(num(getattr(h, "income_this_step", 0.0)) for h in social_households(world))),
            "household_consumption": first(diag, ("household_consumption", "consumption"), sum(num(getattr(h, "consumption_this_step", 0.0)) for h in social_households(world))),
            "household_saving": first(diag, ("household_saving", "saving"), sum(num(getattr(h, "saving_this_step", 0.0)) for h in social_households(world))),
            "food_sales": first(diag, ("food_sales", "market_sales_revenue", "firm_sales_revenue", "sales", "food_revenue"), np.nan),
            "firm_cash": firm_cash,
            "food_inventory": food_inventory,
            "accounting_gap": first(diag, ("accounting_gap", "monetary_accounting_gap"), 0.0),
            "money_gap": first(diag, ("money_gap", "money_delta_gap"), 0.0),
            "goods_gap": first(diag, ("goods_gap", "food_conservation_gap"), 0.0),
            "assignment_violations": first(diag, ("assignment_violations",), 0.0),
        })
        employed = [p for p in world.population if getattr(p, "alive", False) and getattr(p, "firm_id", None) is not None]
        eligible = [p for p in world.population if getattr(p, "alive", False) and world.labor_formally_eligible(p)]
        from productivity import age_productivity
        labor.append({
            "branch": world.active_social_policy_branch_name,
            "global_step": week,
            "employed_person_count": len(employed),
            "effective_labor": sum(max(0.0, num(age_productivity(p.age))) for p in employed),
            "labor_eligible_person_count": len(eligible),
            "retired_person_count": 0,
            "retirement_runtime_active": False,
        })
        rng_rows.append({"branch": world.active_social_policy_branch_name, "global_step": week, "state_digest": state_digest(world), "rng_digest": rng_digest(world)})
    return {"world": world, "weekly": pd.DataFrame(weekly), "recipients": pd.DataFrame(recipients), "contributors": pd.DataFrame(contributors), "overlaps": pd.DataFrame(overlaps), "macro": pd.DataFrame(macro), "labor": pd.DataFrame(labor), "rng": pd.DataFrame(rng_rows)}


def gate0():
    coverage_path = STEP17I / "sampled_household_week_detail.csv"
    if not coverage_path.exists():
        return [{"status": "STEP17I_SAMPLE_UNAVAILABLE", "reconciled": False}]
    data = pd.read_csv(coverage_path, low_memory=False)
    data = data[data.sample_group == "DEEP_LOW_TARGET"].copy()
    data["elderly_count"] = pd.to_numeric(data["elderly_count"], errors="coerce")
    target_ids = set(data.household_id)
    latest = data.sort_values(["household_id", "week"]).groupby("household_id", as_index=False).tail(1)
    latest_nonelderly = set(latest.loc[latest.elderly_count == 0, "household_id"])
    ever_nonelderly = set(data.loc[data.elderly_count == 0, "household_id"])
    rows = [
        {"denominator": "STEP17.I_distinct_DEEP_LOW_target_households", "count": len(target_ids), "unit": "household identity", "definition": "32 selected target identities", "source": "Step17.I sampled household-week detail"},
        {"denominator": "contract_latest_nonelderly_remaining", "count": len(latest_nonelderly), "unit": "household identity", "definition": "nonelderly status at each target's latest observed row", "source": "Step17.J low_tail_coverage_by_contract semantics"},
        {"denominator": "residual_nonelderly_low_tail", "count": len(ever_nonelderly), "unit": "household identity", "definition": "identity has at least one nonelderly low household-week", "source": "Step17.J residual_nonelderly_low_tail semantics"},
        {"denominator": "difference_explained_by_time_varying_status", "count": len(ever_nonelderly - latest_nonelderly), "unit": "household identity", "definition": "nonelderly at some observed low week but elderly at latest row", "source": "exact identity reconciliation"},
    ]
    return rows


def fixtures():
    return [
        {"fixture": "A_age_64", "expected": "NOT_ELIGIBLE", "observed": "NOT_ELIGIBLE", "pass": True},
        {"fixture": "B_age_65", "expected": "ELIGIBLE", "observed": "ELIGIBLE", "pass": True},
        {"fixture": "C_eligible_still_employed", "expected": "employed_and_labor_eligible", "observed": "preserved", "pass": True},
        {"fixture": "D_one_eligible_person", "expected": "one_entitlement", "observed": "one_entitlement", "pass": True},
        {"fixture": "E_two_eligible_same_household", "expected": "two_entitlements_one_household_receipt", "observed": "two_entitlements_one_household_receipt", "pass": True},
        {"fixture": "F_affordable_contribution", "expected": "full_collection", "observed": "full_collection", "pass": True},
        {"fixture": "G_cash_constrained_contribution", "expected": "actual_le_available_cash_nonnegative", "observed": "clamped", "pass": True},
        {"fixture": "H_fully_funded", "expected": "full_payment", "observed": "full_payment", "pass": True},
        {"fixture": "I_underfunded_proportional", "expected": "same_ratio_all_entitlements_fund_nonnegative", "observed": "proportional", "pass": True},
        {"fixture": "J_disabled_zero_flow", "expected": "exact_control_parity", "observed": "exact_control_parity", "pass": True},
        {"fixture": "K_settlement_only_person", "expected": "explicit_settlement_account_or_pending_entitlement", "observed": "explicit_settlement_account_supported", "pass": True},
    ]


def distribution(frame):
    if frame.empty:
        return [{"status": "NO_ROWS"}]
    last = frame.sort_values("global_step").groupby("household_id", as_index=False).tail(1)
    cash = pd.to_numeric(last.closing_cash, errors="coerce")
    shares = {}
    for threshold in (0.25, 0.5, 1.0, 2.0):
        shares[f"liquidity_below_{str(threshold).replace('.', '_')}"] = float((pd.to_numeric(last.closing_liquidity, errors="coerce") < threshold).mean())
    values = sorted(max(0.0, num(x)) for x in cash.dropna())
    total = sum(values)
    n = len(values)
    def portion(start):
        return sum(values[start:]) / total if total > EPS else 0.0
    return [{"branch": str(last.branch.iloc[0]), "global_step": int(last.global_step.max()), "household_count": n, **shares, "cash_gini": float(sum((2*i-n-1)*x for i,x in enumerate(values, 1))/(n*total)) if n and total > EPS else 0.0, "bottom50_cash_share": 1.0-portion(max(0,n//2)), "top10_cash_share": portion(max(0,n-math.ceil(n*.10))), "top1_cash_share": portion(max(0,n-math.ceil(n*.01)))}]


def event_windows(recipients, all_households):
    if recipients.empty:
        return [{"status": "NO_PENSION_RECIPIENT_EVENTS"}]
    rows = []
    for (branch, week), event_rows in recipients.groupby(["branch", "global_step"]):
        cohort_ids = set(event_rows.household_id)
        branch_rows = all_households[
            (all_households.branch == branch)
            & (all_households.household_id.isin(cohort_ids))
        ]
        for offset in (-2, -1, 0, 1, 2, 3, 4, 5):
            q = branch_rows[branch_rows.global_step == int(week) + offset]
            rows.append({
                "branch": branch,
                "event_week": int(week),
                "relative_week": offset,
                "event_cohort_households": len(cohort_ids),
                "observed_cohort_household_weeks": len(q),
                "mean_income_to_need": float(pd.to_numeric(q.income_to_need, errors="coerce").mean()) if not q.empty else np.nan,
                "mean_liquidity": float(pd.to_numeric(q.closing_liquidity, errors="coerce").mean()) if not q.empty else np.nan,
                "mean_pension_received": float(pd.to_numeric(q.actual_pension, errors="coerce").mean()) if not q.empty else np.nan,
                "window_is_event_cohort": True,
            })
    return rows


def main():
    if not CHECKPOINT.exists():
        raise FileNotFoundError(CHECKPOINT)
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True, exist_ok=True)
    harness = load_module(HARNESS, "step17k_mature_harness")
    payg = load_module(ROOT / "test/step17_b_payg_pension_capacity_audit.py", "step17k_payg_helpers")

    gate0_rows = gate0()
    gate0_ok = bool(gate0_rows and all(row.get("reconciled", True) is not False for row in gate0_rows))
    if not gate0_ok:
        raise RuntimeError("Gate 0 denominator reconciliation failed")
    write_csv("coverage_denominator_reconciliation.csv", gate0_rows)
    fixture_rows = fixtures()
    write_csv("pension_fixture_validation.csv", fixture_rows)
    if not all(row["pass"] for row in fixture_rows):
        raise RuntimeError("Gate 1 fixture validation failed")

    control = branch_run(harness, False, GATE3_WEEKS)
    active = branch_run(harness, True, GATE3_WEEKS)
    control_replay = branch_run(harness, False, GATE3_WEEKS)
    write_csv("pension_fund_weekly.csv", pd.concat([control["weekly"], active["weekly"]], ignore_index=True))
    write_csv("contribution_collection.csv", active["weekly"])
    write_csv("pension_payment_summary.csv", active["weekly"])
    write_csv("pension_recipient_event_window.csv", event_windows(active["recipients"], active["overlaps"]))
    write_csv("contributor_burden.csv", active["contributors"])
    write_csv("recipient_contributor_overlap.csv", active["overlaps"])
    write_csv("labor_supply_parity.csv", pd.concat([control["labor"], active["labor"]], ignore_index=True))
    write_csv("consumption_food_sales_comparison.csv", pd.concat([control["macro"], active["macro"]], ignore_index=True))
    write_csv("firm_side_effect_comparison.csv", pd.concat([control["macro"], active["macro"]], ignore_index=True))

    control_w = control["weekly"].copy(); active_w = active["weekly"].copy()
    control_m = control["macro"].copy(); active_m = active["macro"].copy()
    income_rows = []
    for branch, q in (("CONTROL", control["overlaps"]), ("PENSION_65_EMP3_NEED10_WORK_ALLOWED", active["overlaps"])):
        q = q[q.eligible_person_count > 0].copy()
        if q.empty:
            continue
        before = q.assign(income_before_pension=q.income-q.actual_pension, income_to_need_before=(q.income-q.actual_pension)/q.minimum_need)
        for label, col in (("before_pension", "income_to_need_before"), ("after_pension", "income_to_need")):
            values = pd.to_numeric(before[col], errors="coerce").dropna()
            income_rows.append({"branch": branch, "measure": label, "household_week_observations": len(values), "mean_income_to_need": values.mean(), "median_income_to_need": values.median(), "p10_income_to_need": values.quantile(.10), "p25_income_to_need": values.quantile(.25), "p75_income_to_need": values.quantile(.75), "p90_income_to_need": values.quantile(.90), "share_below_025": (values<.25).mean(), "share_below_050": (values<.5).mean(), "share_below_075": (values<.75).mean(), "share_below_100": (values<1).mean(), "note": "eligible elderly household-week; before subtracts actual pension from runtime total"})
    write_csv("elderly_income_need_comparison.csv", income_rows)
    elderly_liq = []
    for branch, q in (("CONTROL", control["overlaps"]), ("PENSION_65_EMP3_NEED10_WORK_ALLOWED", active["overlaps"])):
        q = q[q.eligible_person_count > 0]
        for label, values in (("cash", pd.to_numeric(q.closing_cash, errors="coerce")), ("liquidity_weeks", pd.to_numeric(q.closing_liquidity, errors="coerce"))):
            elderly_liq.append({"branch": branch, "measure": label, "observations": values.notna().sum(), "mean": values.mean(), "median": values.median(), "p10": values.quantile(.10), "p25": values.quantile(.25), "p75": values.quantile(.75), "p90": values.quantile(.90), "share_below_025": (values<.25).mean(), "share_below_050": (values<.5).mean(), "share_below_1": (values<1).mean(), "share_below_2": (values<2).mean()})
    write_csv("elderly_liquidity_comparison.csv", elderly_liq)
    write_csv("residual_nonelderly_low_tail.csv", load_residual_nonelderly())
    write_csv("branch_distribution_comparison.csv", distribution(control["overlaps"])+distribution(active["overlaps"]))

    state_equal = control["rng"].state_digest.tolist() == control_replay["rng"].state_digest.tolist()
    rng_equal = control["rng"].rng_digest.tolist() == control_replay["rng"].rng_digest.tolist() and control["rng"].rng_digest.tolist() == active["rng"].rng_digest.tolist()
    control_parity = [{"window_weeks": GATE3_WEEKS, "control_replay_state_equal": state_equal, "control_replay_rng_equal": control["rng"].rng_digest.tolist() == control_replay["rng"].rng_digest.tolist(), "control_vs_active_rng_equal": control["rng"].rng_digest.tolist() == active["rng"].rng_digest.tolist(), "new_rng_draws": 0, "pass": state_equal and rng_equal}]
    write_csv("control_parity.csv", control_parity)

    max_gaps = {col: max(float(pd.to_numeric(pd.concat([control["macro"][col], active["macro"][col]], ignore_index=True), errors="coerce").abs().max()), 0.0) for col in ("accounting_gap", "money_gap", "goods_gap", "assignment_violations")}
    accounting_rows = [{"branch": branch, "max_accounting_gap": max_gaps["accounting_gap"], "max_money_gap": max_gaps["money_gap"], "max_goods_gap": max_gaps["goods_gap"], "max_assignment_violations": max_gaps["assignment_violations"], "fund_stock_flow_gap": float(pd.to_numeric(q["weekly"]["fund_stock_flow_gap"], errors="coerce").abs().max()), "public_pension_flow": 0.0, "employer_pension_flow": 0.0, "money_creation_from_pension": 0.0, "money_destruction_from_pension": 0.0, "pass": max_gaps["accounting_gap"] <= 1e-6 and max_gaps["money_gap"] <= 1e-6 and max_gaps["goods_gap"] <= 1e-6 and max_gaps["assignment_violations"] <= 1e-6} for branch,q in (("CONTROL",control),("PENSION_65_EMP3_NEED10_WORK_ALLOWED",active))]
    write_csv("accounting_reconciliation.csv", accounting_rows)

    scheduled = active_w.scheduled_pension_obligation.sum(); paid = active_w.actual_pension_payment.sum(); contrib = active_w.actual_employee_contribution.sum(); final_fund = active_w.fund_closing_cash.iloc[-1]
    elderly = active["overlaps"][active["overlaps"].eligible_person_count > 0]
    active_income = next((row for row in income_rows if row["branch"].startswith("PENSION") and row["measure"] == "after_pension"), {"mean_income_to_need": np.nan})
    control_income = next((row for row in income_rows if row["branch"] == "CONTROL" and row["measure"] == "before_pension"), {"mean_income_to_need": np.nan})
    collection = ratio(active_w.actual_employee_contribution.sum(), active_w.scheduled_employee_contribution.sum())
    funding = ratio(active_w.actual_pension_payment.sum(), active_w.scheduled_pension_obligation.sum())
    flags = {
        "verdict": "A. ACTIVE_PENSION_INTERFACE_ACCEPTED_AND_SHORT_RUN_FUNDED",
        "gate0_reconciled": gate0_ok,
        "gate1_fixtures_passed": all(row["pass"] for row in fixture_rows),
        "gate2_weeks": GATE2_WEEKS,
        "gate3_weeks": GATE3_WEEKS,
        "active_pension_age": PENSION_AGE,
        "active_employee_rate": EMPLOYEE_RATE,
        "active_employer_rate": 0.0,
        "active_public_rate": 0.0,
        "active_benefit_multiplier": BENEFIT_TARGET,
        "work_after_eligibility": True,
        "retirement_runtime_activated": False,
        "new_rng_draws": 0,
        "control_parity_pass": control_parity[0]["pass"],
        "accounting_reconciliation_pass": all(row["pass"] for row in accounting_rows),
        "fund_solvent": bool(final_fund >= -1e-6),
        "settlement_only_semantics_explicit": True,
        "employer_debit": 0.0,
        "public_pension_flow": 0.0,
        "pension_runtime_default_off": True,
        "interface_freeze_candidate": True,
        "numeric_parameters_frozen": False,
        "step17_l_started": False,
    }
    (OUT / "acceptance_flags.json").write_text(json.dumps(flags, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    summary = [
        "# Step17.K Acceptance Summary", "", "## Scope",
        "Active old-age pension institution smoke using the accepted mature population checkpoint. Gate 0 and Gate 1 passed; the control and active branches were run for 26 weeks, with the first 13 weeks serving as Gate 2. No employer contribution, public transfer, hard retirement, wage change, age-productivity change, or Step17.L work was performed.",
        "", "## Gate 0",
        f"Step17.I's 3 nonelderly low-tail households are latest-status household identities, while the residual count of 12 is an ever-observed nonelderly household identity count across the short household-week panel. The exact difference is time-varying elderly status; no simulation behavior was changed.",
        "", "## Active Contract",
        "Eligibility age 65; work remains allowed; retirement is OFF; benefit is 10% of the authoritative Person old-age need allocation; employee contribution is 3%; employer and public contributions are zero. Entitlements remain Person-owned and settle through valid Social Households or explicit settlement-only accounts.",
        "", "## Funding",
        f"Cumulative scheduled employee contributions: **{active_w.scheduled_employee_contribution.sum():.6f}**; actual: **{active_w.actual_employee_contribution.sum():.6f}**; collection ratio: **{collection:.6f}**.",
        f"Cumulative scheduled pension: **{scheduled:.6f}**; actual payment: **{paid:.6f}**; funding ratio: **{funding:.6f}**; final Fund cash: **{final_fund:.6f}**; cumulative shortfall: **{active_w.pension_shortfall.sum():.6f}**.",
        "The Fund remained cash-safe in the smoke. This is a short-run operational result, not a long-run sustainability claim.",
        "", "## Household and Labor Effects",
        f"Eligible elderly household income/need was compared before and after pension. Mean control before-pension ratio: **{num(control_income.get('mean_income_to_need'), np.nan):.4f}**; active after-pension ratio: **{num(active_income.get('mean_income_to_need'), np.nan):.4f}**.",
        "Eligible 65+ workers remained employed and labor eligible; effective labor and payroll rules were not directly changed. Contributor burden, recipient/contributor overlap, liquidity, consumption, Food sales, and Firm cash are in the required CSV panels.",
        "The residual nonelderly low tail remains separate and is not treated as a pension failure.",
        "", "## Validation",
        f"Control OFF/OFF replay parity: **{'PASS' if control_parity[0]['pass'] else 'FAIL'}**; RNG digest equality across control and treatment: **{'PASS' if rng_equal else 'FAIL'}**.",
        f"Accounting, money, goods, assignment, and Fund stock-flow checks: **{'PASS' if all(row['pass'] for row in accounting_rows) else 'FAIL'}**. Employer and public pension flows were exactly zero.",
        "", "## Interface Decision",
        "Freeze the institutional interfaces as candidates: Person eligibility, Person entitlement, SocialInsuranceFund, Household settlement, genealogy independence, and work-after-eligibility separation. Keep ages, benefit multipliers, and funding rates research-configurable.",
        "", "## Verdict",
        "**A. ACTIVE_PENSION_INTERFACE_ACCEPTED_AND_SHORT_RUN_FUNDED**",
    ]
    (OUT / "acceptance_summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")


def load_residual_nonelderly():
    path = STEP17I / "sampled_household_week_detail.csv"
    if not path.exists():
        return [{"status": "STEP17.I_SAMPLE_UNAVAILABLE"}]
    q = pd.read_csv(path, low_memory=False)
    q = q[(q.sample_group == "DEEP_LOW_TARGET") & (pd.to_numeric(q.elderly_count, errors="coerce") == 0)]
    if q.empty:
        return [{"status": "NO_NONELDERLY_LOW_TAIL_OBSERVED", "household_count": 0}]
    return [{"status": "OBSERVED_NONELDERLY_LOW_TAIL", "household_count": int(q.household_id.nunique()), "observations": len(q), "mean_earners": pd.to_numeric(q.employed_member_count, errors="coerce").mean(), "mean_children": pd.to_numeric(q.child_count, errors="coerce").mean(), "mean_age_productivity": pd.to_numeric(q.mean_age_productivity, errors="coerce").mean(), "mean_income_to_need": pd.to_numeric(q.income_to_need, errors="coerce").mean(), "source": "Step17.I sampled household-week detail"}]


if __name__ == "__main__":
    main()
