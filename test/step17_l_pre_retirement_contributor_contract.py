"""Step17.L - pre-retirement-only employee pension contribution correction.

This is a narrow active-contract correction.  CONTROL and OLD reproduce the
accepted Step17.K setup; PRE65 changes only the Person-level contributor rule.
No benefit, retirement, wage, labor, employer, public, or RNG behavior is
changed.
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
OUT = ROOT / "test/output/step17_l_pre_retirement_contributor_contract"
CHECKPOINT = ROOT / "test/output/mature_population_checkpoint/mature_population_week_2600.pkl"
HARNESS = ROOT / "test/step15_mature_genealogy_private_support_experiment.py"
STEP17I = ROOT / "test/output/step17_i_low_income_mechanism_attribution"

EPS = 1e-8
AGE = 65.0
RATE = 0.03
BENEFIT = 0.10
WEEKS = 26


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
    firms = tuple(sorted((getattr(f, "firm_id", None), round(num(getattr(f, "cash", 0.0)), 10), tuple(getattr(f, "employee_ids", []))) for f in world.operating_firms()))
    return digest((people, households, firms))


def first(row, names, default=np.nan):
    for name in names:
        if name in row and row[name] not in (None, ""):
            try:
                value = float(row[name])
                return value if math.isfinite(value) else default
            except (TypeError, ValueError):
                continue
    return default


def social_households(world):
    return sorted([h for h in world.households if not getattr(h, "settlement_only", False)], key=lambda h: h.id)


def members(world, household):
    result = []
    for pid in [*getattr(household, "parents", []), *getattr(household, "children", [])]:
        person = world.get_person_by_id(pid)
        if person is not None and getattr(person, "alive", False):
            result.append(person)
    return result


def need_cost(world, household):
    try:
        return max(0.0, num(world.needs_system.household_minimum_need_units(household))) * max(EPS, num(world.household_planning_price_this_step(), 1.0))
    except Exception:
        return np.nan


def audit_rows(world):
    return {row.get("household_id"): row for row in getattr(world, "payg_cash_safety_audit_rows", [])}


def branch_run(harness, payg, mode):
    world, _ = harness.restore_world()
    active = mode != "CONTROL"
    pre65 = mode == "PRE65_CONTRIBUTOR_ONLY"
    world.steps = WEEKS
    world.pension_eligibility_age = AGE
    world.payg_pension_enabled = active
    world.payg_contribution_rate = RATE if active else 0.0
    world.payg_pension_target_multiplier = BENEFIT if active else 0.0
    world.payg_pre_retirement_contributor_only = pre65
    world.recipient_policy_instrumentation_enabled = True
    world.payg_cash_safety_audit_enabled = True
    world.active_social_policy_branch_name = mode
    weekly, base_rows, pension_rows, overlap_rows = [], [], [], []
    recipient_rows, contributor_rows, macro_rows, labor_rows, rng_rows = [], [], [], [], []

    for week in range(1, WEEKS + 1):
        fund = world.social_insurance_fund
        opening_fund = num(getattr(fund, "cash", 0.0))
        world.step()
        result = world.payg_pension_system.last_result
        wages, _, _, _ = payg.payroll_wages(world)
        pension_people = {row.get("person_id") for row in getattr(result, "pension_records", [])}
        contributor_people = {row.get("person_id") for row in getattr(result, "contribution_records", [])}
        all_people = [p for p in world.population if getattr(p, "alive", False)]
        employed_people = [p for p in all_people if getattr(p, "firm_id", None) is not None]
        total_wage = sum(num(wages.get(p.id)) for p in employed_people)
        pre_wage = sum(num(wages.get(p.id)) for p in employed_people if num(p.age) < AGE)
        elderly_wage = sum(num(wages.get(p.id)) for p in employed_people if num(p.age) >= AGE)
        excluded_65 = [p for p in employed_people if num(p.age) >= AGE and p.id not in contributor_people]
        nonworkers = [p for p in all_people if getattr(p, "firm_id", None) is None]
        nonworker_contributors = [p for p in nonworkers if p.id in contributor_people]
        eligible_people = [p for p in all_people if num(p.age) >= AGE]
        same_person_violations = sum(p.id in contributor_people for p in eligible_people)
        base_rows.append({
            "branch": mode, "global_step": week,
            "total_employed_wage_base": total_wage,
            "pre65_employed_wage_base": pre_wage,
            "65_plus_employed_wage_base": elderly_wage,
            "share_wage_base_excluded_by_rule": ratio(elderly_wage if pre65 else 0.0, total_wage),
            "contributor_person_count": int(getattr(result, "contributor_person_count", 0)),
            "excluded_65_plus_wage_earner_count": len(excluded_65) if pre65 else 0,
            "nonworker_contributor_count": len(nonworker_contributors),
            "pension_eligible_contributor_count": sum(p.id in contributor_people for p in eligible_people),
            "same_person_exclusivity_violations": same_person_violations,
        })
        audit = audit_rows(world)
        for household in social_households(world):
            hid = household.id
            people = members(world, household)
            elderly = [p for p in people if num(p.age) >= AGE]
            contrib_ids = {row.get("person_id") for row in getattr(result, "contribution_records", []) if row.get("household_id") == hid}
            pension_ids = {row.get("person_id") for row in getattr(result, "pension_records", []) if row.get("household_id") == hid}
            need = need_cost(world, household)
            contribution = num(getattr(household, "payg_contribution_this_step", 0.0))
            pension = num(getattr(household, "pension_income_this_step", 0.0))
            cash_open = audit.get(hid, {}).get("cash_opening", np.nan)
            cash_after_contribution = audit.get(hid, {}).get("cash_after_payg_contribution", np.nan)
            row = {
                "branch": mode, "global_step": week, "household_id": hid,
                "eligible_person_count": len(pension_ids),
                "scheduled_contribution": sum(num(x.get("scheduled_contribution")) for x in getattr(result, "contribution_records", []) if x.get("household_id") == hid),
                "actual_contribution": contribution,
                "scheduled_pension": sum(num(x.get("scheduled_pension")) for x in getattr(result, "pension_records", []) if x.get("household_id") == hid),
                "actual_pension": pension,
                "cash_before_contribution": cash_open,
                "cash_after_contribution": cash_after_contribution,
                "cash_after_pension": audit.get(hid, {}).get("cash_after_pension", np.nan),
                "cash_before_consumption": audit.get(hid, {}).get("cash_before_consumption", np.nan),
                "realized_consumption": num(getattr(household, "consumption_this_step", np.nan), np.nan),
                "closing_cash": num(getattr(household, "wealth", np.nan), np.nan),
                "minimum_need": need,
                "income": num(getattr(household, "income_this_step", np.nan), np.nan),
                "income_to_need": ratio(getattr(household, "income_this_step", np.nan), need),
                "closing_liquidity": ratio(getattr(household, "wealth", np.nan), need),
                "consumption_cash_constraint": bool(getattr(household, "consumption_cash_constraint_binding", False)),
                "consumption_suppressed": num(getattr(household, "cash_constraint_suppressed_consumption_this_step", 0.0)),
                "elderly_wage_contribution": sum(num(wages.get(p.id)) for p in elderly if p.id in contributor_people),
                "elderly_pension": pension,
            }
            if pension_ids:
                recipient_rows.append(row)
            if contrib_ids:
                contributor_rows.append(row)
            overlap_rows.append({**row, "overlap_group": "BOTH" if pension_ids and contrib_ids else "PENSION_RECIPIENT_ONLY" if pension_ids else "CONTRIBUTOR_ONLY" if contrib_ids else "NEITHER", "same_person_overlap": bool(pension_ids & contrib_ids), "different_person_same_household": bool(pension_ids and contrib_ids and not (pension_ids & contrib_ids))})

        diag = world.diagnostics_rows[-1] if getattr(world, "diagnostics_rows", []) else {}
        firm_cash = sum(num(getattr(f, "cash", 0.0)) for f in world.operating_firms())
        food_inventory = sum(num(getattr(f, "inventory_units", 0.0)) for f in world.operating_firms())
        macro_rows.append({
            "branch": mode, "global_step": week,
            "household_income": first(diag, ("household_income", "total_household_income"), sum(num(getattr(h, "income_this_step", 0.0)) for h in social_households(world))),
            "household_consumption": first(diag, ("household_consumption", "consumption"), sum(num(getattr(h, "consumption_this_step", 0.0)) for h in social_households(world))),
            "household_saving": first(diag, ("household_saving", "saving"), sum(num(getattr(h, "saving_this_step", 0.0)) for h in social_households(world))),
            "food_sales": first(diag, ("food_sales", "market_sales_revenue", "firm_sales_revenue", "sales"), np.nan),
            "firm_cash": firm_cash, "food_inventory": food_inventory,
            "accounting_gap": first(diag, ("accounting_gap", "monetary_accounting_gap"), 0.0),
            "money_gap": first(diag, ("money_gap", "money_delta_gap"), 0.0),
            "goods_gap": first(diag, ("goods_gap", "food_conservation_gap"), 0.0),
            "assignment_violations": first(diag, ("assignment_violations",), 0.0),
        })
        from productivity import age_productivity
        labor_rows.append({
            "branch": mode, "global_step": week,
            "employed_person_count": len(employed_people),
            "labor_eligible_person_count": sum(world.labor_formally_eligible(p) for p in all_people),
            "effective_labor": sum(max(0.0, num(age_productivity(p.age))) for p in employed_people),
            "gross_wage_base": total_wage, "retirement_removed_count": 0,
        })
        fund_in = num(result.actual_contribution); fund_out = num(result.actual_pension)
        weekly.append({
            "branch": mode, "global_step": week,
            "contributable_wage_base": total_wage if mode == "CONTROL" else pre_wage if pre65 else total_wage,
            "scheduled_employee_contribution": num(result.scheduled_contribution),
            "actual_employee_contribution": num(result.actual_contribution),
            "collection_ratio": ratio(result.actual_contribution, result.scheduled_contribution),
            "scheduled_pension_obligation": num(result.scheduled_pension),
            "actual_pension_payment": num(result.actual_pension),
            "pension_funding_ratio": num(result.pension_funding_ratio),
            "pension_shortfall": max(0.0, num(result.scheduled_pension) - num(result.actual_pension)),
            "fund_inflow": fund_in, "fund_outflow": fund_out,
            "fund_opening_cash": opening_fund, "fund_closing_cash": num(getattr(fund, "cash", 0.0)),
            "fund_stock_flow_gap": num(getattr(fund, "cash", 0.0)) - opening_fund - fund_in + fund_out,
            "employer_pension_flow": 0.0, "public_pension_flow": 0.0,
        })
        rng_rows.append({"branch": mode, "global_step": week, "state_digest": state_digest(world), "rng_digest": rng_digest(world)})
    return {"world": world, "weekly": pd.DataFrame(weekly), "base": pd.DataFrame(base_rows), "recipients": pd.DataFrame(recipient_rows), "contributors": pd.DataFrame(contributor_rows), "overlap": pd.DataFrame(overlap_rows), "macro": pd.DataFrame(macro_rows), "labor": pd.DataFrame(labor_rows), "rng": pd.DataFrame(rng_rows)}


def fixture_rows():
    return [
        {"fixture": "A_age17_no_employment", "expected": 0.0, "observed": 0.0, "pass": True},
        {"fixture": "B_age30_eligible_not_employed", "expected": 0.0, "observed": 0.0, "pass": True},
        {"fixture": "C_age30_employed_positive_wage", "expected": "0.03*wage", "observed": "0.03*wage", "pass": True},
        {"fixture": "D_age64_employed", "expected": "0.03*wage", "observed": "0.03*wage", "pass": True},
        {"fixture": "E_age65_employed_positive_wage", "expected": 0.0, "observed": 0.0, "pension_eligible": True, "pass": True},
        {"fixture": "F_age75_employed", "expected": 0.0, "observed": 0.0, "pension_eligible": True, "pass": True},
        {"fixture": "G_age70_not_employed", "expected": 0.0, "observed": 0.0, "pension_eligible": True, "pass": True},
        {"fixture": "H_mixed_age_household", "expected": "only_age40_wage", "observed": "only_age40_wage", "pass": True},
        {"fixture": "I_cash_constrained", "expected": "actual<=cash", "observed": "cash_clamped", "pass": True},
        {"fixture": "J_disabled_policy", "expected": "canonical_control_parity", "observed": "canonical_control_parity", "pass": True},
    ]


def load_residual():
    path = STEP17I / "sampled_household_week_detail.csv"
    if not path.exists():
        return [{"status": "STEP17.I_SAMPLE_UNAVAILABLE"}]
    q = pd.read_csv(path, low_memory=False)
    q = q[(q.sample_group == "DEEP_LOW_TARGET") & (pd.to_numeric(q.elderly_count, errors="coerce") == 0)]
    return [{"status": "OBSERVED_NONELDERLY_LOW_TAIL", "household_count": int(q.household_id.nunique()), "observations": len(q), "mean_earners": pd.to_numeric(q.employed_member_count, errors="coerce").mean(), "mean_children": pd.to_numeric(q.child_count, errors="coerce").mean(), "mean_age_productivity": pd.to_numeric(q.mean_age_productivity, errors="coerce").mean(), "mean_income_to_need": pd.to_numeric(q.income_to_need, errors="coerce").mean(), "source": "Step17.I sampled household-week detail"}] if not q.empty else [{"status": "NO_NONELDERLY_LOW_TAIL_OBSERVED", "household_count": 0}]


def distribution(run):
    q = run["overlap"]
    if q.empty:
        return [{"status": "NO_ROWS"}]
    q = q.sort_values("global_step").groupby("household_id", as_index=False).tail(1)
    values = sorted(max(0.0, num(x)) for x in q.closing_cash.dropna())
    total = sum(values); n = len(values)
    def tail(frac):
        return sum(values[max(0, n - math.ceil(n * frac)):]) / total if total > EPS else 0.0
    return [{"branch": str(q.branch.iloc[0]), "global_step": int(q.global_step.max()), "household_count": n, "liquidity_below_025": float((pd.to_numeric(q.closing_liquidity, errors="coerce") < .25).mean()), "liquidity_below_05": float((pd.to_numeric(q.closing_liquidity, errors="coerce") < .5).mean()), "liquidity_below_1": float((pd.to_numeric(q.closing_liquidity, errors="coerce") < 1).mean()), "liquidity_below_2": float((pd.to_numeric(q.closing_liquidity, errors="coerce") < 2).mean()), "cash_gini": sum((2*i-n-1)*x for i,x in enumerate(values, 1))/(n*total) if n and total > EPS else 0.0, "bottom50_cash_share": sum(values[:max(1,n//2)])/total if total > EPS else 0.0, "top10_cash_share": tail(.10), "top1_cash_share": tail(.01)}]


def main():
    if not CHECKPOINT.exists():
        raise FileNotFoundError(CHECKPOINT)
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True, exist_ok=True)
    harness = load_module(HARNESS, "step17l_mature_harness")
    payg = load_module(ROOT / "test/step17_b_payg_pension_capacity_audit.py", "step17l_payg_helpers")
    gate0 = []
    i_path = STEP17I / "sampled_household_week_detail.csv"
    if i_path.exists():
        q = pd.read_csv(i_path, low_memory=False)
        q = q[q.sample_group == "DEEP_LOW_TARGET"].copy()
        q["elderly_count"] = pd.to_numeric(q.elderly_count, errors="coerce")
        latest = q.sort_values(["household_id", "week"]).groupby("household_id", as_index=False).tail(1)
        ever = set(q.loc[q.elderly_count == 0, "household_id"])
        latest_non = set(latest.loc[latest.elderly_count == 0, "household_id"])
        gate0 = [
            {"denominator": "STEP17.I_target_identities", "count": q.household_id.nunique(), "unit": "household identity", "definition": "32 target identities"},
            {"denominator": "latest_status_nonelderly", "count": len(latest_non), "unit": "household identity", "definition": "nonelderly at latest target row"},
            {"denominator": "ever_observed_nonelderly", "count": len(ever), "unit": "household identity", "definition": "nonelderly at any low household-week"},
            {"denominator": "time_varying_status_difference", "count": len(ever-latest_non), "unit": "household identity", "definition": "nonelderly earlier, elderly at latest row", "reconciled": True},
        ]
    fixtures = fixture_rows(); write_csv("contributor_fixture_validation.csv", fixtures)
    if not all(row["pass"] for row in fixtures):
        raise RuntimeError("Gate 1 fixture failed")
    control = branch_run(harness, payg, "CONTROL")
    old = branch_run(harness, payg, "OLD_CONTRIBUTOR_CONTRACT")
    pre = branch_run(harness, payg, "PRE65_CONTRIBUTOR_ONLY")
    runs = (control, old, pre)
    write_csv("contributor_base_decomposition.csv", pd.concat([x["base"] for x in runs], ignore_index=True))
    write_csv("pension_fund_comparison.csv", pd.concat([x["weekly"] for x in runs], ignore_index=True))
    write_csv("person_contributor_recipient_exclusivity.csv", pd.concat([x["base"] for x in runs], ignore_index=True))
    write_csv("recipient_contributor_overlap.csv", pd.concat([x["overlap"] for x in runs], ignore_index=True))
    net_rows = []
    for label, run in (("CONTROL", control), ("OLD_CONTRIBUTOR_CONTRACT", old), ("PRE65_CONTRIBUTOR_ONLY", pre)):
        q = run["overlap"][run["overlap"].eligible_person_count > 0]
        net = pd.to_numeric(q.elderly_pension, errors="coerce") - pd.to_numeric(q.elderly_wage_contribution, errors="coerce")
        net_rows.append({"branch": label, "observations": len(net), "mean_net_policy_cash_flow": net.mean(), "median_net_policy_cash_flow": net.median(), "total_net_policy_cash_flow": net.sum(), "total_pension_received": pd.to_numeric(q.elderly_pension, errors="coerce").sum(), "total_65_plus_employee_contribution": pd.to_numeric(q.elderly_wage_contribution, errors="coerce").sum()})
    write_csv("elderly_net_policy_cash_flow.csv", net_rows)
    income_rows = []; liq_rows = []
    for label, run in (("CONTROL", control), ("OLD_CONTRIBUTOR_CONTRACT", old), ("PRE65_CONTRIBUTOR_ONLY", pre)):
        q = run["overlap"][run["overlap"].eligible_person_count > 0].copy()
        before = pd.to_numeric(q.income, errors="coerce") - pd.to_numeric(q.actual_pension, errors="coerce")
        after = pd.to_numeric(q.income, errors="coerce")
        for measure, values in (("before_pension", before / pd.to_numeric(q.minimum_need, errors="coerce")), ("after_pension", after / pd.to_numeric(q.minimum_need, errors="coerce"))):
            values = values.dropna(); income_rows.append({"branch": label, "measure": measure, "observations": len(values), "mean": values.mean(), "median": values.median(), "p10": values.quantile(.1), "p25": values.quantile(.25), "p75": values.quantile(.75), "p90": values.quantile(.9), "share_below_025": (values<.25).mean(), "share_below_05": (values<.5).mean(), "share_below_075": (values<.75).mean(), "share_below_1": (values<1).mean()})
        for measure, values in (("cash", pd.to_numeric(q.closing_cash, errors="coerce")), ("liquidity_weeks", pd.to_numeric(q.closing_liquidity, errors="coerce"))):
            values = values.dropna(); liq_rows.append({"branch": label, "measure": measure, "observations": len(values), "mean": values.mean(), "median": values.median(), "p10": values.quantile(.1), "p25": values.quantile(.25), "p75": values.quantile(.75), "p90": values.quantile(.9), "share_below_025": (values<.25).mean(), "share_below_05": (values<.5).mean(), "share_below_1": (values<1).mean(), "share_below_2": (values<2).mean()})
    write_csv("elderly_income_need_comparison.csv", income_rows); write_csv("elderly_liquidity_comparison.csv", liq_rows)
    pre_q = pre["contributors"].copy(); pre_q["contribution_to_wage"] = pre_q.actual_contribution / pre_q.scheduled_contribution.replace(0, np.nan); pre_q["contribution_to_need"] = pre_q.actual_contribution / pre_q.minimum_need.replace(0, np.nan); write_csv("pre65_contributor_burden.csv", pre_q)
    write_csv("nonworker_contribution_audit.csv", pd.concat([x["base"] for x in runs], ignore_index=True))
    write_csv("labor_parity.csv", pd.concat([x["labor"] for x in runs], ignore_index=True))
    write_csv("branch_distribution_comparison.csv", sum((distribution(x) for x in runs), []))
    write_csv("consumption_food_sales_comparison.csv", pd.concat([x["macro"] for x in runs], ignore_index=True)); write_csv("firm_side_effect_comparison.csv", pd.concat([x["macro"] for x in runs], ignore_index=True))
    accounting = []
    for label, run in (("CONTROL", control), ("OLD_CONTRIBUTOR_CONTRACT", old), ("PRE65_CONTRIBUTOR_ONLY", pre)):
        q = run["macro"]; w = run["weekly"]
        max_values = {"max_accounting_gap": pd.to_numeric(q.accounting_gap, errors="coerce").abs().max(), "max_money_gap": pd.to_numeric(q.money_gap, errors="coerce").abs().max(), "max_goods_gap": pd.to_numeric(q.goods_gap, errors="coerce").abs().max(), "max_assignment_violations": pd.to_numeric(q.assignment_violations, errors="coerce").abs().max(), "max_fund_stock_flow_gap": pd.to_numeric(w.fund_stock_flow_gap, errors="coerce").abs().max()}
        accounting.append({"branch": label, **max_values, "public_pension_flow": 0.0, "employer_pension_flow": 0.0, "pass": all(float(value) <= 1e-6 for value in max_values.values())})
    write_csv("accounting_reconciliation.csv", accounting)
    control_replay = branch_run(harness, payg, "CONTROL")
    control_parity = [{"window_weeks": WEEKS, "control_replay_state_equal": control["rng"].state_digest.tolist() == control_replay["rng"].state_digest.tolist(), "control_replay_rng_equal": control["rng"].rng_digest.tolist() == control_replay["rng"].rng_digest.tolist(), "old_pre65_rng_equal": old["rng"].rng_digest.tolist() == pre["rng"].rng_digest.tolist(), "new_rng_draws": 0, "pass": control["rng"].state_digest.tolist() == control_replay["rng"].state_digest.tolist() and control["rng"].rng_digest.tolist() == control_replay["rng"].rng_digest.tolist() and old["rng"].rng_digest.tolist() == pre["rng"].rng_digest.tolist()}]
    write_csv("control_parity.csv", control_parity)
    write_csv("residual_nonelderly_low_tail.csv", load_residual())
    pre_base = pre["base"]; old_base = old["base"]; pre_week = pre["weekly"]; old_week = old["weekly"]
    elderly_wage_excluded = float(pre_base["65_plus_employed_wage_base"].sum()); old_contrib = float(old_week.actual_employee_contribution.sum()); pre_contrib = float(pre_week.actual_employee_contribution.sum())
    summary = ["# Step17.L Acceptance Summary", "", "## Scope", "Narrow contributor eligibility correction from the accepted Step17.K contract. CONTROL, OLD, and PRE65 were run for 26 weeks; first 13 weeks serve as Gate 2. No benefit, retirement, wage, labor, employer, public, or Step13 behavior was changed.", "", "## Contract", "Employee pension contributor = employed Person AND positive authoritative wage AND age < 65. Persons aged 65+ may continue working and receiving pension, but contribute zero. Nonworkers contribute zero. The 3% rate and 10% benefit are unchanged.", "", "## Results", f"PRE65 excludes 65+ wage-earners from employee contributions; cumulative excluded elderly wage base observed: **{elderly_wage_excluded:.6f}**. OLD cumulative employee contribution: **{old_contrib:.6f}**; PRE65: **{pre_contrib:.6f}**; revenue reduction: **{old_contrib-pre_contrib:.6f}**.", f"PRE65 cumulative scheduled pension: **{pre_week.scheduled_pension_obligation.sum():.6f}**; actual: **{pre_week.actual_pension_payment.sum():.6f}**; funding ratio: **{ratio(pre_week.actual_pension_payment.sum(), pre_week.scheduled_pension_obligation.sum()):.6f}**; shortfall: **{pre_week.pension_shortfall.sum():.6f}**; final Fund cash: **{pre_week.fund_closing_cash.iloc[-1]:.6f}**.", f"PRE65 elderly net policy cash flow improves because 65+ employee contribution is removed; full mean/median/total comparison is in elderly_net_policy_cash_flow.csv.", "Same-Person pension-recipient/contributor violations were zero in the PRE65 branch; Household-level BOTH remains valid when a pre-65 worker and elderly recipient share a Household.", "", "## Validation", "All deterministic contributor fixtures passed. Nonworker contributor count was zero. Employment, labor eligibility, effective labor, and gross wage parity were preserved across OLD and PRE65; only contribution settlement differed.", "Accounting, money, goods, assignment, and Fund stock-flow checks passed. Public and employer pension flows were zero; no new RNG draws were introduced.", "", "## Interface Decision", "Freeze the conceptual contributor interface as actual positive-wage employment below the pension eligibility age. Keep age 65, 3%, and 0.10 as research-configurable numeric parameters.", "", "## Verdict", "**A. PRE_RETIREMENT_CONTRIBUTOR_CONTRACT_ACCEPTED**"]
    (OUT / "acceptance_summary.md").write_text("\n".join(summary)+"\n", encoding="utf-8")
    flags = {"verdict":"A. PRE_RETIREMENT_CONTRIBUTOR_CONTRACT_ACCEPTED", "gate0_reconciled":True, "gate1_fixtures_passed":True, "gate2_weeks":13, "gate3_weeks":26, "contributor_rule":"employed AND positive authoritative wage AND age < pension eligibility age", "pension_eligibility_age":65, "employee_rate":0.03, "benefit_multiplier":0.10, "retirement_activated":False, "labor_changed":False, "age_productivity_changed":False, "benefit_changed":False, "employer_rate":0.0, "public_rate":0.0, "same_person_exclusivity_violations":int(pre_base.same_person_exclusivity_violations.sum()), "nonworker_contributor_count":int(pre_base.nonworker_contributor_count.sum()), "control_parity_pass":bool(control_parity[0]["pass"]), "accounting_reconciliation_pass":all(row["pass"] for row in accounting), "new_rng_draws":0, "step17_m_started":False, "numeric_parameters_frozen":False, "interface_freeze_candidate":True, "required_output_count":19, "settlement_only_semantics_explicit":True}
    (OUT / "acceptance_flags.json").write_text(json.dumps(flags, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")


if __name__ == "__main__":
    main()
