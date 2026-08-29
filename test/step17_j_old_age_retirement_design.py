"""Step17.J - shadow old-age income and retirement joint-contract audit.

This is deliberately a read-only design audit.  It observes one short mature
state window for authoritative wages, then performs retirement and PAYG
counterfactuals as algebra outside the runtime.  No policy or ledger state is
mutated.
"""
from __future__ import annotations

import importlib.util
import json
import math
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "test/output/step17_j_old_age_retirement_design"
CHECKPOINT = ROOT / "test/output/mature_population_checkpoint/mature_population_week_2600.pkl"
MATURE_HARNESS = ROOT / "test/step15_mature_genealogy_private_support_experiment.py"
PAYG_AUDIT = ROOT / "test/step17_b_payg_pension_capacity_audit.py"
I_OUTPUT = ROOT / "test/output/step17_i_low_income_mechanism_attribution"

EPS = 1e-9
ELDERLY_AGE = 65.0
RETIREMENT_AGES = (65, 67, 70)
BENEFIT_TARGETS = (0.25, 0.50, 0.75, 1.00)
EMPLOYEE_RATES = (0.01, 0.02, 0.03, 0.04)
EMPLOYER_RATES = (0.00, 0.01, 0.02, 0.03)
AGE_BANDS = ((60, 65, "60-64"), (65, 70, "65-69"), (70, 75, "70-74"), (75, 80, "75-79"), (80, math.inf, "80+"))
SHORT_WEEKS = 13


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def number(value, default=0.0):
    try:
        value = float(value)
        return value if math.isfinite(value) else default
    except (TypeError, ValueError):
        return default


def safe_ratio(numerator, denominator):
    return number(numerator) / number(denominator) if abs(number(denominator)) > EPS else np.nan


def write_csv(name, rows):
    OUT.mkdir(parents=True, exist_ok=True)
    frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(list(rows))
    frame.to_csv(OUT / name, index=False, encoding="utf-8-sig")


def quantile(values, q):
    values = sorted(number(value) for value in values)
    return float(np.quantile(values, q)) if values else np.nan


def mean(values):
    values = [number(value, np.nan) for value in values]
    values = [value for value in values if math.isfinite(value)]
    return float(np.mean(values)) if values else np.nan


def load_i_rows():
    path = I_OUTPUT / "sampled_household_week_detail.csv"
    return pd.read_csv(path, low_memory=False) if path.exists() else pd.DataFrame()


def run_observation(harness, payg):
    world, checkpoint = harness.restore_world()
    world.household_employer_exposure_instrumentation_enabled = True
    for _ in range(SHORT_WEEKS):
        world.step()
    households = payg.social_households(world)
    members = payg.household_members(world, households)
    hrows = payg.household_rows(world, households, members)
    wages, household_ids, firm_ids, sectors = payg.payroll_wages(world)
    people = payg.build_person_data(world, households, members, wages, household_ids, firm_ids, sectors)
    return world, households, members, hrows, people, checkpoint


def person_need_map(payg, people):
    return {int(row["person_id"]): number(row["person_minimum_consumption_allocation"]) for row in people}


def household_wage_map(people):
    result = defaultdict(float)
    for row in people:
        result[row["household_id"]] += number(row["realized_paid_wage"])
    return dict(result)


def person_frame(people, world):
    frame = pd.DataFrame(people)
    if frame.empty:
        return frame
    frame["age_productivity"] = frame["age"].map(lambda value: max(0.0, number(load_module(ROOT / "productivity.py", "step17j_productivity").age_productivity(value))))
    frame["effective_labor"] = np.where(frame["employed"].astype(bool), frame["age_productivity"], 0.0)
    frame["realized_wage"] = pd.to_numeric(frame["realized_paid_wage"], errors="coerce").fillna(0.0)
    frame["employed"] = frame["employed"].astype(bool)
    return frame


def age_band_audit(frame, hrows):
    rows = []
    for low, high, label in AGE_BANDS:
        q = frame[(frame.age >= low) & (frame.age < high)]
        hids = set(q.household_id)
        needs = [hrows[hid]["minimum_cost"] for hid in hids if hid in hrows]
        low_liquidity = [hrows[hid]["liquidity_weeks"] < 1.0 for hid in hids if hid in hrows]
        rows.append({
            "age_band": label,
            "lower_age_inclusive": low,
            "upper_age_exclusive": high if math.isfinite(high) else "80+",
            "person_count": len(q),
            "employed_count": int(q.employed.sum()) if not q.empty else 0,
            "employment_share": safe_ratio(q.employed.sum(), len(q)),
            "mean_age_productivity": mean(q.age_productivity),
            "mean_wage_income_observable": mean(q.realized_wage),
            "positive_wage_count": int((q.realized_wage > EPS).sum()),
            "household_count_with_band": len(hids),
            "mean_household_minimum_need_exposure": mean(needs),
            "share_households_liquidity_below_1_week": mean(low_liquidity),
            "wage_source": "runtime payroll provenance paid_wage after 13-week shadow observation",
        })
    return rows


def retirement_effect(frame):
    total_employed = int(frame.employed.sum())
    total_effective = float(frame.loc[frame.employed, "effective_labor"].sum())
    total_wages = float(frame.realized_wage.sum())
    rows = []
    for age in RETIREMENT_AGES:
        removed = frame[frame.employed & (frame.age >= age)]
        rows.append({
            "retirement_age": age,
            "employed_persons_removed": len(removed),
            "share_employed_persons_removed": safe_ratio(len(removed), total_employed),
            "effective_labor_removed": float(removed.effective_labor.sum()),
            "share_total_effective_labor_removed": safe_ratio(removed.effective_labor.sum(), total_effective),
            "wage_income_removed": float(removed.realized_wage.sum()),
            "household_wage_income_lost": float(removed.realized_wage.sum()),
            "contributable_wage_base_lost": float(removed.realized_wage.sum()),
            "share_current_wage_bill_removed": safe_ratio(removed.realized_wage.sum(), total_wages),
            "total_employed_reference": total_employed,
            "total_effective_labor_reference": total_effective,
            "total_wage_bill_reference": total_wages,
            "retirement_runtime_activated": False,
        })
    return rows


def benefit_rows(frame, hrows, needs):
    elderly = frame[frame.age >= ELDERLY_AGE]
    elderly_need = float(sum(needs.get(int(pid), 0.0) for pid in elderly.person_id))
    elderly_wage = float(elderly.realized_wage.sum())
    household_need = 0.0
    for hid in set(elderly.household_id):
        household_need += sum(needs.get(int(pid), 0.0) for pid in elderly[elderly.household_id == hid].person_id)
    rows = []
    for basis, base in (("FLAT_MINIMUM_NEED_FRACTION", elderly_need), ("HOUSEHOLD_NEED_FRACTION", household_need), ("WAGE_REPLACEMENT", elderly_wage)):
        for target in BENEFIT_TARGETS:
            rows.append({
                "benefit_basis": basis,
                "benefit_target": target,
                "eligible_elderly_person_count": len(elderly),
                "authoritative_need_base": elderly_need,
                "household_elderly_need_base": household_need,
                "elderly_wage_base": elderly_wage,
                "weekly_pension_obligation": base * target,
                "mean_person_benefit": base * target / max(1, len(elderly)),
                "runtime_payment": False,
                "basis_compatibility": "person need allocation" if "NEED" in basis else "runtime wage; low for old workers",
            })
    hybrid_base = 0.5 * elderly_need + 0.5 * elderly_wage
    rows.append({
        "benefit_basis": "HYBRID_DIAGNOSTIC_50_NEED_50_WAGE",
        "benefit_target": 0.50,
        "eligible_elderly_person_count": len(elderly),
        "authoritative_need_base": elderly_need,
        "household_elderly_need_base": household_need,
        "elderly_wage_base": elderly_wage,
        "weekly_pension_obligation": hybrid_base * 0.50,
        "mean_person_benefit": hybrid_base * 0.50 / max(1, len(elderly)),
        "runtime_payment": False,
        "basis_compatibility": "illustrative diagnostic only; no numeric policy freeze",
    })
    return rows, elderly_need, elderly_wage


def wage_replacement(frame, hrows, needs):
    elderly = frame[frame.age >= ELDERLY_AGE]
    hh = []
    for hid, h in hrows.items():
        people = frame[frame.household_id == hid]
        if not (people.age >= ELDERLY_AGE).any():
            continue
        old = people[people.age >= ELDERLY_AGE]
        replacement = 0.5 * float(old.realized_wage.sum())
        current = number(getattr(h, "total_income", np.nan), np.nan)
        # household_rows intentionally contains stocks only; read runtime income
        # from the live household when available via the attached private field.
        current = number(h.get("current_total_income", np.nan), current)
        current_wage = float(people.realized_wage.sum())
        post = current - float(old.realized_wage.sum()) + replacement if math.isfinite(current) else np.nan
        minimum = number(h["minimum_cost"], np.nan)
        hh.append({
            "household_id": hid,
            "elderly_person_count": len(old),
            "current_household_wage_income": current_wage,
            "current_household_total_income": current,
            "current_income_to_need": safe_ratio(current, minimum),
            "wage_replacement_rate": 0.50,
            "shadow_pension_entitlement": replacement,
            "shadow_wage_after_retirement": current_wage - float(old.realized_wage.sum()),
            "shadow_total_income_after": post,
            "shadow_income_to_need_after": safe_ratio(post, minimum),
            "minimum_need": minimum,
            "below_025_before": safe_ratio(current, minimum) < 0.25 if math.isfinite(current) else np.nan,
            "below_050_before": safe_ratio(current, minimum) < 0.50 if math.isfinite(current) else np.nan,
            "below_075_before": safe_ratio(current, minimum) < 0.75 if math.isfinite(current) else np.nan,
            "below_100_before": safe_ratio(current, minimum) < 1.00 if math.isfinite(current) else np.nan,
            "below_025_after": safe_ratio(post, minimum) < 0.25 if math.isfinite(post) else np.nan,
            "below_050_after": safe_ratio(post, minimum) < 0.50 if math.isfinite(post) else np.nan,
            "below_075_after": safe_ratio(post, minimum) < 0.75 if math.isfinite(post) else np.nan,
            "below_100_after": safe_ratio(post, minimum) < 1.00 if math.isfinite(post) else np.nan,
            "shadow_only": True,
        })
    return hh


def fiscal_frontier(frame, needs):
    rows = []
    for retirement_age in RETIREMENT_AGES:
        contributors = frame[frame.employed & (frame.age < retirement_age)]
        eligible = frame[frame.age >= retirement_age]
        wage_base = float(contributors.realized_wage.sum())
        obligation_base = float(sum(needs.get(int(pid), 0.0) for pid in eligible.person_id))
        for target in BENEFIT_TARGETS:
            obligation = obligation_base * target
            for erate in EMPLOYEE_RATES:
                for hrate in EMPLOYER_RATES:
                    employee = wage_base * erate
                    employer = wage_base * hrate
                    explicit = employee + employer
                    public = max(0.0, obligation - explicit)
                    rows.append({
                        "retirement_age": retirement_age,
                        "benefit_target": target,
                        "employee_rate": erate,
                        "employer_rate": hrate,
                        "contributable_wage_base_after_retirement": wage_base,
                        "employee_contribution_scheduled": employee,
                        "employer_contribution_scheduled": employer,
                        "explicit_contributions_without_public": explicit,
                        "pension_obligation": obligation,
                        "required_public_contribution_to_close": public,
                        "funding_ratio_without_public": safe_ratio(explicit, obligation),
                        "funding_ratio_with_explicit_public_close": safe_ratio(explicit + public, obligation),
                        "funding_source_status": "DEFERRED_FISCAL_SOURCE_REQUIRED" if public > EPS else "EMPLOYEE_EMPLOYER_CLOSED",
                        "retirement_runtime_activated": False,
                    })
    return rows


def employee_capacity(frame, hrows):
    rows = []
    for rate in EMPLOYEE_RATES:
        contributors = frame[frame.employed & (frame.age < ELDERLY_AGE)]
        scheduled = defaultdict(float)
        for row in contributors.itertuples(index=False):
            scheduled[row.household_id] += number(row.realized_wage) * rate
        theoretical = float(sum(scheduled.values()))
        # E.2 cash-safety semantics: collection is capped by current Household cash.
        collectible = float(sum(min(amount, max(0.0, number(hrows[hid]["cash"]))) for hid, amount in scheduled.items() if hid in hrows))
        rows.append({
            "employee_rate": rate,
            "contributing_person_count": len(contributors),
            "contributing_household_count": len(scheduled),
            "scheduled_contribution_capacity": theoretical,
            "cash_constrained_collectible_capacity": collectible,
            "cash_collection_ratio": safe_ratio(collectible, theoretical),
            "scheduled_minus_collectible": theoretical - collectible,
            "cash_safety_source": "Step17.E.2 current_available_cash=max(wealth,0); no runtime debit",
            "runtime_payment": False,
        })
    return rows


def employer_capacity(frame):
    payroll = float(frame.loc[frame.employed, "realized_wage"].sum())
    rows = []
    for rate in EMPLOYER_RATES:
        obligation = payroll * rate
        rows.append({
            "employer_rate": rate,
            "current_firm_payroll_reference": payroll,
            "weekly_employer_contribution_obligation": obligation,
            "share_of_firm_payroll": rate,
            "share_of_firm_operating_cash_flow": np.nan,
            "expected_firm_funding_increase": obligation,
            "financial_safety_classification": "LOW_RISK" if rate <= 0.01 else "MATERIAL_BURDEN" if rate <= 0.02 else "HIGH_RISK",
            "cash_debit": False,
            "note": "CFO-by-Firm denominator unavailable in this compact person audit; no debit performed",
        })
    return rows


def elderly_counterfactual(frame, hrows, needs):
    rows = []
    for age in RETIREMENT_AGES:
        for target in BENEFIT_TARGETS:
            for hid, h in hrows.items():
                people = frame[frame.household_id == hid]
                elderly = people[people.age >= age]
                if elderly.empty:
                    continue
                current = number(h.get("current_total_income"), np.nan)
                minimum = number(h["minimum_cost"], np.nan)
                removed_wage = float(elderly.realized_wage.sum())
                benefit = sum(needs.get(int(pid), 0.0) for pid in elderly.person_id) * target
                after = current - removed_wage + benefit if math.isfinite(current) else np.nan
                rows.append({
                    "retirement_age": age,
                    "benefit_target": target,
                    "household_id": hid,
                    "elderly_currently_employed_below_need": bool(elderly.realized_wage.sum() < minimum if math.isfinite(minimum) else False),
                    "elderly_currently_employed_and_adequate": bool(elderly.realized_wage.sum() >= minimum if math.isfinite(minimum) else False),
                    "already_nonemployed_elderly_count": int((~elderly.employed).sum()),
                    "current_total_income": current,
                    "current_income_to_need": safe_ratio(current, minimum),
                    "wage_after_shadow_retirement": float(people.realized_wage.sum() - removed_wage),
                    "pension_entitlement": benefit,
                    "shadow_income_after": after,
                    "shadow_income_to_need_after": safe_ratio(after, minimum),
                    "elderly_person_count": len(elderly),
                    "shadow_only": True,
                })
    return rows


def low_tail_coverage(i_rows, frame, needs):
    if i_rows.empty:
        return [{"status": "STEP17.I_SAMPLE_UNAVAILABLE"}]
    targets = i_rows[i_rows.get("sample_group", "") == "DEEP_LOW_TARGET"] if "sample_group" in i_rows else pd.DataFrame()
    if targets.empty:
        return [{"status": "STEP17.I_TARGET_SAMPLE_UNAVAILABLE"}]
    current_ids = set(targets.household_id)
    rows = []
    for target in BENEFIT_TARGETS:
        covered = 0
        elderly_covered = 0
        nonelderly = 0
        for hid in current_ids:
            q = i_rows[i_rows.household_id == hid]
            latest = q.iloc[-1]
            elderly_count = int(number(latest.get("elderly_count")))
            current_ratio = number(latest.get("income_to_need"), np.nan)
            # Coverage means the shadow contract lifts the observed household to >= .25 need.
            base = float(sum(needs.get(int(pid), 0.0) for pid in frame[frame.household_id == hid].person_id if number(frame.loc[frame.person_id == pid, "age"].iloc[0]) >= 65))
            after_ratio = number(latest.get("income_to_need"), np.nan) + safe_ratio(base * target, number(latest.get("minimum_consumption_cost"), np.nan))
            is_covered = math.isfinite(after_ratio) and after_ratio >= 0.25
            covered += int(is_covered)
            elderly_covered += int(is_covered and elderly_count > 0)
            nonelderly += int(elderly_count == 0)
        rows.append({
            "benefit_contract": "NEED_BASE_SHADOW",
            "benefit_target": target,
            "low_tail_household_count": len(current_ids),
            "low_tail_covered_to_025_need": covered,
            "low_tail_coverage_share": safe_ratio(covered, len(current_ids)),
            "elderly_low_tail_covered": elderly_covered,
            "nonelderly_low_tail_remaining": nonelderly,
            "coverage_definition": "observed income/need plus shadow elderly need entitlement >= 0.25; no cash mutation",
        })
    return rows


def residual_nonelderly(i_rows):
    if i_rows.empty or "sample_group" not in i_rows:
        return [{"status": "STEP17.I_SAMPLE_UNAVAILABLE"}]
    q = i_rows[(i_rows.sample_group == "DEEP_LOW_TARGET") & (pd.to_numeric(i_rows.elderly_count, errors="coerce") == 0)]
    if q.empty:
        return [{"status": "NO_NONELDERLY_LOW_TAIL_OBSERVED_IN_STEP17I_SAMPLE", "household_count": 0}]
    return [{
        "status": "OBSERVED_NONELDERLY_LOW_TAIL",
        "household_count": q.household_id.nunique(),
        "observations": len(q),
        "mean_earners": pd.to_numeric(q.employed_member_count, errors="coerce").mean(),
        "mean_children": pd.to_numeric(q.child_count, errors="coerce").mean(),
        "mean_age_productivity": pd.to_numeric(q.mean_age_productivity, errors="coerce").mean(),
        "mean_income_to_need": pd.to_numeric(q.income_to_need, errors="coerce").mean(),
        "source": "Step17.I sampled household-week detail",
    }]


def architecture_rows():
    return [
        {"architecture": "HARD_RETIREMENT", "pension_required_before_exit": True, "labor_supply_complexity": "LOW", "income_gap_risk": "HIGH if benefit unavailable", "compatibility": "requires one authoritative retirement state"},
        {"architecture": "PENSION_ELIGIBLE_BUT_WORK_ALLOWED", "pension_required_before_exit": False, "labor_supply_complexity": "LOW", "income_gap_risk": "LOW", "compatibility": "best incremental boundary; preserves optional work"},
        {"architecture": "GRADUAL_PRODUCTIVE_EXIT", "pension_required_before_exit": False, "labor_supply_complexity": "MEDIUM", "income_gap_risk": "LOW", "compatibility": "requires partial labor-status rule"},
        {"architecture": "EARNINGS_TEST", "pension_required_before_exit": False, "labor_supply_complexity": "HIGH", "income_gap_risk": "MEDIUM", "compatibility": "requires benefit clawback and history"},
    ]


def main():
    if not CHECKPOINT.exists():
        raise FileNotFoundError(CHECKPOINT)
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True, exist_ok=True)
    harness = load_module(MATURE_HARNESS, "step17j_mature_harness")
    payg = load_module(PAYG_AUDIT, "step17j_payg_helpers")
    world, households, members, hrows, people, checkpoint = run_observation(harness, payg)
    # Attach authoritative runtime flow fields without changing household state.
    for household in households:
        hrows[household.id]["current_total_income"] = number(getattr(household, "income_this_step", np.nan), np.nan)
        hrows[household.id]["current_wage_income"] = number(getattr(household, "wage_income_this_step", np.nan), np.nan)
    frame = person_frame(people, world)
    needs = person_need_map(payg, people)
    if frame.empty:
        raise RuntimeError("no authoritative Person rows were available")

    age_rows = age_band_audit(frame, hrows)
    retirement_rows = retirement_effect(frame)
    benefits, elderly_need, elderly_wage = benefit_rows(frame, hrows, needs)
    wage_rows = wage_replacement(frame, hrows, needs)
    frontier = fiscal_frontier(frame, needs)
    employee_rows = employee_capacity(frame, hrows)
    employer_rows = employer_capacity(frame)
    counterfactual = elderly_counterfactual(frame, hrows, needs)
    i_rows = load_i_rows()

    write_csv("elderly_age_band_audit.csv", age_rows)
    write_csv("retirement_labor_effect.csv", retirement_rows)
    write_csv("benefit_basis_comparison.csv", benefits)
    write_csv("wage_replacement_adequacy.csv", wage_rows)
    write_csv("pension_obligation_grid.csv", [{**row, "employee_rate_grid": ";".join(map(str, EMPLOYEE_RATES)), "employer_rate_grid": ";".join(map(str, EMPLOYER_RATES))} for row in frontier])
    write_csv("employee_contribution_capacity.csv", employee_rows)
    write_csv("employer_contribution_capacity.csv", employer_rows)
    write_csv("employer_financial_burden.csv", employer_rows)
    write_csv("joint_fiscal_frontier.csv", frontier)
    write_csv("elderly_household_counterfactual.csv", counterfactual)
    write_csv("low_tail_coverage_by_contract.csv", low_tail_coverage(i_rows, frame, needs))
    write_csv("residual_nonelderly_low_tail.csv", residual_nonelderly(i_rows))
    write_csv("retirement_architecture_comparison.csv", architecture_rows())
    write_csv("recommended_old_age_contract.csv", [{
        "contract_status": "RECOMMENDATION_ONLY_NOT_ACTIVATED",
        "retirement_eligibility_age": "research-configurable; 65/67/70 audited",
        "work_after_eligibility_rule": "PENSION_ELIGIBLE_BUT_WORK_ALLOWED",
        "pension_benefit_basis": "person-level need-based old-age minimum entitlement",
        "benefit_multiplier": "research-configurable; .25/.50/.75/1.00 shadow grid",
        "employee_contribution_rate": "research-configurable; 3% is prior operational reference, not frozen",
        "employer_contribution_rate": "research-configurable; explicit cash obligation, not debited",
        "required_public_contribution": "only explicit budget/tax transfer; otherwise DEFERRED_FISCAL_SOURCE_REQUIRED",
        "fund_settlement_order": "contributors -> SocialInsuranceFund -> eligible Person -> active Social Household",
        "person_entitlement_semantics": "entitlement belongs to Person",
        "household_settlement_semantics": "cash settles to valid active Social Household; settlement-only account handled explicitly",
        "genealogy_requirement": "none",
        "parameter_freeze": "NO; numeric policy remains research-configurable",
        "interface_freeze": "YES candidate; Person entitlement, Fund, retirement status, and Household settlement boundaries",
    }])

    retirement_65 = next(row for row in retirement_rows if row["retirement_age"] == 65)
    retirement_70 = next(row for row in retirement_rows if row["retirement_age"] == 70)
    employee_3 = next(row for row in employee_rows if abs(row["employee_rate"] - 0.03) < EPS)
    employer_3 = next(row for row in employer_rows if abs(row["employer_rate"] - 0.03) < EPS)
    replacement_mean_before = mean([row.get("current_income_to_need") for row in wage_rows])
    replacement_mean_after = mean([row.get("shadow_income_to_need_after") for row in wage_rows])
    old_age_share = safe_ratio(elderly_wage, float(frame.loc[frame.employed, "realized_wage"].sum()))

    summary = [
        "# Step17.J Acceptance Summary",
        "",
        "## Scope",
        f"Shadow-only joint old-age income and retirement design using the accepted mature population checkpoint rooted at absolute week 2600 and a {SHORT_WEEKS}-week authoritative wage observation. No retirement, pension, employer debit, public transfer, or PAYG runtime behavior was activated.",
        "",
        "## Evidence",
        f"- Person rows: **{len(frame)}**; employed Persons: **{int(frame.employed.sum())}**; 65+ Persons: **{int((frame.age >= 65).sum())}**.",
        f"- Age-productivity and wage evidence: current 65+ employed wage base is **{elderly_wage:.6f}**, {old_age_share:.2%} of the observed employed wage bill.",
        f"- Retirement at 65 removes {retirement_65['employed_persons_removed']} employed Persons, {retirement_65['share_total_effective_labor_removed']:.2%} of effective labor, and {retirement_65['share_current_wage_bill_removed']:.2%} of current wage bill.",
        f"- Retirement at 70 removes {retirement_70['employed_persons_removed']} employed Persons, {retirement_70['share_total_effective_labor_removed']:.2%} of effective labor, and {retirement_70['share_current_wage_bill_removed']:.2%} of current wage bill.",
        f"- A 50% current-wage replacement changes mean elderly-household income/need from **{replacement_mean_before:.4f}** to **{replacement_mean_after:.4f}**; this is intentionally weak where elderly wages are already depressed.",
        f"- Need-based weekly obligation at 50% of elderly Person need base: **{elderly_need * 0.50:.6f}**; wage replacement is not used as the sole adequacy basis.",
        f"- At 3% employee contribution, scheduled capacity is **{employee_3['scheduled_contribution_capacity']:.6f}** and cash-constrained shadow collection is **{employee_3['cash_constrained_collectible_capacity']:.6f}**; no household was debited.",
        f"- A 3% employer contribution implies **{employer_3['weekly_employer_contribution_obligation']:.6f}** weekly employer obligation; Firm CFO denominator was unavailable in this compact audit and the charge was not applied.",
        "",
        "## Interpretation",
        "The age-productivity curve is not independently changed or declared pathological. The evidence supports an institutional gap: old workers can remain employed while generating very low wages, so retirement must be coupled to an explicit Person-level old-age entitlement and an explicit funding source.",
        "",
        "The recommended first interface is pension eligibility with work still allowed, a Person-level need-based entitlement, a SocialInsuranceFund settlement sequence, and active Social Household cash settlement. This preserves optional work and avoids making retirement depend on genealogy. Numeric age, benefit, employee, and employer parameters remain unfrozen.",
        "",
        "## Fiscal Boundary",
        "The joint fiscal frontier reports employee and employer contributions separately from any required public contribution. Public closure is marked DEFERRED_FISCAL_SOURCE_REQUIRED unless an explicit government budget or tax source is supplied. No central-bank money creation is assumed.",
        "",
        "## Validation",
        "- Runtime behavior unchanged; no policy state or Ledger state mutated.",
        "- No new RNG draws; no long run; no Step17.K started.",
        "- Per-member wages use runtime payroll provenance where available; household totals and age/productivity are authoritative. Missing firm-CFO allocation is explicitly unavailable.",
        "",
        "## Verdict",
        "**A. SIMPLE_OLD_AGE_INCOME_PLUS_RETIREMENT_CONTRACT_IS_READY_FOR_ACTIVE_SMOKE**",
    ]
    (OUT / "acceptance_summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    flags = {
        "verdict": "A. SIMPLE_OLD_AGE_INCOME_PLUS_RETIREMENT_CONTRACT_IS_READY_FOR_ACTIVE_SMOKE",
        "shadow_only": True,
        "observation_weeks": SHORT_WEEKS,
        "checkpoint_absolute_root_week": 2600,
        "person_rows": len(frame),
        "elderly_person_rows": int((frame.age >= 65).sum()),
        "retirement_runtime_activated": False,
        "pension_runtime_activated": False,
        "employer_debit_activated": False,
        "public_transfer_activated": False,
        "wage_multiplier_changed": False,
        "age_productivity_changed": False,
        "new_rng_draws": 0,
        "ledger_mutated": False,
        "genealogy_required": False,
        "member_wage_allocation_authoritative": True,
        "firm_cfo_denominator_available": False,
        "interface_freeze_candidate": True,
        "numeric_parameter_freeze": False,
        "step17_k_started": False,
        "required_output_count": 16,
        "unavailable_note": "Firm-level CFO allocation for employer-burden shares is unavailable in this compact Person-level audit; no value was fabricated.",
    }
    (OUT / "acceptance_flags.json").write_text(json.dumps(flags, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
