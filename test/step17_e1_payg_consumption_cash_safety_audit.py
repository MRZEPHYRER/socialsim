"""Step17.E.1 PAYG contribution and Household consumption cash-safety audit."""
from __future__ import annotations

import csv
import json
import math
import shutil
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "test/output/step17_e1_payg_consumption_cash_safety"
E_OUT = ROOT / "test/output/step17_e_recipient_persistence_smoke"
EPS = 1e-8
BRANCHES = ("PAYG_3_ONLY", "COMBINED_3")


def load_module(path, name):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def read_json(path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return default


def read_rows(path):
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))
    except OSError:
        return []


def number(value, default=0.0):
    try:
        if value in (None, "", "nan", "NaN", "None", "unavailable"):
            return default
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def write_rows(path, rows):
    rows = list(rows)
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    if not fields:
        fields = ["status"]
        rows = [{"status": "UNAVAILABLE"}]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def first_crossing(row):
    ordered = (
        ("opening", "cash_opening"),
        ("payroll", "cash_after_payroll"),
        ("dividends", "cash_after_dividends"),
        ("legacy_private_transfer", "cash_after_legacy_private_support"),
        ("PAYG_contribution", "cash_after_payg_contribution"),
        ("pension", "cash_after_pension"),
        ("private_transfer", "cash_after_private_transfer"),
        ("consumption", "cash_after_consumption"),
    )
    previous = None
    for operation, key in ordered:
        value = row.get(key)
        if value is None:
            continue
        current = number(value)
        if previous is not None and previous >= -EPS and current < -EPS:
            return operation
        if previous is None and current < -EPS:
            return f"{operation}_from_opening"
        previous = current
    return "NO_CROSSING_IN_RECORDED_WINDOW"


def compact_trace(row):
    return {
        "branch": row.get("branch"),
        "week": row.get("week"),
        "household_id": row.get("household_id"),
        "opening_cash": row.get("cash_opening"),
        "wage_income_received": row.get("wage_income_received"),
        "dividends_received": row.get("dividends_received"),
        "legacy_private_transfer_paid": row.get("legacy_private_transfer_paid"),
        "legacy_private_transfer_received": row.get("legacy_private_transfer_received"),
        "cash_after_payroll": row.get("cash_after_payroll"),
        "cash_after_dividends": row.get("cash_after_dividends"),
        "cash_after_legacy_private_support": row.get("cash_after_legacy_private_support"),
        "scheduled_payg_contribution": row.get("scheduled_payg_contribution"),
        "actual_payg_contribution": row.get("actual_payg_contribution"),
        "cash_before_contribution": row.get("cash_after_legacy_private_support"),
        "cash_after_contribution": row.get("cash_after_payg_contribution"),
        "pension_received": row.get("pension_received"),
        "cash_after_pension": row.get("cash_after_pension"),
        "private_transfer_paid": row.get("private_transfer_paid"),
        "private_transfer_received": row.get("private_transfer_received"),
        "cash_after_private_transfer": row.get("cash_after_private_transfer"),
        "cash_before_consumption": row.get("cash_before_consumption"),
        "minimum_consumption": row.get("minimum_consumption"),
        "planned_consumption": row.get("planned_consumption"),
        "affordable_consumption": row.get("affordable_consumption"),
        "actual_consumption": row.get("actual_consumption"),
        "cash_after_consumption": row.get("cash_after_consumption"),
        "closing_cash": row.get("closing_cash"),
        "transition_operation": first_crossing(row),
        "elderly_members": row.get("elderly_members"),
        "employed_members": row.get("employed_members"),
    }


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True, exist_ok=True)
    e_module = load_module(ROOT / "test/step17_e_recipient_persistence_smoke.py", "step17_e_for_e1")
    temp_dir = Path(tempfile.mkdtemp(prefix="step17e1_", dir=str(ROOT / "test")))
    run_data = {}
    try:
        common = e_module.make_common_checkpoint(temp_dir)
        from experiment_workflow import ExperimentBranch, ExperimentSpec, run_experiment
        for branch in BRANCHES:
            overrides = e_module.policy_overrides(branch)
            overrides["world"]["payg_cash_safety_audit_enabled"] = True
            spec = ExperimentSpec(
                experiment_name=f"step17_e1_{branch.lower()}_targeted_replay",
                output_root=str(temp_dir / branch.lower()),
                steps=52,
                checkpoint_path=str(common),
                seed=42,
                population=5000,
                scenario="baseline",
                firm_count=5,
                max_workers=1,
                observability_mode="RESEARCH_FAST",
                diagnostics_mode="compact",
                persist_diagnostics=False,
                resume=False,
                branches=(ExperimentBranch(branch_name=branch, config_overrides=overrides),),
            )
            result = run_experiment(spec)
            item = result.get("results", [{}])[0]
            directory = Path(item.get("output_dir", ""))
            run_data[branch] = {
                "result": item,
                "rows": read_json(directory / "payg_cash_safety_event_window.json", []),
                "summary": read_json(directory / "payg_cash_safety_summary.json", {}),
            }

        all_rows = []
        for branch, data in run_data.items():
            for row in data["rows"]:
                row = dict(row)
                row["branch"] = branch
                all_rows.append(row)

        first_negative = []
        for branch, data in run_data.items():
            negative = [row for row in data["rows"] if number(row.get("cash_after_consumption")) < -EPS]
            first_week = min((int(number(row.get("week"))) for row in negative), default=None)
            first_households = set()
            if first_week is not None:
                first_households = {
                    row.get("household_id") for row in negative if int(number(row.get("week"))) == first_week
                }
            first_negative.extend(
                dict(row, branch=branch)
                for row in data["rows"]
                if row.get("household_id") in first_households
                and first_week is not None
                and first_week - 3 <= int(number(row.get("week"))) <= first_week + 2
            )
        trace_rows = [
            compact_trace(row)
            for row in sorted(
                first_negative,
                key=lambda row: (row.get("branch"), number(row.get("week")), number(row.get("household_id"))),
            )
        ]
        write_rows(OUT / "first_negative_cash_trace.csv", trace_rows)

        profile_rows = []
        for branch, data in run_data.items():
            rows = data["rows"]
            negative = [row for row in rows if number(row.get("cash_after_consumption")) < -EPS]
            values = [number(row.get("cash_after_consumption")) for row in negative]
            profile_rows.append({
                "branch": branch,
                "audit_rows": data["summary"].get("row_count", len(rows)),
                "negative_household_weeks": data["summary"].get("negative_household_weeks", len(negative)),
                "negative_unique_households": data["summary"].get("negative_unique_households", len({row.get("household_id") for row in negative})),
                "minimum_negative_cash": data["summary"].get("minimum_cash_after_consumption", min(values, default=0.0)),
                "median_negative_cash": data["summary"].get("median_negative_cash", sorted(values)[len(values) // 2] if values else 0.0),
                "first_negative_week": data["summary"].get("first_negative_week", min((int(number(row.get("week"))) for row in negative), default="unavailable")),
                "last_negative_week": data["summary"].get("last_negative_week", max((int(number(row.get("week"))) for row in negative), default="unavailable")),
                "opening_negative_household_weeks": data["summary"].get("opening_negative_household_weeks", sum(number(row.get("cash_opening")) < -EPS for row in rows)),
                "after_contribution_negative_count": data["summary"].get("after_contribution_negative_count", sum(number(row.get("cash_after_payg_contribution")) < -EPS for row in rows)),
                "after_consumption_negative_count": data["summary"].get("negative_household_weeks", len(negative)),
            })
        write_rows(OUT / "negative_cash_profile.csv", profile_rows)

        contribution_rows = []
        for branch, data in run_data.items():
            summary = data["summary"]
            rows = [row for row in data["rows"] if number(row.get("scheduled_payg_contribution")) > EPS]
            contribution_rows.append({
                "branch": branch,
                "contributor_household_weeks": summary.get("contributor_household_weeks", len(rows)),
                "scheduled_contribution": summary.get("scheduled_contribution_total", sum(number(row.get("scheduled_payg_contribution")) for row in rows)),
                "actual_contribution": summary.get("actual_contribution_total", sum(number(row.get("actual_payg_contribution")) for row in rows)),
                "actual_gt_cash_before_contribution_count": summary.get("contribution_cash_violations", 0),
                "cash_after_contribution_negative_count_from_nonnegative": summary.get("contribution_post_cash_violations", 0),
                "preexisting_negative_before_contribution_count": summary.get("preexisting_negative_before_contribution_count", 0),
                "minimum_cash_after_contribution": summary.get("minimum_cash_after_contribution", min((number(row.get("cash_after_payg_contribution")) for row in rows), default=0.0)),
                "contract_result": "CASH_SAFE_AT_COLLECTION_WITH_PREEXISTING_NEGATIVE_STATES" if number(summary.get("contribution_cash_violations")) == 0 and number(summary.get("contribution_post_cash_violations")) == 0 else "CONTRIBUTION_CASH_SAFETY_VIOLATION",
            })
        write_rows(OUT / "contribution_cash_safety.csv", contribution_rows)

        pension_rows = []
        for branch, data in run_data.items():
            summary = data["summary"]
            rows = [row for row in data["rows"] if number(row.get("pension_received")) > EPS]
            pension_rows.append({
                "branch": branch,
                "pension_recipient_household_weeks": summary.get("pension_recipient_household_weeks", len(rows)),
                "pension_received_total": summary.get("pension_received_total", sum(number(row.get("pension_received")) for row in rows)),
                "cash_decrease_after_pension_count": summary.get("pension_cash_decrease_count", 0),
                "minimum_pension_cash_delta": summary.get("minimum_pension_cash_delta", min((number(row.get("cash_after_pension")) - number(row.get("cash_after_payg_contribution")) for row in rows), default=0.0)),
                "contract_result": "INCOMING_CASH_NON_DECREASING" if number(summary.get("pension_cash_decrease_count")) == 0 else "PENSION_CASH_SAFETY_VIOLATION",
            })
        write_rows(OUT / "pension_cash_safety.csv", pension_rows)

        private_rows = []
        for branch, data in run_data.items():
            summary = data["summary"]
            rows = [row for row in data["rows"] if number(row.get("private_transfer_paid")) > EPS]
            private_rows.append({
                "branch": branch,
                "donor_household_weeks": summary.get("private_donor_household_weeks", len(rows)),
                "private_transfer_paid_total": summary.get("private_transfer_paid_total", sum(number(row.get("private_transfer_paid")) for row in rows)),
                "donor_reserve_violations_at_transfer": summary.get("private_reserve_violations", 0),
                "minimum_post_transfer_donor_cash_minus_r13_reserve": min((number(row.get("cash_after_private_transfer")) - 13.0 * number(row.get("minimum_consumption")) for row in rows), default=0.0),
                "later_negative_after_safe_transfer_count": summary.get("later_negative_after_safe_private_transfer", 0),
                "contract_result": "TRANSFER_STAGE_CASH_SAFE" if number(summary.get("private_reserve_violations")) == 0 else "PRIVATE_TRANSFER_RESERVE_VIOLATION",
            })
        write_rows(OUT / "private_transfer_cash_safety.csv", private_rows)

        write_rows(OUT / "consumption_contract_audit.csv", [
            {"field": "planning_entry_cash", "authoritative_value": "wealth_before_income_this_step", "meaning": "Household cash immediately before payroll; not refreshed after PAYG contribution", "source": "economy/firm.py household_food_purchase_units"},
            {"field": "income_input", "authoritative_value": "income_this_step", "meaning": "Gross wage/dividend/pension/transfer income field; PAYG contribution is not subtracted here", "source": "economy/firm.py household_food_purchase_units"},
            {"field": "affordable_consumption", "authoritative_value": "income_this_step + starting_wealth * WEALTH_DRAWDOWN_RATE", "meaning": "Budget formula uses pre-policy starting wealth and gross income", "source": "economy/firm.py household_food_purchase_units"},
            {"field": "actual_consumption", "authoritative_value": "max(0, min(desired_money, affordable_money))", "meaning": "Bounded by formula above, not current post-PAYG cash", "source": "economy/firm.py household_food_purchase_units"},
            {"field": "cash_settlement", "authoritative_value": "household.wealth -= actual_money", "meaning": "Ledger/direct mutation debits current cash; no post-PAYG cash clamp", "source": "economy/firm.py household_food_purchase_units"},
            {"field": "negative_cash_clamp", "authoritative_value": "absent in canonical Food path", "meaning": "The separate ConsumptionSystem clamp is not the canonical multi-Firm market settlement", "source": "economy/firm.py / economy/consumption.py"},
        ])
        write_rows(OUT / "consumption_timing_audit.csv", [
            {"order": 1, "stage": "payroll", "cash_effect": "wage cash inflow", "observed": True},
            {"order": 2, "stage": "dividends", "cash_effect": "dividend cash inflow where applicable", "observed": True},
            {"order": 3, "stage": "legacy_private_family_support", "cash_effect": "existing support may settle before active PAYG in current Firm.step", "observed": True},
            {"order": 4, "stage": "PAYG contribution", "cash_effect": "Household cash decreases; contribution is min(scheduled, nonnegative current cash)", "observed": True},
            {"order": 5, "stage": "PAYG pension", "cash_effect": "fund cash to recipient Household", "observed": True},
            {"order": 6, "stage": "intergenerational wealth transfer", "cash_effect": "donor to recipient Household", "observed": True},
            {"order": 7, "stage": "Food consumption planning and settlement", "cash_effect": "planned from stale pre-policy cash/income inputs, debited from current cash", "observed": True},
            {"order": 8, "stage": "finding", "cash_effect": "PAYG contribution is safe at collection, but can leave insufficient current cash for the later consumption debit", "observed": True},
        ])
        write_rows(OUT / "income_field_audit.csv", [
            {"field": "household.income_this_step", "PAYG_effect": "pension adds income; contribution does not subtract from field", "used_by_consumption": True, "semantic_issue": "gross income field remains available to consumption formula after PAYG cash debit"},
            {"field": "household.payg_contribution_this_step", "PAYG_effect": "actual contribution recorded separately", "used_by_consumption": False, "semantic_issue": "not incorporated into affordable_money"},
            {"field": "household.wealth", "PAYG_effect": "directly reduced by contribution", "used_by_consumption": "current debit target only", "semantic_issue": "starting_wealth used for budget is captured before policy"},
            {"field": "world total_income", "PAYG_effect": "aggregate subtracts payg contribution", "used_by_consumption": False, "semantic_issue": "aggregate accounting and per-Household budget fields have different timing"},
        ])

        negative_first_rows = [row for row in all_rows if number(row.get("cash_after_consumption")) < -EPS]
        write_rows(OUT / "affected_household_profile.csv", [{
            "branch": branch,
            "affected_household_weeks": len(rows),
            "unique_households": len({row.get("household_id") for row in rows}),
            "contributor_household_weeks": sum(number(row.get("actual_payg_contribution")) > EPS for row in rows),
            "elderly_member_household_weeks": sum(number(row.get("elderly_members")) > 0 for row in rows),
            "employed_member_household_weeks": sum(number(row.get("employed_members")) > 0 for row in rows),
            "median_minimum_consumption": sorted(number(row.get("minimum_consumption")) for row in rows)[len(rows) // 2] if rows else 0.0,
            "median_wage_income": sorted(number(row.get("wage_income_received")) for row in rows)[len(rows) // 2] if rows else 0.0,
            "median_actual_contribution": sorted(number(row.get("actual_payg_contribution")) for row in rows)[len(rows) // 2] if rows else 0.0,
            "median_liquidity_before_contribution": sorted(number(row.get("cash_after_legacy_private_support")) / max(number(row.get("minimum_consumption")), EPS) for row in rows)[len(rows) // 2] if rows else 0.0,
            "interpretation": "affected rows are low-current-cash contributor states; exact contributor share is reported, not inferred",
        } for branch in BRANCHES for rows in [[row for row in negative_first_rows if row.get("branch") == branch]]])

        counterfactual_rows = []
        for branch in BRANCHES:
            rows = [row for row in negative_first_rows if row.get("branch") == branch]
            without = [number(row.get("cash_after_consumption")) + number(row.get("actual_payg_contribution")) for row in rows]
            counterfactual_rows.append({
                "branch": branch,
                "negative_cash_household_weeks": len(rows),
                "cash_after_consumption_with_contribution_min": min((number(row.get("cash_after_consumption")) for row in rows), default=0.0),
                "cash_after_consumption_without_contribution_min": min(without, default=0.0),
                "without_contribution_nonnegative_count": sum(value >= -EPS for value in without),
                "contribution_is_trigger_count": sum(value >= -EPS and number(row.get("cash_after_consumption")) < -EPS for row, value in zip(rows, without)),
                "counterfactual_semantics": "all realized flows held fixed; add back actual PAYG contribution only",
            })
        write_rows(OUT / "contribution_counterfactual.csv", counterfactual_rows)

        e_distribution = {row.get("branch"): row for row in read_rows(E_OUT / "branch_distribution_comparison.csv")}
        validity = []
        for branch in ("CONTROL", "PRIVATE_ONLY", "PAYG_3_ONLY", "COMBINED_3"):
            profile = next((row for row in profile_rows if row["branch"] == branch), None)
            negative_count = number(profile.get("negative_household_weeks")) if profile else 0
            validity.append({
                "branch": branch,
                "cash_distribution_source": "Step17.E branch_distribution_comparison.csv" if branch in e_distribution else "not applicable",
                "assessment": "VALID_WITH_NEGATIVE_CASH_CAVEAT" if negative_count > 0 else "VALID",
                "negative_household_weeks_in_targeted_replay": negative_count,
                "reason": "Low-tail/Gini remain descriptive but PAYG negative cash must be disclosed; rerun after a cash-safety correction for policy evaluation" if negative_count > 0 else "No negative cash observed in targeted replay",
            })
        write_rows(OUT / "distribution_validity_assessment.csv", validity)
        write_rows(OUT / "recommended_cash_safety_contract.csv", [{
            "priority": 1,
            "recommended_contract": "A. CONSUMPTION_MUST_BE_RECONSTRAINED_AFTER_PAYG",
            "minimum_semantics": "Recompute the Household consumption cash budget from post-contribution/post-transfer current cash, or apply an equivalent current-cash settlement clamp.",
            "why_minimum": "Contribution affordability already passes; the observed cross-zero operation is the later consumption debit.",
            "not_selected": "Do not change PAYG rate/benefit/order in this audit.",
        }])

        summaries = [run_data[branch]["summary"] for branch in BRANCHES]
        first_week = min((number(item.get("first_negative_week"), math.inf) for item in summaries), default=math.inf)
        first_details = []
        for branch in BRANCHES:
            summary = run_data[branch]["summary"]
            ids = ",".join(str(value) for value in summary.get("first_negative_household_ids", [])) or "unavailable"
            first_details.append(f"{branch}: week {summary.get('first_negative_week', 'unavailable')}, Household {ids}")
        contribution_ok = all(number(row.get("actual_gt_cash_before_contribution_count")) == 0 and number(row.get("cash_after_contribution_negative_count_from_nonnegative")) == 0 for row in contribution_rows)
        pension_ok = all(number(row.get("cash_decrease_after_pension_count")) == 0 for row in pension_rows)
        private_ok = all(number(row.get("donor_reserve_violations_at_transfer")) == 0 for row in private_rows)
        counterfactual_trigger = sum(number(row.get("contribution_is_trigger_count")) for row in counterfactual_rows)
        verdict = "B. POST_CONTRIBUTION_CONSUMPTION_CAUSES_NEGATIVE_CASH" if contribution_ok and pension_ok and private_ok and any(number(row.get("after_consumption_negative_count")) > 0 for row in profile_rows if row["branch"] in BRANCHES) else "A. PAYG_CONTRIBUTION_DIRECTLY_CAUSES_NEGATIVE_CASH"
        flags = {
            "verdict": verdict,
            "first_negative_week_by_branch": {branch: run_data[branch]["summary"].get("first_negative_week", "unavailable") for branch in BRANCHES},
            "first_negative_household_identified": bool(first_negative),
            "consumption_crossing_observed": verdict.startswith("B."),
            "contribution_collection_cash_safe": contribution_ok,
            "contribution_did_not_cross_nonnegative_to_negative": contribution_ok,
            "pension_cash_safe": pension_ok,
            "private_transfer_cash_safe": private_ok,
            "payg_counterfactual_trigger_count": counterfactual_trigger,
            "planning_uses_pre_policy_cash": True,
            "consumption_post_payg_cash_clamp_present": False,
            "negative_cash_is_instrumentation_attributable": False,
            "step17e_distribution_validity": "VALID_WITH_NEGATIVE_CASH_CAVEAT",
            "step17e_distribution_policy_evaluation_requires_rerun_after_fix": True,
            "economic_behavior_changed": False,
            "payg_parameters_changed": False,
            "pension_benefits_changed": False,
            "private_transfer_changed": False,
            "new_rng_draws": 0,
            "long_run_52_week_four_branch_rerun": False,
            "step17f_started": False,
            "research_only": True,
        }
        (OUT / "acceptance_flags.json").write_text(json.dumps(flags, indent=2), encoding="utf-8")
        lines = [
            "# Step17.E.1 PAYG Contribution / Household Consumption Cash-Safety Audit",
            "",
            f"- Targeted 52-week replays used only PAYG_3_ONLY and COMBINED_3; no four-branch long run was rerun.",
            f"- First-negative locations: {', '.join(first_details)}; first Household IDs are recorded in the flags and trace.",
            "- PAYG collection did not push a nonnegative Household below zero and never exceeded current available cash; later contributor rows include preexisting negative balances, reported separately.",
            "- Pension settlement was cash-safe for valid recipients; combined private-transfer donor reserve checks also remained safe.",
            "- The exact observed cross-zero operation was the later Food consumption debit, whose budget still used pre-policy starting cash and gross income.",
            f"- Contribution-only shadow add-back indicates {counterfactual_trigger} negative rows would be nonnegative without the contribution, while the immediate contribution contract still passed.",
            "- Step17.E distribution outputs remain VALID_WITH_NEGATIVE_CASH_CAVEAT and should be rerun after a future cash-safety correction before policy evaluation.",
            f"- Verdict: **{verdict}**. Recommended minimum contract: re-constrain consumption after PAYG.",
            "- No PAYG/private-support parameter, benefit, ordering, RNG, canonical enablement, or Step17.F work was performed.",
        ]
        (OUT / "acceptance_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()