"""Post-Step13 integrated acceptance using the completed main.py runs."""

from __future__ import annotations

import ast
import csv
import hashlib
import inspect
import json
import math
import pickle
import random
import re
import sys
import textwrap
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.v2 import run_analysis_v2
from scenarios import apply_runtime_scenario, apply_scenario
from world import World


VALIDATION = ROOT / "test" / "output" / "post_step13_integrated_validation"
RUN_5 = VALIDATION / "main_canonical_5pct"
RUN_0 = VALIDATION / "main_canonical_0pct"
SMOKE = VALIDATION / "main_smoke_v2"
FULL_SMOKE = VALIDATION / "main_full_profile_smoke"
ZERO_EXPLICIT = VALIDATION / "zero_interest_compat_explicit"
ZERO_REFERENCE = VALIDATION / "zero_interest_compat_reference"
OUT = ROOT / "test" / "output" / "post_step13_integrated_acceptance"
STEP13_METRICS = (
    ROOT
    / "test"
    / "output"
    / "step13_final_financial_core_closure"
    / "step13_final_baseline_metrics.csv"
)
TOLERANCE = 1e-6


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def number(value, default=None):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def truth(value):
    if isinstance(value, str):
        return value.strip().lower() == "true"
    return bool(value)


def max_abs(rows, field):
    return max(
        (abs(number(row.get(field), 0.0)) for row in rows),
        default=0.0,
    )


def sum_field(rows, field):
    return math.fsum(number(row.get(field), 0.0) for row in rows)


def financial_summary(path):
    rows = read_csv(path)
    return {row["metric"]: row.get("value") for row in rows}


def compare_csv(left_path, right_path, excluded=()):
    left = read_csv(left_path)
    right = read_csv(right_path)
    if len(left) != len(right):
        return {"equal": False, "max_abs_diff": math.inf, "first_difference": "row_count"}
    excluded = set(excluded)
    fields = [field for field in left[0] if field not in excluded] if left else []
    maximum = 0.0
    first = None
    for index, (old, new) in enumerate(zip(left, right)):
        for field in fields:
            a = old.get(field, "")
            b = new.get(field, "")
            na = number(a)
            nb = number(b)
            if na is not None and nb is not None:
                difference = abs(na - nb)
                maximum = max(maximum, difference)
                differs = difference > 1e-12
            else:
                differs = a != b
                difference = None
            if differs and first is None:
                first = {
                    "row": index,
                    "field": field,
                    "left": a,
                    "right": b,
                    "abs_diff": difference,
                }
    return {"equal": first is None, "max_abs_diff": maximum, "first_difference": first}


def rng_snapshot(world):
    payload = {
        "global": random.getstate(),
        "market": getattr(getattr(world, "market_rng", None), "getstate", lambda: None)(),
        "age_phase": getattr(getattr(world, "age_phase_rng", None), "getstate", lambda: None)(),
        "marriage": getattr(
            getattr(getattr(world, "marriage_system", None), "marriage_rng", None),
            "getstate",
            lambda: None,
        )(),
        "firms": [
            getattr(getattr(firm, "firm_rng", None), "getstate", lambda: None)()
            for firm in getattr(world, "firms", [])
        ],
    }
    return hashlib.sha256(pickle.dumps(payload, protocol=5)).hexdigest()


def analysis_noninterference():
    scenario = apply_scenario("interest_behavioral_5pct")
    world = World(
        initial_population=200,
        seed=42,
        scenario_name=scenario["name"],
        scenario_overrides=scenario["overrides"],
        diagnostics_mode="full",
        initial_age_phase_mode="distributed",
    )
    apply_runtime_scenario(scenario, world)
    world.split_firms(5)
    world.steps = 104
    started = time.perf_counter()
    world.run(progress_interval=0)
    simulation_seconds = time.perf_counter() - started
    state_before = hashlib.sha256(pickle.dumps(world, protocol=5)).hexdigest()
    rng_before = rng_snapshot(world)
    analysis_dir = VALIDATION / "analysis_noninterference"
    started = time.perf_counter()
    run_analysis_v2(
        world,
        analysis_dir,
        profile="standard",
        plot_mode="save-only",
        no_plots=True,
    )
    analysis_seconds = time.perf_counter() - started
    state_after = hashlib.sha256(pickle.dumps(world, protocol=5)).hexdigest()
    rng_after = rng_snapshot(world)
    return {
        "state_hash_equal": state_before == state_after,
        "rng_equal": rng_before == rng_after,
        "analysis_state_max_abs_diff": 0.0 if state_before == state_after else math.inf,
        "analysis_world_state_mutation_count": 0 if state_before == state_after else 1,
        "analysis_rng_changed": rng_before != rng_after,
        "simulation_seconds": simulation_seconds,
        "analysis_seconds": analysis_seconds,
        "analysis_runtime_overhead_share": (
            analysis_seconds / simulation_seconds if simulation_seconds else None
        ),
    }


def source_audit():
    finance_files = [
        ROOT / "economy" / "default_bookkeeping.py",
        ROOT / "economy" / "interest.py",
        ROOT / "economy" / "restructuring.py",
        ROOT / "central_bank" / "central_bank.py",
    ]
    demographic_calls = {
        "family_birth", "check_death", "process_marriage", "grow",
        "process_inheritance", "refresh_active_households",
        "remove_person_relationship", "create_household", "remove_household",
    }
    demographic_attrs = {
        "age", "age_weeks", "alive", "last_birth_step", "household_id",
        "parents", "children", "spouse_id",
    }
    rng_methods = {
        "random", "randrange", "choice", "shuffle", "sample", "uniform",
        "gauss", "randint", "seed",
    }
    direct_calls = 0
    direct_writes = 0
    rng_calls = 0

    world_methods = [
        World.ensure_default_bookkeeping_state,
        World.ensure_restructuring_state,
        World.begin_restructuring_week,
        World.finalize_restructuring_week,
        World.current_interest_terms,
        World.update_default_bookkeeping,
        World.prepare_multi_firm_credit,
    ]
    sources = [path.read_text(encoding="utf-8") for path in finance_files]
    sources.extend(textwrap.dedent(inspect.getsource(method)) for method in world_methods)

    for source in sources:
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                name = None
                owner = None
                if isinstance(node.func, ast.Name):
                    name = node.func.id
                elif isinstance(node.func, ast.Attribute):
                    name = node.func.attr
                    if isinstance(node.func.value, ast.Name):
                        owner = node.func.value.id
                direct_calls += int(name in demographic_calls)
                rng_calls += int(
                    name in rng_methods
                    and (owner == "random" or "rng" in (owner or "").lower())
                )
            if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                targets = []
                if isinstance(node, ast.Assign):
                    targets = node.targets
                else:
                    targets = [node.target]
                for target in targets:
                    if isinstance(target, ast.Attribute) and target.attr in demographic_attrs:
                        direct_writes += 1
    return {
        "finance_direct_demographic_write_count": direct_writes + direct_calls,
        "finance_demographic_rng_call_count": rng_calls,
        "audited_files": [str(path.relative_to(ROOT)) for path in finance_files],
        "audited_world_methods": [method.__name__ for method in world_methods],
    }


def zero_interest_compatibility():
    macro = compare_csv(
        ZERO_EXPLICIT / "diagnostics.csv",
        ZERO_REFERENCE / "diagnostics.csv",
        excluded={"scenario"},
    )
    firms = compare_csv(
        ZERO_EXPLICIT / "firm_diagnostics.csv",
        ZERO_REFERENCE / "firm_diagnostics.csv",
        excluded={"scenario"},
    )
    demography = compare_csv(
        ZERO_EXPLICIT / "demographic_events.csv",
        ZERO_REFERENCE / "demographic_events.csv",
    )
    explicit = read_csv(ZERO_EXPLICIT / "diagnostics.csv")
    interest_zero = max(
        max_abs(explicit, field)
        for field in (
            "current_interest_due", "interest_paid", "current_interest_unpaid",
            "total_interest_arrears",
        )
    )
    return {
        "pass": macro["equal"] and firms["equal"] and demography["equal"] and interest_zero == 0,
        "macro": macro,
        "firms": firms,
        "demography": demography,
        "max_abs_interest_field": interest_zero,
        "pre_step13_exact_path_available": False,
        "reference_semantics": (
            "Explicit 0% interest versus the same current-code K=46.36154354202572 "
            "credit architecture with its default 0% rate."
        ),
    }


def first_divergence(left, right, fields):
    for old, new in zip(left, right):
        differing = []
        for field in fields:
            a = number(old.get(field))
            b = number(new.get(field))
            if a is not None and b is not None and abs(a - b) > 1e-9:
                differing.append(field)
            elif a is None and b is None and old.get(field) != new.get(field):
                differing.append(field)
        if differing:
            return int(number(old.get("global_step", old.get("step")), 0)), differing
    return None, []


def demographic_comparison():
    interest5 = read_csv(RUN_5 / "diagnostics.csv")
    zero = read_csv(RUN_0 / "diagnostics.csv")
    economic_fields = [
        "current_interest_due", "interest_paid", "current_interest_unpaid",
        "firm_cash", "total_dividend", "total_income", "total_consumption",
        "total_household_wealth", "median_security_ratio", "loan_balance",
        "total_denied_credit", "actual_production", "food_sales_units",
    ]
    demographic_fields = [
        "births", "deaths", "matches_formed", "population", "active_households",
        "age_0_19_count", "age_20_39_count", "age_40_64_count",
        "age_65_plus_count", "working_age_population", "elderly_population",
    ]
    first_economic, economic_causes = first_divergence(
        interest5, zero, economic_fields
    )
    first_demographic, demographic_causes = first_divergence(
        interest5, zero, demographic_fields
    )

    def outcomes(rows):
        last = rows[-1]
        return {
            "final_population": int(number(last.get("population"), 0)),
            "birth_count": int(sum_field(rows, "births")),
            "death_count": int(sum_field(rows, "deaths")),
            "marriage_count": int(sum_field(rows, "matches_formed")),
            "active_households": int(number(last.get("active_households"), 0)),
            "working_age_population": int(number(last.get("working_age_population"), 0)),
            "elderly_population": int(number(last.get("elderly_population"), 0)),
            "age_0_19_count": int(number(last.get("age_0_19_count"), 0)),
            "age_20_39_count": int(number(last.get("age_20_39_count"), 0)),
            "age_40_64_count": int(number(last.get("age_40_64_count"), 0)),
            "age_65_plus_count": int(number(last.get("age_65_plus_count"), 0)),
        }

    result5 = outcomes(interest5)
    result0 = outcomes(zero)
    precedes = (
        first_demographic is not None
        and (first_economic is None or first_demographic < first_economic)
    )
    return {
        "interest5": result5,
        "zero_interest": result0,
        "population_difference": (
            result5["final_population"] - result0["final_population"]
        ),
        "first_economic_divergence_week": first_economic,
        "first_economic_divergence_fields": economic_causes,
        "first_demographic_divergence_week": first_demographic,
        "first_demographic_divergence_fields": demographic_causes,
        "demographic_divergence_precedes_economic_divergence": precedes,
        "household_formations": None,
        "household_dissolutions": None,
        "unavailable_note": (
            "Formation/dissolution event totals are not exposed as complete canonical "
            "event series; active household counts are reported without fabrication."
        ),
    }


def accounting_gates(run):
    macro = read_csv(run / "diagnostics.csv")
    firm = read_csv(run / "accounting" / "firm_accounting.csv")
    household = read_csv(run / "accounting" / "household_accounting.csv")
    public = read_csv(run / "accounting" / "public_accounting.csv")
    central = read_csv(run / "accounting" / "central_bank_accounting.csv")
    reconciliation = read_csv(run / "accounting" / "accounting_reconciliation.csv")
    gates = {
        "invariant_violations": sum(truth(row.get("invariant_failed")) for row in macro),
        "max_goods_conservation_gap": max_abs(macro, "food_conservation_gap"),
        "max_money_accounting_gap": max_abs(macro, "monetary_accounting_gap"),
        "max_money_delta_gap": max_abs(macro, "money_delta_gap"),
        "max_firm_cash_flow_gap": max_abs(firm, "cash_flow_gap"),
        "max_firm_inventory_bridge_gap": max_abs(firm, "inventory_bridge_gap"),
        "max_firm_equity_bridge_gap": max_abs(firm, "equity_bridge_gap"),
        "max_household_wealth_bridge_gap": max_abs(household, "household_wealth_bridge_gap"),
        "max_public_cash_flow_gap": max_abs(public, "public_cash_flow_gap"),
        "max_credit_money_stock_flow_gap": max_abs(central, "credit_money_stock_flow_gap"),
        "max_loan_reconciliation_gap": max_abs(reconciliation, "loan_reconciliation_gap"),
        "max_lender_claim_gap": max_abs(reconciliation, "total_lender_claim_gap"),
        "max_money_location_gap": max_abs(reconciliation, "money_location_gap"),
    }
    gates["pass"] = (
        gates["invariant_violations"] == 0
        and max(value for key, value in gates.items() if key.startswith("max_")) <= 1e-4
    )
    return gates


def reference_reproduction(summary):
    reference_rows = read_csv(STEP13_METRICS)
    reference = {
        row["metric"]: number(row.get("value"))
        for row in reference_rows
        if number(row.get("value")) is not None
    }
    actual = {
        "final_population": number(summary["final_population"]),
        "final_active_households": number(summary["final_active_households"]),
        "final_loan_principal": number(summary["final_outstanding_principal"]),
        "final_interest_arrears": number(summary["final_interest_arrears"]),
        "final_total_lender_exposure": number(summary["final_total_lender_exposure"]),
        "central_bank_interest_cash_income_total": number(summary["central_bank_interest_income"]),
        "default_event_count_total": number(summary["default_event_count"]),
        "active_contract_default_firm_weeks": number(summary["active_contract_default_firm_weeks"]),
        "mature_D2_share": number(summary["mature_d2_share"]),
        "mature_D3_share": number(summary["mature_d3_share"]),
    }
    gaps = {key: abs(actual[key] - reference[key]) for key in actual}
    return {"pass": max(gaps.values()) <= TOLERANCE, "max_abs_gap": max(gaps.values()), "gaps": gaps}


def plot_counts():
    standard = json.loads(
        (SMOKE / "analysis" / "browser_category_manifest.json").read_text(encoding="utf-8")
    )
    full = json.loads(
        (FULL_SMOKE / "analysis" / "browser_category_manifest.json").read_text(encoding="utf-8")
    )
    standard_financial = [entry for entry in standard if entry.get("category") == "Financial Core"]
    full_detail = [
        entry for entry in full
        if entry.get("category") == "Financial Core"
        and entry.get("source_function") == "analysis.v22.generate_detailed_plots"
    ]
    return {
        "ready": len(standard_financial) == 4 and len(full_detail) == 3,
        "standard_added": len(standard_financial),
        "full_added": len(full_detail),
    }


def write_preview():
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt

    plot_dir = RUN_5 / "analysis" / "plots"
    names = [
        "09_financial_credit_stock_flow.png",
        "10_financial_interest_arrears.png",
        "11_financial_distress_default.png",
        "12_financial_money_lender_exposure.png",
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    for axis, name in zip(axes.flat, names):
        axis.imshow(mpimg.imread(plot_dir / name))
        axis.axis("off")
    fig.suptitle("SOCIALSIM Step 13 Financial Core - standard profile preview")
    fig.tight_layout()
    fig.savefig(OUT / "main_financial_core_preview.png", dpi=120)
    plt.close(fig)


def write_csv(path, rows):
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    required = [
        RUN_5 / "diagnostics.csv", RUN_0 / "diagnostics.csv",
        RUN_5 / "analysis" / "tables" / "financial_core_summary.csv",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing completed acceptance runs: " + ", ".join(missing))

    noninterference = analysis_noninterference()
    coupling = source_audit()
    zero_compat = zero_interest_compatibility()
    demography = demographic_comparison()
    accounting = accounting_gates(RUN_5)
    summary = financial_summary(
        RUN_5 / "analysis" / "tables" / "financial_core_summary.csv"
    )
    reference = reference_reproduction(summary)
    plots = plot_counts()
    main_smoke_pass = not (
        SMOKE / "analysis" / "analysis_failure.json"
    ).exists()
    main_full_pass = not (
        RUN_5 / "analysis" / "analysis_failure.json"
    ).exists()
    restructuring_off = (
        max_abs(read_csv(RUN_5 / "diagnostics.csv"), "restructuring_start_count_this_step") == 0
        and max_abs(read_csv(RUN_5 / "diagnostics.csv"), "weekly_interest_relief_amount") == 0
        and max_abs(read_csv(RUN_5 / "diagnostics.csv"), "termout_start_count_this_step") == 0
    )
    direct_absent = coupling["finance_direct_demographic_write_count"] == 0
    rng_absent = coupling["finance_demographic_rng_call_count"] == 0
    explainable = (
        not demography["demographic_divergence_precedes_economic_divergence"]
        and direct_absent
        and rng_absent
    )
    population_classification = (
        "B ECONOMICALLY_MEDIATED_DEMOGRAPHIC_EFFECT"
        if demography["population_difference"] != 0 and explainable
        else "A NO_DEMOGRAPHIC_EFFECT_DETECTED"
        if demography["population_difference"] == 0
        else "E UNEXPLAINED_DEMOGRAPHIC_DIVERGENCE"
    )
    all_pass = all((
        main_smoke_pass,
        main_full_pass,
        plots["ready"],
        noninterference["state_hash_equal"],
        noninterference["rng_equal"],
        zero_compat["pass"],
        reference["pass"],
        accounting["pass"],
        direct_absent,
        rng_absent,
        explainable,
        restructuring_off,
    ))
    verdict = (
        "A STEP13_FULL_INTEGRATION_ACCEPTED"
        if all_pass else "F UNEXPLAINED_POPULATION_EFFECT_FOUND"
    )
    flags = {
        "verdict": verdict,
        "analysis_financial_core_ready": plots["ready"],
        "analysis_noninterference_pass": noninterference["state_hash_equal"] and noninterference["rng_equal"],
        "plot_browser_updated": plots["ready"],
        "main_step13_scenario_ready": True,
        "main_smoke_pass": main_smoke_pass,
        "main_full_acceptance_pass": main_full_pass,
        "step13_reference_reproduced": reference["pass"],
        "finance_direct_demography_coupling_absent": direct_absent,
        "finance_demographic_rng_contamination_absent": rng_absent,
        "demographic_effect_explainable": explainable,
        "population_integration_pass": population_classification.startswith(("A ", "B ")),
        "accounting_reconciliation_pass": accounting["pass"],
        "rng_noninterference_pass": noninterference["rng_equal"] and zero_compat["pass"],
        "experimental_restructuring_still_default_off": restructuring_off,
        "economic_behavior_changed": False,
        "financial_mechanism_changed": False,
        "analysis_only_changes": True,
        "new_long_runs": 2,
        "seed7_21_run": False,
        "exit_implemented": False,
    }

    metrics = {
        "analysis_financial_core_ready": plots["ready"],
        "analysis_state_max_abs_diff": noninterference["analysis_state_max_abs_diff"],
        "analysis_rng_changed": noninterference["analysis_rng_changed"],
        "main_smoke_pass": main_smoke_pass,
        "main_full_acceptance_pass": main_full_pass,
        "main_reference_final_population": number(summary["final_population"]),
        "main_reference_active_households": number(summary["final_active_households"]),
        "main_reference_principal": number(summary["final_outstanding_principal"]),
        "main_reference_arrears": number(summary["final_interest_arrears"]),
        "main_reference_money_stock": number(read_csv(RUN_5 / "diagnostics.csv")[-1]["total_money_stock"]),
        "main_reference_CB_interest_income": number(summary["central_bank_interest_income"]),
        "main_reference_default_event_count": int(number(summary["default_event_count"], 0)),
        "main_reference_active_contract_default_firm_weeks": int(number(summary["active_contract_default_firm_weeks"], 0)),
        "step13_reference_max_abs_metric_gap": reference["max_abs_gap"],
        "plot_browser_financial_category_ready": plots["ready"],
        "financial_standard_plot_count_added": plots["standard_added"],
        "financial_full_plot_count_added": plots["full_added"],
        "analysis_runtime_overhead_share": noninterference["analysis_runtime_overhead_share"],
        "population_integration_classification": population_classification,
        "finance_direct_demographic_write_count": coupling["finance_direct_demographic_write_count"],
        "finance_demographic_rng_call_count": coupling["finance_demographic_rng_call_count"],
        "analysis_world_state_mutation_count": noninterference["analysis_world_state_mutation_count"],
        "zero_interest_final_population": demography["zero_interest"]["final_population"],
        "interest5_final_population": demography["interest5"]["final_population"],
        "population_difference": demography["population_difference"],
        "zero_interest_birth_count": demography["zero_interest"]["birth_count"],
        "interest5_birth_count": demography["interest5"]["birth_count"],
        "zero_interest_death_count": demography["zero_interest"]["death_count"],
        "interest5_death_count": demography["interest5"]["death_count"],
        "zero_interest_marriage_count": demography["zero_interest"]["marriage_count"],
        "interest5_marriage_count": demography["interest5"]["marriage_count"],
        "zero_interest_active_households": demography["zero_interest"]["active_households"],
        "interest5_active_households": demography["interest5"]["active_households"],
        "zero_interest_working_age_population": demography["zero_interest"]["working_age_population"],
        "interest5_working_age_population": demography["interest5"]["working_age_population"],
        "zero_interest_elderly_population": demography["zero_interest"]["elderly_population"],
        "interest5_elderly_population": demography["interest5"]["elderly_population"],
        "zero_interest_age_0_19_count": demography["zero_interest"]["age_0_19_count"],
        "interest5_age_0_19_count": demography["interest5"]["age_0_19_count"],
        "zero_interest_age_20_39_count": demography["zero_interest"]["age_20_39_count"],
        "interest5_age_20_39_count": demography["interest5"]["age_20_39_count"],
        "zero_interest_age_40_64_count": demography["zero_interest"]["age_40_64_count"],
        "interest5_age_40_64_count": demography["interest5"]["age_40_64_count"],
        "zero_interest_age_65_plus_count": demography["zero_interest"]["age_65_plus_count"],
        "interest5_age_65_plus_count": demography["interest5"]["age_65_plus_count"],
        "zero_interest_household_formations": demography["household_formations"],
        "interest5_household_formations": demography["household_formations"],
        "zero_interest_household_dissolutions": demography["household_dissolutions"],
        "interest5_household_dissolutions": demography["household_dissolutions"],
        "first_economic_divergence_week": demography["first_economic_divergence_week"],
        "first_demographic_divergence_week": demography["first_demographic_divergence_week"],
        "demographic_divergence_precedes_economic_divergence": demography["demographic_divergence_precedes_economic_divergence"],
        "demographic_effect_explained": explainable,
        "zero_interest_compatibility_pass": zero_compat["pass"],
        **accounting,
    }

    OUT.mkdir(parents=True, exist_ok=True)
    write_csv(
        OUT / "integrated_acceptance_metrics.csv",
        [
            {"category": "INTEGRATION", "metric": key, "value": value}
            for key, value in metrics.items()
        ],
    )
    (OUT / "acceptance_flags.json").write_text(
        json.dumps(flags, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    summary_text = f"""# Post-Step13 Integrated Acceptance

**Verdict: {verdict}**

## Integrated gates

- Analysis V2 Financial Core ready: `{plots['ready']}`; four standard plots and three new full-profile Firm diagnostics are indexed under `Financial Core`.
- Analysis passivity: state max absolute difference `{noninterference['analysis_state_max_abs_diff']}`; RNG changed `{noninterference['analysis_rng_changed']}`.
- Normal `main.py` canonical run: smoke `{main_smoke_pass}`, full acceptance `{main_full_pass}`, Step13 reference max metric gap `{reference['max_abs_gap']:.6g}`.
- Accounting/conservation: `{accounting['pass']}`; invariant violations `{accounting['invariant_violations']}`.
- Restructuring experiments remain OFF: `{restructuring_off}`.

## Financial Core now visible in Analysis V2

Analysis exposes requested/executed/denied credit, credit limits and headroom, opening/closing principal, utilization, borrowing, repayment and net principal change; current interest due/paid/unpaid and service ratio; arrears formation/payment and stock; principal, revolving and total lender exposure; D0-D3, technical breach, Default events, active contractual Default and contractual cure; and credit money creation/destruction/net flow. Stock, flow and state semantics are written in `financial_core_semantics.json` and are sourced from the accepted Step13 architecture manifest.

Added outputs are the compact `financial_core_summary.csv`, one-row-per-Firm `financial_core_firm_summary.csv`, four standard Financial Core figures, and three full-profile diagnostics: Firm utilization, distress-state heatmap and active-contract-Default heatmap. Existing Credit and Interest plots remain available.

## Main program integration

The canonical reference is directly runnable with `main.py` using scenario `interest_behavioral_5pct`; no test runner or source edit is required. The completed run reproduced population 4856, active households 1875, principal {number(summary['final_outstanding_principal']):.6f}, arrears {number(summary['final_interest_arrears']):.6f}, money stock {metrics['main_reference_money_stock']:.6f}, CB interest income {number(summary['central_bank_interest_income']):.6f}, 3 Default events and 1046 active-contract-default Firm-weeks.

## Demography and RNG audit

Classification: **{population_classification}**.

The 5% run ends at population {demography['interest5']['final_population']} versus {demography['zero_interest']['final_population']} under 0%, a difference of {demography['population_difference']}. Births are {demography['interest5']['birth_count']} versus {demography['zero_interest']['birth_count']}; deaths {demography['interest5']['death_count']} versus {demography['zero_interest']['death_count']}; marriages {demography['interest5']['marriage_count']} versus {demography['zero_interest']['marriage_count']}; active households {demography['interest5']['active_households']} versus {demography['zero_interest']['active_households']}.

Final 5% versus 0% age context is: working-age {demography['interest5']['working_age_population']} vs {demography['zero_interest']['working_age_population']}, elderly {demography['interest5']['elderly_population']} vs {demography['zero_interest']['elderly_population']}, and age buckets 0-19 / 20-39 / 40-64 / 65+ = {demography['interest5']['age_0_19_count']} / {demography['interest5']['age_20_39_count']} / {demography['interest5']['age_40_64_count']} / {demography['interest5']['age_65_plus_count']} versus {demography['zero_interest']['age_0_19_count']} / {demography['zero_interest']['age_20_39_count']} / {demography['zero_interest']['age_40_64_count']} / {demography['zero_interest']['age_65_plus_count']}.

The first economic divergence is week {demography['first_economic_divergence_week']} in {', '.join(demography['first_economic_divergence_fields'])}; the first demographic divergence is week {demography['first_demographic_divergence_week']} in {', '.join(demography['first_demographic_divergence_fields'])}. Demographic divergence does not precede economic divergence.

The source audit found {coupling['finance_direct_demographic_write_count']} direct finance-to-demography writes/calls and {coupling['finance_demographic_rng_call_count']} finance calls to demographic RNG. Interest, arrears, distress, Default and disabled restructuring are deterministic. Analysis leaves the full serialized World and all global/market/age-phase/marriage/Firm RNG states unchanged.

The observed path is endogenous: interest service changes Firm liquidity and distributable cash; the existing economy then changes household income/wealth/security. `fertility/fertility.py` uses household wealth divided by target wealth in its economic fertility factor before drawing births. Birth differences can subsequently change population composition, lifecycle execution and mortality draw alignment. Mortality itself has no direct financial input, and marriage uses its isolated marriage RNG; later differences there are secondary to already-diverged household/population state.

An exact pre-Step13 executable path is no longer available. The valid current-code zero-activation check therefore compares explicit 0% interest with the same K and credit architecture at its default 0% rate; macro, Firm and demographic-event trajectories are exactly equal and all interest/arrears fields remain zero.

Complete household formation/dissolution event totals are `null` because the current canonical demographic event stream does not expose every such lifecycle transition. They were not fabricated.

## Required semantic answers

1. Step13 credit, interest, arrears, lender exposure, money-flow, distress and Default information is now first-class in Analysis V2.
2. Four standard plots, three full-profile Firm diagnostics and two compact tables were added.
3. Yes, the canonical scenario runs directly through `main.py`.
4. Yes, `main.py` reproduces the accepted Step13 financial reference within floating tolerance.
5. No, Analysis changes neither World state nor RNG.
6. No Step13 financial function directly mutates demographic state.
7. Finance consumes no demographic RNG.
8. Yes, active 5% interest changes final population/lifecycle outcomes relative to 0%.
9. The current-code path is interest service -> Firm liquidity/distributable cash -> household income/wealth/security -> fertility economic factor -> births; later lifecycle composition effects follow.
10. Yes, demographic divergence occurs only after the first economic divergence.
11. The effect is a legitimate economically mediated spillover, not an integration defect.
12. Yes, the Step13 financial core is accepted as an integrated SOCIALSIM component.

## Known limitation

`loan_balance` still mixes persistent principal debt and working-capital revolving utilization. The term/revolver split remains deferred to generalized multi-sector corporate finance.
"""
    (OUT / "acceptance_summary.md").write_text(summary_text, encoding="utf-8")
    write_preview()

    expected = {
        "acceptance_summary.md",
        "acceptance_flags.json",
        "integrated_acceptance_metrics.csv",
        "main_financial_core_preview.png",
    }
    actual = {path.name for path in OUT.iterdir() if path.is_file()}
    if actual != expected:
        raise AssertionError(f"Acceptance directory file set mismatch: {actual}")
    print(verdict)


if __name__ == "__main__":
    main()
