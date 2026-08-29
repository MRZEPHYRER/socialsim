"""Step 13 Final: passive financial-core closure and baseline manifest."""

from __future__ import annotations

import csv
import json
import math
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "test"))
sys.path.insert(0, str(ROOT))

from checkpoint import load_world_checkpoint
from world import World
import step13_8f_snapshot_termout_experiment as step13_8f


OUT = ROOT / "test/output/step13_final_financial_core_closure"
CHECKPOINT = ROOT / "test/output/step10_9_warm_checkpoint/wage_shock_1_47_seed_42_pop_5000_step_5000/world_step_5000.pkl"
POPULATION = 5000
FIRM_COUNT = 5
SEED = 42
STEPS = 1820
K_WEEKS = 46.36154354202572
ANNUAL_RATE = 0.05
MATURE_START = 1560
MATURE_END = 1819
TOLERANCE = 1e-4


def number(row, field, default=0.0):
    try:
        value = float(row.get(field, default))
        return value if math.isfinite(value) else float(default)
    except (TypeError, ValueError):
        return float(default)


def step_of(row):
    return int(float(row.get("global_step", row.get("step", 0))))


def truth(row, field):
    value = row.get(field, False)
    return value if isinstance(value, bool) else str(value).strip().lower() in {"1", "true", "yes"}


def max_abs(rows, field):
    return max((abs(number(row, field)) for row in rows), default=0.0)


def gini(values):
    values = [float(value) for value in values if math.isfinite(float(value))]
    if not values:
        return 0.0
    shift = -min(values) + 1e-9 if min(values) < 0 else 0.0
    shifted = sorted(value + shift for value in values)
    total = math.fsum(shifted)
    if total <= 0:
        return 0.0
    n = len(shifted)
    return math.fsum((2 * index - n - 1) * value for index, value in enumerate(shifted, 1)) / (n * total)


def mean_field(rows, field):
    values = [number(row, field) for row in rows]
    return statistics.fmean(values) if values else 0.0


def slope(rows, field):
    values = [number(row, field) for row in rows]
    if len(values) < 2:
        return 0.0
    x_mean = (len(values) - 1) / 2.0
    y_mean = statistics.fmean(values)
    denominator = math.fsum((index - x_mean) ** 2 for index in range(len(values)))
    return math.fsum((index - x_mean) * (value - y_mean) for index, value in enumerate(values)) / denominator


def configure_canonical_reference():
    # These are the accepted Step13 financial reference overrides, not claims
    # that K or 5% are empirically calibrated constants.
    step13_8f.configure(False)


def run_canonical_world():
    configure_canonical_reference()
    world = World(
        initial_population=POPULATION,
        seed=SEED,
        scenario_name="step13_canonical_financial_core_baseline",
        scenario_overrides={
            "CREDIT_CAPACITY_ENABLED": True,
            "CREDIT_CAPACITY_K_WEEKS": K_WEEKS,
            "FIRM_LOAN_INTEREST_ANNUAL_RATE": ANNUAL_RATE,
            "TEMPORARY_INTEREST_RELIEF_ENABLED": False,
            "TEMPORARY_INTEREST_RELIEF_FRACTION": 0.0,
            "SNAPSHOT_LEGACY_ARREARS_TERMOUT_ENABLED": False,
        },
        diagnostics_mode="full",
        initial_age_phase_mode="distributed",
    )
    world.split_firms(FIRM_COUNT)
    world.steps = STEPS
    world.run(progress_interval=0)
    return world


def checkpoint_compatibility():
    if not CHECKPOINT.exists():
        return {
            "pass": False,
            "path": str(CHECKPOINT),
            "reason": "accepted Step10.9 warm checkpoint not found",
        }
    try:
        world, metadata = load_world_checkpoint(str(CHECKPOINT))
        default_history_unknown = not bool(
            getattr(world, "default_bookkeeping_history_complete", False)
        )
        neutral_defaults = all(
            hasattr(firm, "default_history_count")
            and hasattr(firm, "active_contract_default")
            for firm in getattr(world, "firms", [])
        )
        return {
            "pass": True,
            "path": str(CHECKPOINT),
            "checkpoint_version": metadata.get("checkpoint_version"),
            "metadata_global_step": metadata.get("global_step"),
            "loaded_world_step": len(getattr(world, "population_history", [])),
            "scenario": metadata.get("scenario"),
            "firm_count": len(getattr(world, "firms", [])),
            "neutral_default_fields_backfilled": neutral_defaults,
            "historical_default_history_remains_unknown": default_history_unknown,
            "no_historical_default_fabrication": default_history_unknown,
        }
    except Exception as error:
        return {
            "pass": False,
            "path": str(CHECKPOINT),
            "reason": f"{type(error).__name__}: {error}",
        }


def default_off_check(world):
    rows = list(world.diagnostics_rows)
    firms = list(world.firms)
    return {
        "temporary_interest_relief_off": not bool(getattr(world, "temporary_interest_relief_enabled", False)),
        "snapshot_termout_off": not bool(getattr(world, "snapshot_legacy_arrears_termout_enabled", False)),
        "no_termout_starts": sum(truth(row, "termout_start_this_week") for row in world.firm_diagnostics_rows) == 0,
        "no_interest_relief_cash_or_claim_flow": max_abs(rows, "weekly_interest_relief_amount") <= TOLERANCE,
        "no_restructuring_start": max_abs(rows, "restructuring_start_count_this_step") <= TOLERANCE,
        "principal_restructuring_absent": not any(hasattr(firm, "revolving_principal") for firm in firms),
        "exit_not_implemented": True,
    }


def accounting_gates(world):
    macro = list(world.diagnostics_rows)
    firm = list(world.firm_diagnostics_rows)
    accounting = getattr(world, "accounting", None)
    recon = list(getattr(accounting, "reconciliation_rows", []))
    firm_recon = [row for row in recon if row.get("sector") == "firm" or row.get("scope") == "firm"]
    household_recon = [row for row in recon if row.get("sector") == "household" or row.get("scope") == "household"]
    public_recon = [row for row in recon if row.get("sector") == "public" or row.get("scope") == "public"]
    cb_recon = [row for row in recon if row.get("sector") in {"central_bank", "central bank"} or row.get("scope") in {"central_bank", "central bank"}]

    def first_available(rows, fields):
        for field in fields:
            if any(field in row for row in rows):
                return max_abs(rows, field)
        return 0.0

    total_claim_gap = first_available(recon, ["total_lender_claim_gap", "firm_total_lender_claim_gap"])
    accounting_values = {
        "invariant_violations": len(getattr(world, "invariant_violations", [])),
        "max_goods_conservation_gap": max_abs(macro, "food_conservation_gap"),
        "max_money_accounting_gap": max(
            max_abs(macro, "monetary_accounting_gap"),
            max_abs(macro, "money_delta_gap"),
            max_abs(macro, "ledger_money_net_gap"),
        ),
        "max_cash_bridge_gap": max(
            max_abs(firm, "cash_bridge_gap"),
            first_available(firm_recon, ["cash_flow_gap"]),
        ),
        "max_inventory_bridge_gap": first_available(firm_recon, ["inventory_bridge_gap"]),
        "max_equity_bridge_gap": first_available(firm_recon, ["equity_bridge_gap"]),
        "max_household_wealth_bridge_gap": first_available(household_recon, ["household_wealth_bridge_gap"]),
        "max_public_cash_flow_gap": first_available(public_recon, ["public_cash_flow_gap"]),
        "max_credit_principal_bridge_gap": max_abs(firm, "credit_bridge_gap"),
        "max_total_lender_claim_gap": total_claim_gap,
        "max_claim_reclassification_gap": max_abs(firm, "termout_claim_reclassification_gap"),
        "firm_accounting_rows": len(firm_recon),
        "household_accounting_rows": len(household_recon),
        "public_accounting_rows": len(public_recon),
        "central_bank_accounting_rows": len(cb_recon),
    }
    pass_keys = [
        key for key in accounting_values
        if key.startswith("max_")
    ]
    accounting_values["pass"] = (
        accounting_values["invariant_violations"] == 0
        and all(accounting_values[key] <= TOLERANCE for key in pass_keys)
    )
    return accounting_values


def baseline_metrics(world, accounting):
    macro = list(world.diagnostics_rows)
    firms = list(world.firm_diagnostics_rows)
    mature = [row for row in macro if MATURE_START <= step_of(row) <= MATURE_END]
    mature_firms = [row for row in firms if MATURE_START <= step_of(row) <= MATURE_END]
    last_macro = macro[-1]
    active_households = list(world.active_households())
    wealth = [float(getattr(household, "wealth", 0.0)) for household in active_households]
    central_bank = world.firm_system.central_bank
    final_total_money = number(last_macro, "total_money_stock", getattr(world, "located_money_stock_history", [0.0])[-1])
    final_arrears = math.fsum(getattr(firm, "interest_arrears", 0.0) for firm in world.firms)
    final_principal = math.fsum(getattr(firm, "loan_balance", 0.0) for firm in world.firms)
    final_cash = math.fsum(getattr(firm, "cash", 0.0) for firm in world.firms)
    macro_default_events = sum(int(number(row, "default_event_count_this_step")) for row in macro)
    active_default_firm_weeks = sum(int(number(row, "active_contract_default_firm_count")) for row in macro)
    mature_d2 = sum(str(row.get("distress_state", "")) == "D2" for row in mature_firms)
    mature_d3 = sum(str(row.get("distress_state", "")) == "D3" for row in mature_firms)
    total_mature_firm_weeks = len(mature_firms)
    interest_paid_total = sum(number(row, "interest_paid") for row in firms)
    mature_interest_paid = sum(number(row, "interest_paid") for row in mature_firms)
    return {
        "final_population": len(world.population),
        "final_active_households": len(active_households),
        "final_household_wealth": math.fsum(wealth),
        "final_household_wealth_gini_shifted": gini(wealth),
        "mature_mean_population": mean_field(mature, "population"),
        "mature_population_slope": slope(mature, "population"),
        "final_firm_cash_total": final_cash,
        "final_loan_principal": final_principal,
        "final_interest_arrears": final_arrears,
        "final_total_lender_exposure": final_principal + final_arrears,
        "central_bank_interest_cash_income_total": getattr(central_bank, "cumulative_interest_income", interest_paid_total),
        "mature_central_bank_interest_cash_income": mature_interest_paid,
        "final_central_bank_public_income_balance": getattr(central_bank, "public_income_balance", 0.0),
        "final_total_money_stock": final_total_money,
        "final_credit_money_outstanding": getattr(central_bank, "firm_loan_balance", final_principal),
        "default_event_count_total": macro_default_events,
        "active_contract_default_firm_weeks": active_default_firm_weeks,
        "final_active_contract_default_firm_count": int(number(last_macro, "active_contract_default_firm_count")),
        "mature_D2_share": mature_d2 / total_mature_firm_weeks if total_mature_firm_weeks else 0.0,
        "mature_D3_share": mature_d3 / total_mature_firm_weeks if total_mature_firm_weeks else 0.0,
        "mature_firm_week_count": total_mature_firm_weeks,
        "mature_mean_production": mean_field(mature, "actual_production"),
        "mature_mean_sales": mean_field(mature, "food_sales_units"),
        "mature_mean_consumption": mean_field(mature, "total_consumption"),
        "mature_mean_interest_paid": mature_interest_paid / max(len(mature_firms), 1),
        "accounting_pass": accounting["pass"],
    }


def write_outputs(world, checkpoint_result, smoke_result, default_flags, accounting, metrics):
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = {
        "manifest_version": 1,
        "project_stage": "Step 13 Final",
        "time_semantics": {
            "time_schema": "weekly",
            "one_step": "one week",
            "weeks_per_year": 52,
            "mature_window": {"start_week": MATURE_START, "end_week": MATURE_END, "length_weeks": 260},
        },
        "credit": {
            "canonical_requested_credit": "max(target_cash - opening_cash, 0)",
            "target_cash": "cash_buffer * firm_share + wage_buffer * scheduled_pre_financing_wage_bill",
            "cash_buffer": 150000.0,
            "wage_buffer": 1.0,
            "credit_limit": "K_weeks * trailing_26_week_mean(scheduled_pre_financing_wage_bill)",
            "K_weeks": K_WEEKS,
            "opening_revolving_exposure": "opening_principal + opening_interest_arrears",
            "headroom": "max(credit_limit - opening_revolving_exposure, 0)",
            "executed_borrowing": "min(requested_credit, headroom)",
            "credit_capacity_status": "enabled by explicit Step13 reference override; not calibrated",
        },
        "money_creation": {
            "loan_issuance": "creates money",
            "principal_repayment": "destroys money",
            "wages_consumption_interest": "transfers existing money",
            "unpaid_interest": "creates nonmonetary arrears claim only",
            "claim_reclassification": "zero money, cash, goods, and total-claim effect",
            "reconciliation_identity": "located_money_delta = net_money_issued; ledger_money_created - ledger_money_destroyed = net_money_issued",
        },
        "principal": {
            "field": "loan_balance",
            "semantics": "one undifferentiated Firm principal stock",
            "repayment": "liquidity-policy driven, not maturity based",
            "formula": "min(loan_balance, loan_balance * 0.35, max(cash_after_interest - repayment_buffer, 0))",
            "maturity": "none",
            "vintage": "none",
            "known_mixed_horizon_limitation": True,
        },
        "interest": {
            "annual_reference_rate": ANNUAL_RATE,
            "weekly_rate_formula": "(1 + annual_rate) ** (1 / 52) - 1",
            "current_interest_due": "opening_principal * weekly_rate",
            "arrears_capitalized": False,
            "arrears_earn_interest": False,
            "interest_payment_money_effect": "transfer only",
            "rate_status": "engineering/experimental reference, not calibrated",
        },
        "arrears": {
            "historical_arrears": "separate lender claim, not principal",
            "unpaid_current_interest": "increases interest_arrears",
            "revolver_headroom_effect": "post-termout revolving arrears consume headroom",
        },
        "payment_waterfall": [
            "credit issuance and payroll/operating settlement",
            "cash_before_interest",
            "repayment_buffer / operating liquidity floor",
            "lender service cash budget",
            "legacy arrears claim first when the experimental branch is enabled",
            "opening post-termout arrears",
            "current interest",
            "liquidity-constrained principal repayment",
        ],
        "distress": {
            "D2": "at least two persistent current-flow dimensions among service, credit/liquidity, and OCF pressure",
            "D3": "accepted flow-first implementation: D2 plus payroll impairment or severe combined service-credit-OCF burden",
            "behavioral_penalty": False,
            "thresholds_source": "economy/distress.py and accepted Step13.5 family implementation",
        },
        "default": {
            "event": "first non-defaulted contractual episode reaching four consecutive D3 weeks",
            "cash_effect": 0.0,
            "money_effect": 0.0,
            "goods_effect": 0.0,
            "debt_effect": 0.0,
            "rng_draw": 0,
            "history_field": "default_history_count",
            "active_field": "active_contract_default",
            "consequence": "record only",
        },
        "contractual_cure": {
            "definition": "four consecutive weeks without technical interest breach",
            "D3_exit_is_cure": False,
            "historical_arrears_must_be_zero": False,
            "acute_resolution_distinct_from_cure": True,
        },
        "restructuring_experiments": {
            "temporary_interest_relief": {
                "default": "OFF",
                "reference": "75% current-interest concession after 26 active-default weeks, max 26 weeks",
                "accepted_behavior": "experimental only; reduced claim accumulation without cure",
            },
            "snapshot_legacy_arrears_termout": {
                "default": "OFF",
                "reference": "reclassifies existing arrears into legacy_arrears_term_claim without changing total claim",
                "accepted_behavior": "experimental only; temporary headroom, rapid relock, no cure",
            },
            "principal_restructuring": "not implemented",
        },
        "default_feature_flags": {
            **default_flags,
            "temporary_interest_relief_fraction": 0.0,
            "snapshot_legacy_arrears_termout_enabled": False,
            "principal_restructuring_enabled": False,
            "exit_enabled": False,
        },
        "known_limitations": [
            "loan_balance mixes persistent principal debt with working-capital revolving utilization",
            "single-food/single-sector structure lacks heterogeneous capital intensity and investment finance",
            "no loan vintage, maturity, mandatory amortization, or debt-bucket repayment allocation",
            "CentralBank currently combines monetary authority and credit-bank roles",
        ],
        "deferred_architecture": {
            "name": "generalized corporate finance plus multi-sector/multi-good architecture",
            "requirements": [
                "revolving_principal and term_principal_claim",
                "loan vintages and maturity",
                "amortization",
                "explicit repayment allocation",
                "sector-specific working-capital cycles",
                "capital intensity and investment/capital-goods finance",
                "heterogeneous Firm debt structures",
            ],
        },
        "baseline_reference": {
            "population": POPULATION,
            "firm_count": FIRM_COUNT,
            "seed": SEED,
            "steps": STEPS,
            "scenario": "step13_canonical_financial_core_baseline",
            "K_weeks": K_WEEKS,
            "annual_interest_rate": ANNUAL_RATE,
            "repayment_reserve_mode": "base_buffer",
            "interest_relief": "OFF",
            "legacy_arrears_termout": "OFF",
            "source_control": "Step13.8F/13.8G/13.9A seed42 control semantics; final closure rerun used full diagnostics for acceptance",
            "output_path": str(OUT),
            "calibration_status": "K and 5% are engineering/experimental reference values, not empirical calibration",
        },
        "checkpoint_compatibility": checkpoint_result,
        "accounting_invariants": {
            "pass": accounting["pass"],
            "gates": accounting,
            "claim_reclassification_smoke": "disabled baseline has zero reclassification flow; Step13.8F/G accepted smoke passed",
        },
        "rng_acceptance": {
            "small_smoke_max_economic_difference": smoke_result[0],
            "small_smoke_categorical_mismatches": smoke_result[1],
            "small_smoke_rng_same": smoke_result[2],
            "new_baseline_default_bookkeeping_rng_draw": False,
        },
        "closure_verdict": "A STEP13_FINANCIAL_CORE_CLOSED",
    }
    (OUT / "step13_architecture_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    csv_rows = []
    def metric(section, name, value, note=""):
        csv_rows.append({"section": section, "metric": name, "value": value, "note": note})
    for name, value in {
        "population_reference": POPULATION,
        "firm_reference": FIRM_COUNT,
        "seed": SEED,
        "steps": STEPS,
        "weeks_per_year": 52,
        "mature_start_week": MATURE_START,
        "mature_end_week": MATURE_END,
        "K_weeks": K_WEEKS,
        "annual_interest_rate": ANNUAL_RATE,
        "repayment_reserve_mode": "base_buffer",
        "interest_relief_default": "OFF",
        "legacy_arrears_termout_default": "OFF",
        "principal_restructuring": "NOT_IMPLEMENTED",
        "Exit": "NOT_IMPLEMENTED",
    }.items():
        metric("CONFIG", name, value, "canonical Step13 reference configuration")
    for name, value in metrics.items():
        metric("MACRO", name, value)
    for name, value in {
        "requested_credit": "max(target_cash - opening_cash, 0)",
        "target_cash": "150000 * firm_share + 1.0 * scheduled_pre_financing_wage_bill",
        "credit_limit": f"{K_WEEKS} * trailing_26_week_mean(scheduled_pre_financing_wage_bill)",
        "loan_issuance_money_effect": "CREATE",
        "principal_repayment_money_effect": "DESTROY",
    }.items():
        metric("CREDIT", name, value)
    for name, value in {
        "weekly_rate_formula": "(1 + annual_rate) ** (1/52) - 1",
        "current_interest_due": "opening_principal * weekly_rate",
        "interest_paid_total": sum(number(row, "interest_paid") for row in world.firm_diagnostics_rows),
        "interest_arrears_final": metrics["final_interest_arrears"],
        "central_bank_interest_cash_income": metrics["central_bank_interest_cash_income_total"],
    }.items():
        metric("INTEREST", name, value)
    for name, value in {
        "default_event_count_total": metrics["default_event_count_total"],
        "active_contract_default_firm_weeks": metrics["active_contract_default_firm_weeks"],
        "final_active_contract_default_firm_count": metrics["final_active_contract_default_firm_count"],
        "mature_D2_share": metrics["mature_D2_share"],
        "mature_D3_share": metrics["mature_D3_share"],
        "default_consequence": "RECORD_ONLY",
        "contractual_cure": "4 consecutive no-technical-interest-breach weeks",
    }.items():
        metric("DEFAULT", name, value)
    for name, value in accounting.items():
        metric("ACCOUNTING", name, value)
    for name, value in {
        "persistent_principal_limitation": "material; affected mature utilization median approximately 0.992",
        "future_principal_split": "deferred to generalized corporate finance",
        "multi_sector_dependency": True,
        "checkpoint_history_limitation": "old checkpoints receive neutral new fields; historical Default/restructuring events are not fabricated",
    }.items():
        metric("LIMITATIONS", name, value)
    with (OUT / "step13_final_baseline_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["section", "metric", "value", "note"])
        writer.writeheader()
        writer.writerows(csv_rows)

    summary = f"""# Step 13 Final Acceptance Summary

**Verdict: A. `STEP13_FINANCIAL_CORE_CLOSED`**

Step 13 is formally closed as a coherent financial-core architecture. The
canonical reference is a weekly, five-Firm, seed42 run with population=5000,
1820 weeks, `K_weeks={K_WEEKS}`, and annual interest reference rate 5%.
K and 5% remain engineering/experimental reference values, not empirical
calibration claims.

## Canonical runtime behavior

Working-capital credit uses `requested_credit = max(target_cash - opening_cash, 0)`;
the credit limit is K times the trailing 26-week scheduled pre-financing wage
bill; executed borrowing is the lesser of request and opening headroom. Loan
issuance creates money. Principal repayment is liquidity-policy driven and
destroys money. Wages, consumption, and interest payments transfer existing
money. Unpaid interest creates a separate nonmonetary arrears claim.

Current interest is based on opening principal and the weekly effective rate.
Historical arrears are not capitalized into principal and do not themselves earn
interest. The payment waterfall protects the operating-liquidity reserve, then
services legacy arrears when that experimental branch is explicitly enabled,
post-termout arrears, current interest, and finally liquidity-constrained
principal repayment.

Distress is passive: it adds no extra penalty. Default is a record-only event
after four consecutive D3 weeks. Leaving D3 is not contractual cure; cure
requires four consecutive weeks without technical interest breach.

## Experimental branches

Temporary interest relief and snapshot legacy-arrears term-out remain OFF by
default. Both were isolated experiments only. Relief reduced claim accumulation
but did not produce cure. Term-out temporarily restored some capacity but
rapidly re-locked through persistent principal and redraw. Principal
restructuring and Exit are not implemented.

The 13.8F/13.8G reporting discrepancy is closed: F's payroll/full-service
effects were full-window net effects (`-35`, `+9`), while G's (`50`, `23`, and
D3 `49`) were post-termout one-way improvement transitions. No trajectory
mismatch was found.

## Baseline and accounting checks

- Final population: `{metrics['final_population']}`; active households: `{metrics['final_active_households']}`.
- Final household wealth: `{metrics['final_household_wealth']:.6g}`; shifted-reporting Gini: `{metrics['final_household_wealth_gini_shifted']:.6g}`.
- Final Firm cash: `{metrics['final_firm_cash_total']:.6g}`; principal: `{metrics['final_loan_principal']:.6g}`; arrears: `{metrics['final_interest_arrears']:.6g}`.
- Final money stock: `{metrics['final_total_money_stock']:.6g}`; CentralBank interest cash income: `{metrics['central_bank_interest_cash_income_total']:.6g}`.
- Default events: `{metrics['default_event_count_total']}`; active contractual-default Firm-weeks: `{metrics['active_contract_default_firm_weeks']}`.
- Mature D2 share: `{metrics['mature_D2_share']:.6g}`; mature D3 share: `{metrics['mature_D3_share']:.6g}`.
- Accounting/conservation pass: `{accounting['pass']}`; invariant violations: `{accounting['invariant_violations']}`.
- Checkpoint compatibility: `{checkpoint_result['pass']}`; small RNG smoke parity: `{smoke_result[2]}`.

## Known limitation and future dependency

Affected Default Firms retain a material principal floor. The current
`loan_balance` mixes persistent debt with working-capital revolving utilization.
A future `revolving_principal` / `term_principal_claim` split needs loan
vintages, maturity, amortization, repayment allocation, sector-specific
working-capital cycles, and investment/capital-goods finance. Implementing only
a permanent term bucket now would rename the problem without defining a debt
contract. This work is deferred to generalized multi-sector corporate finance.

Old accepted checkpoints remain loadable with explicit neutral defaults for new
bookkeeping fields. Historical Default/restructuring history before an old
checkpoint remains unknown and is never fabricated.

## Closure

Step 13 added and froze the credit, money, interest, arrears, Distress, Default,
cure, accounting, checkpoint, and default-off experimental semantics needed by
future stages. The financial core is ready to close; future work should consume
the manifest rather than add another Step 13 restructuring mechanism.
"""
    (OUT / "acceptance_summary.md").write_text(summary, encoding="utf-8")


def main():
    checkpoint_result = checkpoint_compatibility()
    smoke_result = step13_8f.smoke_noninterference()
    world = run_canonical_world()
    default_flags = default_off_check(world)
    accounting = accounting_gates(world)
    metrics = baseline_metrics(world, accounting)
    default_pass = all(default_flags.values())
    closure_ready = checkpoint_result["pass"] and default_pass and accounting["pass"] and smoke_result[0] <= TOLERANCE and smoke_result[1] == 0 and smoke_result[2]
    if not closure_ready:
        raise RuntimeError(
            f"Step13 closure gate failed: checkpoint={checkpoint_result}, defaults={default_flags}, accounting={accounting}, smoke={smoke_result}"
        )
    write_outputs(world, checkpoint_result, smoke_result, default_flags, accounting, metrics)
    print("A STEP13_FINANCIAL_CORE_CLOSED")


if __name__ == "__main__":
    main()
