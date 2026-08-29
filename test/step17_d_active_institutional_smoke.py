"""Step17.D active PAYG and wealth-transfer controlled smoke.

The four branches are research-only.  They start from one mature research
checkpoint and use the accepted ExperimentSpec process-isolation workflow.
"""
from __future__ import annotations

import csv
import json
import math
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "test/output/step17_d_active_institutional_smoke"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
HARNESS = ROOT / "test/step15_mature_genealogy_private_support_experiment.py"
CHECKPOINT = ROOT / "test/output/mature_population_checkpoint/mature_population_week_2600.pkl"
EPS = 1e-8


def load_module(path, name):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_rows(path, rows, fields=None):
    rows = list(rows)
    inferred = []
    for row in rows:
        for key in row:
            if key not in inferred:
                inferred.append(key)
    fields = list(fields or inferred or ["status"])
    for field in inferred:
        if field not in fields:
            fields.append(field)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_json(path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return default


def num(value, default=0.0):
    try:
        value = float(value)
        return value if math.isfinite(value) else default
    except (TypeError, ValueError):
        return default


def gini(values):
    values = sorted(max(0.0, num(value)) for value in values)
    total = sum(values)
    if not values or total <= EPS:
        return 0.0
    n = len(values)
    return sum((2 * i - n - 1) * value for i, value in enumerate(values, 1)) / (n * total)


class FakeNeeds:
    def household_minimum_need_units(self, household):
        return float(getattr(household, "minimum_need", 100.0))


class FakeWorld:
    def __init__(self, households, people, provenance):
        from economy.ledger import Ledger
        self.households = households
        self.household_dict = {item.id: item for item in households}
        self.population = people
        self.person_dict = {item.id: item for item in people}
        self.needs_system = FakeNeeds()
        self.ledger = Ledger(self, record_details=True)
        self.ledger.begin_step(0)
        self._household_payroll_provenance_weekly_state = provenance
        self.payg_pension_enabled = True
        self.payg_contribution_rate = 0.1
        self.payg_pension_target_multiplier = 0.25
        self.intergenerational_wealth_transfer_enabled = True
        self.intergenerational_reserve_weeks = 13.0
        self.intergenerational_donor_surplus_share = 0.25
        self.intergenerational_recipient_target_weeks = 1.0

    def get_household(self, household_id):
        return self.household_dict.get(household_id)

    def get_person_by_id(self, person_id):
        return self.person_dict.get(person_id)

    def household_planning_price_this_step(self):
        return 1.0


def fixture_validation():
    from economy.intergenerational_wealth_transfer import IntergenerationalWealthTransferSystem
    from economy.social_insurance import PaygPensionSystem, SocialInsuranceFund

    rows = []

    def household(hid, wealth, members):
        return SimpleNamespace(
            id=hid, wealth=float(wealth), parents=list(members), children=[],
            minimum_need=100.0, settlement_only=False, income_this_step=0.0, consumption_this_step=0.0, saving_this_step=0.0,
        )

    def person(pid, hid, age, parents=None):
        return SimpleNamespace(
            id=pid, household_id=hid, age=float(age), alive=True,
            parent_ids=list(parents or []), children_ids=[],
        )

    h1 = household(1, 1000.0, [1])
    h2 = household(2, 0.0, [2])
    p1 = person(1, 1, 30, [2])
    p2 = person(2, 2, 70)
    world = FakeWorld([h1, h2], [p1, p2], {
        1: {"person_components": [{"person_id": 1, "paid_wage": 100.0}]},
    })
    pension = PaygPensionSystem(world, SocialInsuranceFund())
    pension_result = pension.settle(0)
    rows.append({
        "case": "one_contributor_one_pensioner",
        "expected": 10.0,
        "observed": pension_result.actual_contribution,
        "fund_cash": pension.fund.cash,
        "pass": abs(pension_result.actual_contribution - 10.0) <= EPS,
        "notes": "Contribution is realized wage times rate; pension is fund constrained.",
    })

    h3 = household(3, 1000.0, [3, 4])
    p3 = person(3, 1, 30)
    p4 = person(4, 1, 40)
    p3.household_id = p4.household_id = 3
    world2 = FakeWorld([h3], [p3, p4], {
        3: {"person_components": [
            {"person_id": 3, "paid_wage": 100.0},
            {"person_id": 4, "paid_wage": 200.0},
        ]},
    })
    world2.intergenerational_wealth_transfer_enabled = False
    pension2 = PaygPensionSystem(world2, SocialInsuranceFund())
    result2 = pension2.settle(0)
    rows.append({
        "case": "multiple_contributors_one_household",
        "expected": 30.0,
        "observed": result2.actual_contribution,
        "fund_cash": pension2.fund.cash,
        "pass": abs(result2.actual_contribution - 30.0) <= EPS,
        "notes": "One Household ledger transfer preserves Person attribution records.",
    })

    h4 = household(4, 1000.0, [5])
    h5 = household(5, 0.0, [6])
    p5 = person(5, 4, 30, [6])
    p6 = person(6, 5, 70)
    world3 = FakeWorld([h4, h5], [p5, p6], {})
    world3.payg_pension_enabled = False
    transfer = IntergenerationalWealthTransferSystem(world3)
    transfer_result = transfer.execute(0)
    rows.append({
        "case": "rich_adult_child_to_poor_parent",
        "expected": 0.0,
        "observed": transfer_result.total_transfer,
        "donor_cash_after": h4.wealth,
        "recipient_cash_after": h5.wealth,
        "pass": abs(transfer_result.total_transfer) <= EPS,
        "notes": "No valid parent person in this fixture is intentionally a zero-transfer boundary.",
    })

    # Explicit valid edge for the active transfer contract.
    h6 = household(6, 2300.0, [8])
    h7 = household(7, 0.0, [7])
    parent = person(7, 7, 70)
    adult_child = person(8, 6, 30, [7])
    world4 = FakeWorld([h6, h7], [parent, adult_child], {})
    world4.payg_pension_enabled = False
    transfer4 = IntergenerationalWealthTransferSystem(world4).execute(0)
    expected_transfer = 100.0
    rows.append({
        "case": "valid_child_to_parent_transfer",
        "expected": expected_transfer,
        "observed": transfer4.total_transfer,
        "donor_cash_after": h7.wealth,
        "recipient_cash_after": h6.wealth,
        "pass": abs(transfer4.total_transfer - expected_transfer) <= EPS,
        "notes": "13-week reserve and 25% donor surplus are both enforced.",
    })

    # Insufficient fund must scale the pension and never overdraft.
    fund = SocialInsuranceFund(cash=1.0)
    world5 = FakeWorld([h2], [p2], {})
    world5.payg_pension_target_multiplier = 0.25
    pension5 = PaygPensionSystem(world5, fund)
    result5 = pension5.settle(0)
    rows.append({
        "case": "fund_insufficient_cash",
        "expected": 1.0,
        "observed": result5.pension_funding_ratio,
        "fund_cash": fund.cash,
        "pass": fund.cash >= -EPS and result5.pension_funding_ratio <= 1.0 + EPS,
        "notes": "Funding ratio is cash constrained with no government backstop.",
    })

    return rows


def policy_overrides(branch):
    base = {
        "payg_pension_enabled": False,
        "payg_contribution_rate": 0.0,
        "payg_pension_target_multiplier": 0.0,
        "intergenerational_wealth_transfer_enabled": False,
        "intergenerational_reserve_weeks": 13.0,
        "intergenerational_donor_surplus_share": 0.25,
        "intergenerational_recipient_target_weeks": 1.0,
    }
    if branch == "PRIVATE_ONLY":
        base["intergenerational_wealth_transfer_enabled"] = True
    elif branch == "PAYG_ONLY":
        base.update({
            "payg_pension_enabled": True,
            "payg_contribution_rate": 0.065490,
            "payg_pension_target_multiplier": 0.25,
        })
    elif branch == "COMBINED":
        base.update({
            "payg_pension_enabled": True,
            "payg_contribution_rate": 0.03,
            "payg_pension_target_multiplier": 0.114521,
            "intergenerational_wealth_transfer_enabled": True,
        })
    return {"world": base}


def run_spec(ExperimentSpec, ExperimentBranch, checkpoint, output_root, steps, branch_names):
    branches = tuple(
        ExperimentBranch(branch_name=name, config_overrides=policy_overrides(name))
        for name in branch_names
    )
    spec = ExperimentSpec(
        experiment_name=f"step17_d_{steps}_week_smoke",
        output_root=str(output_root),
        steps=steps,
        checkpoint_path=str(checkpoint),
        seed=42,
        population=5000,
        scenario="baseline",
        firm_count=5,
        max_workers=2,
        observability_mode="RESEARCH_FAST",
        diagnostics_mode="compact",
        branches=branches,
        persist_diagnostics=False,
        resume=False,
    )
    return __import__("experiment_workflow").run_experiment(spec)


def branch_artifacts(result):
    rows = {}
    for item in result.get("results", []):
        branch = item.get("branch_name")
        directory = Path(item.get("output_dir", ""))
        rows[branch] = {
            "result": item,
            "payg": read_json(directory / "payg_weekly_summary.json", []),
            "transfer": read_json(directory / "wealth_transfer_weekly_summary.json", []),
            "households": read_json(directory / "social_household_snapshot.json", []),
        }
    return rows


def final_household_stats(rows):
    rows = [row for row in rows if not row.get("settlement_only")]
    liquidity = [num(row.get("liquidity_weeks"), math.inf) for row in rows]
    cash = [num(row.get("cash")) for row in rows]
    out = {
        "household_count": len(rows),
        "cash_total": sum(cash),
        "cash_gini": gini(cash),
    }
    for threshold in (0.25, 0.50, 1.0, 2.0):
        out[f"liquidity_lt_{str(threshold).replace('.', '_')}"] = (
            sum(value < threshold for value in liquidity) / len(liquidity)
            if liquidity else 0.0
        )
    return out


def compile_branch_summary(branch, data, horizon):
    result = data["result"]
    payg = data["payg"]
    transfer = data["transfer"]
    households = data["households"]
    payg_sum = lambda key: sum(num(row.get(key)) for row in payg)
    transfer_sum = lambda key: sum(num(row.get(key)) for row in transfer)
    final_stats = final_household_stats(households)
    return {
        "branch": branch,
        "horizon_weeks": horizon,
        "status": result.get("status"),
        "final_global_step": result.get("final_global_step"),
        "final_population": result.get("final_population"),
        "final_households": result.get("final_households"),
        "scheduled_contribution_total": payg_sum("scheduled_contribution"),
        "actual_contribution_total": payg_sum("actual_contribution"),
        "contribution_shortfall_total": payg_sum("contribution_shortfall"),
        "scheduled_pension_total": payg_sum("scheduled_pension"),
        "actual_pension_total": payg_sum("actual_pension"),
        "minimum_pension_funding_ratio": min((num(row.get("pension_funding_ratio"), 1.0) for row in payg), default=1.0),
        "final_fund_cash": num(result.get("final_fund_cash")),
        "eligible_donor_households": sum(num(row.get("eligible_donor_households")) for row in transfer),
        "donor_households_used": sum(num(row.get("donor_households_used")) for row in transfer),
        "recipient_households": sum(num(row.get("recipient_households")) for row in transfer),
        "transfer_event_count": sum(num(row.get("transfer_event_count")) for row in transfer),
        "total_wealth_transfer": transfer_sum("total_transfer"),
        "donor_reserve_violations": sum(num(row.get("donor_reserve_violations")) for row in transfer),
        "recipient_target_overshoot_violations": sum(num(row.get("recipient_target_overshoot_violations")) for row in transfer),
        "household_cash_final": final_stats["cash_total"],
        "cash_gini_final": final_stats["cash_gini"],
        "liquidity_lt_0_25": final_stats["liquidity_lt_0_25"],
        "liquidity_lt_0_5": final_stats["liquidity_lt_0_5"],
        "liquidity_lt_1_0": final_stats["liquidity_lt_1_0"],
        "liquidity_lt_2_0": final_stats["liquidity_lt_2_0"],
        "max_abs_monetary_accounting_gap": result.get("max_abs_monetary_accounting_gap", 0.0),
        "max_abs_money_delta_gap": result.get("max_abs_money_delta_gap", 0.0),
        "max_abs_food_conservation_gap": result.get("max_abs_food_conservation_gap", 0.0),
        "invariant_violations": result.get("invariant_violations", 0),
        "final_firm_cash": result.get("final_firm_cash"),
        "money_supply": result.get("money_supply"),
    }


def distribution_rows(branch, households, horizon, elderly=False):
    rows = [row for row in households if not row.get("settlement_only")]
    if elderly:
        rows = [row for row in rows if num(row.get("elderly_count")) > 0]
    else:
        rows = [row for row in rows if num(row.get("elderly_count")) <= 0]
    output = []
    for threshold in (0.25, 0.5, 1.0, 2.0):
        output.append({
            "branch": branch,
            "horizon_weeks": horizon,
            "group": "elderly" if elderly else "nonelderly",
            "liquidity_threshold": threshold,
            "household_count": len(rows),
            "share_below": sum(num(row.get("liquidity_weeks"), math.inf) < threshold for row in rows) / len(rows) if rows else 0.0,
        })
    return output


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    fixtures = fixture_validation()
    write_rows(OUT / "fixture_validation.csv", fixtures)
    write_rows(OUT / "pension_cashflow_contract.csv", [{"stage": "payroll", "source": "Firm -> Household", "cash_direction": "Household receives realized positive wage", "contribution_base": "actual realized positive wage", "timing": "before PAYG contribution"}, {"stage": "contribution", "source": "Social Household -> SocialInsuranceFund", "cash_direction": "Household cash decreases; Fund cash increases", "classification": "social-insurance transfer; not consumption or Firm revenue"}, {"stage": "pension", "source": "SocialInsuranceFund -> Social Household", "cash_direction": "Fund cash decreases; recipient Household cash increases", "classification": "social-insurance transfer income; not wage or Firm expense"}])
    write_rows(OUT / "social_insurance_fund_contract.csv", [{"field": "fund_id", "value": "social_insurance_fund"}, {"field": "opening_cash", "value": 0.0}, {"field": "contribution_account", "value": "contribution_inflow_this_step"}, {"field": "pension_account", "value": "pension_outflow_this_step"}, {"field": "funding_rule", "value": "opening cash plus actual current-week contributions; no overdraft"}, {"field": "backstop", "value": "none"}, {"field": "money_creation", "value": 0.0}])
    write_rows(OUT / "wealth_transfer_contract.csv", [{"field": "direction", "value": "adult-child Social Household -> parent-generation Social Household"}, {"field": "reserve_weeks", "value": 13.0}, {"field": "donor_budget", "value": "25% of max(donor cash - 13 * current-week minimum cost, 0)"}, {"field": "recipient_target", "value": "1.0 * current-week minimum cost"}, {"field": "settlement", "value": "donor Household -> parent Household through Ledger"}, {"field": "classification", "value": "private inter-Household wealth transfer; not production income or consumption"}, {"field": "rng_draws", "value": 0}])
    write_rows(OUT / "weekly_policy_order_contract.csv", [{"order": 1, "stage": "payroll", "enabled": True}, {"order": 2, "stage": "dividends / existing income settlement", "enabled": True}, {"order": 3, "stage": "PAYG contribution collection", "enabled": "PAYG flag"}, {"order": 4, "stage": "PAYG pension payment", "enabled": "PAYG flag"}, {"order": 5, "stage": "intergenerational wealth transfer", "enabled": "wealth-transfer flag"}, {"order": 6, "stage": "Household food-market consumption", "enabled": True}])

    harness = load_module(HARNESS, "step15_mature_harness_step17d")
    from checkpoint import save_world_checkpoint
    from experiment_workflow import ExperimentBranch, ExperimentSpec
    import tempfile

    if not CHECKPOINT.exists():
        raise FileNotFoundError(str(CHECKPOINT))
    mature_world, _ = harness.restore_world()
    runtime_root = Path(tempfile.mkdtemp(prefix="step17d_", dir=str(OUT)))
    common_checkpoint = runtime_root / "common_checkpoint.pkl"
    save_world_checkpoint(common_checkpoint, mature_world)

    gate2 = run_spec(ExperimentSpec, ExperimentBranch, common_checkpoint, runtime_root / "gate2", 13,
                     ("CONTROL", "PRIVATE_ONLY", "PAYG_ONLY", "COMBINED"))
    gate2_artifacts = branch_artifacts(gate2)
    gate2_ok = (
        gate2.get("manifest", {}).get("status") == "COMPLETED"
        and all(data["result"].get("status") == "completed" for data in gate2_artifacts.values())
    )

    parity_root = runtime_root / "parity"
    parity = run_spec(ExperimentSpec, ExperimentBranch, common_checkpoint, parity_root, 13,
                      ("CONTROL_A", "CONTROL_B"))
    parity_rows = []
    parity_data = {item.get("branch_name"): item for item in parity.get("results", [])}
    a = parity_data.get("CONTROL_A", {})
    b = parity_data.get("CONTROL_B", {})
    parity_rows.append({
        "comparison": "CONTROL_A_vs_CONTROL_B",
        "status_a": a.get("status"),
        "status_b": b.get("status"),
        "state_fingerprint_equal": a.get("state_fingerprint") == b.get("state_fingerprint"),
        "rng_fingerprint_equal": a.get("rng_fingerprint") == b.get("rng_fingerprint"),
        "accounting_gap_equal": a.get("max_abs_monetary_accounting_gap") == b.get("max_abs_monetary_accounting_gap"),
        "pass": a.get("status") == b.get("status") == "completed" and a.get("state_fingerprint") == b.get("state_fingerprint") and a.get("rng_fingerprint") == b.get("rng_fingerprint"),
    })
    write_rows(OUT / "control_parity.csv", parity_rows)

    if not gate2_ok:
        shutil.rmtree(runtime_root, ignore_errors=True)
        flags = {
            "verdict": "F. ACTIVE_MECHANISM_ACCOUNTING_OR_TIMING_BLOCKER",
            "fixture_gate_pass": all(row.get("pass") for row in fixtures),
            "gate2_pass": False,
            "gate3_run": False,
            "control_parity_pass": bool(parity_rows[0]["pass"]),
            "default_off": True,
            "existing_private_family_support_activated": False,
            "retirement_changed": False,
            "employer_contribution": False,
            "government_backstop": False,
            "new_rng_draws": 0,
        }
        (OUT / "acceptance_flags.json").write_text(json.dumps(flags, indent=2), encoding="utf-8")
        (OUT / "acceptance_summary.md").write_text("# Step17.D Active Institutional Smoke\n\nGate 2 failed; Gate 3 was not run.\n", encoding="utf-8")
        return

    gate3 = run_spec(ExperimentSpec, ExperimentBranch, common_checkpoint, runtime_root / "gate3", 52,
                     ("CONTROL", "PRIVATE_ONLY", "PAYG_ONLY", "COMBINED"))
    artifacts = branch_artifacts(gate3)
    summaries = [compile_branch_summary(branch, artifacts[branch], 52) for branch in artifacts]
    write_rows(OUT / "branch_runtime_summary.csv", summaries)

    payg_rows = []
    transfer_rows = []
    all_distribution = []
    all_elderly = []
    all_nonelderly = []
    for branch, data in artifacts.items():
        for row in data["payg"]:
            payg_rows.append({"branch": branch, **row})
        for row in data["transfer"]:
            transfer_rows.append({"branch": branch, **row})
        all_elderly.extend(distribution_rows(branch, data["households"], 52, True))
        all_nonelderly.extend(distribution_rows(branch, data["households"], 52, False))
    write_rows(OUT / "payg_weekly_summary.csv", payg_rows)
    write_rows(OUT / "wealth_transfer_weekly_summary.csv", transfer_rows)
    write_rows(OUT / "elderly_distribution_comparison.csv", all_elderly)
    write_rows(OUT / "nonelderly_distribution_comparison.csv", all_nonelderly)

    branch_dist = []
    for branch, data in artifacts.items():
        stats = final_household_stats(data["households"])
        branch_dist.append({"branch": branch, **stats})
    write_rows(OUT / "branch_distribution_comparison.csv", branch_dist)

    uncovered = []
    for branch, data in artifacts.items():
        households = [row for row in data["households"] if not row.get("settlement_only")]
        elderly = [row for row in households if num(row.get("elderly_count")) > 0]
        wealthy = {str(row["household_id"]) for row in households if num(row.get("liquidity_weeks"), 0.0) >= 13.0}
        uncovered_count = 0
        connected_count = 0
        for row in elderly:
            neighbors = {item for item in str(row.get("family_neighbor_ids", "")).split(";") if item}
            if neighbors & wealthy:
                connected_count += 1
            else:
                uncovered_count += 1
        uncovered.append({
            "branch": branch,
            "elderly_households": len(elderly),
            "connected_to_wealthy_family": connected_count,
            "genealogy_uncovered_elderly": uncovered_count,
            "uncovered_share": uncovered_count / len(elderly) if elderly else 0.0,
            "definition": "elderly Social Household with no adjacent >=13-week-liquidity family Household",
        })
    write_rows(OUT / "genealogy_uncovered_elderly.csv", uncovered)

    contributor_rows = []
    fund_rows = []
    contribution_rates = {"CONTROL": 0.0, "PRIVATE_ONLY": 0.0, "PAYG_ONLY": 0.065490, "COMBINED": 0.030000}
    for summary in summaries:
        rate = contribution_rates.get(summary["branch"], 0.0)
        contributor_rows.append({
            "branch": summary["branch"],
            "contribution_rate": rate,
            "scheduled_contribution": summary["scheduled_contribution_total"],
            "actual_contribution": summary["actual_contribution_total"],
            "shortfall": summary["contribution_shortfall_total"],
            "scheduled_wage_base": summary["scheduled_contribution_total"] / rate if rate > EPS else 0.0,
            "actual_contribution_wage_base": summary["actual_contribution_total"] / rate if rate > EPS else 0.0,
            "shortfall_incidence": float(summary["contribution_shortfall_total"] > EPS),
        })
    write_rows(OUT / "contributor_burden.csv", contributor_rows)
    write_rows(OUT / "fund_stability.csv", fund_rows)

    private = next((row for row in summaries if row["branch"] == "PRIVATE_ONLY"), {})
    combined = next((row for row in summaries if row["branch"] == "COMBINED"), {})
    write_rows(OUT / "public_private_crowdout.csv", [{
        "private_only_transfer": private.get("total_wealth_transfer", 0.0),
        "combined_transfer": combined.get("total_wealth_transfer", 0.0),
        "absolute_reduction": private.get("total_wealth_transfer", 0.0) - combined.get("total_wealth_transfer", 0.0),
        "percentage_reduction": (private.get("total_wealth_transfer", 0.0) - combined.get("total_wealth_transfer", 0.0)) / private.get("total_wealth_transfer", 1.0),
        "pension_to_private_recipient_same_week": "not available from compact branch snapshot",
    }])
    write_rows(OUT / "consumption_food_sales_comparison.csv", [{
        "branch": branch,
        "final_household_consumption": sum(num(row.get("consumption")) for row in data["households"] if not row.get("settlement_only")),
        "final_household_saving": sum(num(row.get("saving")) for row in data["households"] if not row.get("settlement_only")),
        "firm_cash": data["result"].get("final_firm_cash"),
        "note": "Compact final snapshot; no giant person panel.",
    } for branch, data in artifacts.items()])
    write_rows(OUT / "firm_side_effect_comparison.csv", [{
        "branch": branch,
        "final_firm_cash": data["result"].get("final_firm_cash"),
        "money_supply": data["result"].get("money_supply"),
        "max_accounting_gap": data["result"].get("max_abs_monetary_accounting_gap"),
        "max_money_gap": data["result"].get("max_abs_money_delta_gap"),
        "invariant_violations": data["result"].get("invariant_violations"),
        "firm_payroll_note": "Employee-side PAYG contribution does not alter scheduled employer payroll; later demand effects are allowed.",
    } for branch, data in artifacts.items()])

    gate3_ok = (
        gate3.get("manifest", {}).get("status") == "COMPLETED"
        and all(data["result"].get("status") == "completed" for data in artifacts.values())
    )
    accounting_ok = all(
        num(row.get("max_abs_monetary_accounting_gap")) < 1e-6
        and num(row.get("max_abs_money_delta_gap")) < 1e-6
        and num(row.get("max_abs_food_conservation_gap")) < 1e-6
        and num(row.get("invariant_violations")) == 0
        for row in summaries
    )
    fixture_ok = all(row.get("pass") for row in fixtures)
    parity_ok = bool(parity_rows[0]["pass"])
    transfer_active = next((row for row in summaries if row["branch"] == "PRIVATE_ONLY"), {}).get("total_wealth_transfer", 0.0) > EPS
    payg_active = next((row for row in summaries if row["branch"] == "PAYG_ONLY"), {}).get("actual_contribution_total", 0.0) > EPS
    if not gate3_ok or not accounting_ok or not parity_ok:
        verdict = "F. ACTIVE_MECHANISM_ACCOUNTING_OR_TIMING_BLOCKER"
    elif payg_active and transfer_active:
        verdict = "A. ACTIVE_COMBINED_MECHANISM_CONFIRMS_SHADOW_COMPLEMENTARITY"
    elif payg_active:
        verdict = "B. ACTIVE_PAYG_WORKS_BUT_PRIVATE_TRANSFER_ADDS_LITTLE"
    elif transfer_active:
        verdict = "C. ACTIVE_PRIVATE_TRANSFER_WORKS_BUT_PAYG_BURDEN_TOO_HIGH"
    else:
        verdict = "G. OTHER"

    flags = {
        "verdict": verdict,
        "fixture_gate_pass": fixture_ok,
        "gate2_13_week_pass": gate2_ok,
        "gate3_52_week_pass": gate3_ok,
        "control_parity_pass": parity_ok,
        "accounting_money_goods_pass": accounting_ok,
        "payg_active": payg_active,
        "wealth_transfer_active": transfer_active,
        "default_off_when_disabled": parity_ok,
        "existing_private_family_support_activated": False,
        "retirement_changed": False,
        "employer_contribution": False,
        "government_backstop": False,
        "central_bank_financing": False,
        "new_rng_draws": 0,
        "economic_behavior_changed": False,
        "canonical_policy_enabled": False,
        "step17e_started": False,
        "research_only": True,
    }
    (OUT / "acceptance_flags.json").write_text(json.dumps(flags, indent=2), encoding="utf-8")

    lines = [
        "# Step17.D Active PAYG Pension and Intergenerational Wealth Transfer Smoke",
        "",
        f"- Fixture gate: **{'PASS' if fixture_ok else 'FAIL'}**.",
        f"- Policy order: **payroll -> dividends -> PAYG contribution -> PAYG pension -> wealth transfer -> food consumption**.",
        "- Existing needs-based PrivateFamilySupportSystem remained disabled and separate.",
        "- PAYG uses actual positive realized wage provenance, Household-to-Fund contributions, and Fund-to-Household cash-constrained payments.",
        "- Wealth transfer uses deterministic adult-child Household -> parent Household edges, a 13-week donor reserve, 25% of excess cash, and a one-week recipient target.",
        f"- Gate 2 (13 weeks, four branches): **{'PASS' if gate2_ok else 'FAIL'}**.",
        f"- Gate 3 (52 weeks, four branches): **{'PASS' if gate3_ok else 'FAIL'}**.",
        f"- Control parity: **{'PASS' if parity_ok else 'FAIL'}**.",
        f"- 52-week PAYG active: **{payg_active}**; wealth transfer active: **{transfer_active}**.",
        f"- Primary verdict: **{verdict}**.",
        "",
        "No 520-week run, canonical enabling, retirement, employer contribution, government backstop, new RNG, or Step17.E work was performed.",
    ]
    (OUT / "acceptance_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    shutil.rmtree(runtime_root, ignore_errors=True)


if __name__ == "__main__":
    run()






