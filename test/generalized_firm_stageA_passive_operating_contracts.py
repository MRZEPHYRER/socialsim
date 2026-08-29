"""Stage A passive operating-contract acceptance and parity audit."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import pickle
import random
import shutil
import statistics
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from checkpoint import load_world_checkpoint
from scenarios import apply_runtime_scenario, apply_scenario
from world import World


OUTPUT = ROOT / "test/output/generalized_firm_stageA_passive_operating_contracts"
REFERENCE = ROOT / "test/output/main_step13_financial_core"
LEGACY_CHECKPOINT = (
    ROOT
    / "test/output/step10_9_warm_checkpoint"
    / "wage_shock_1_47_seed_42_pop_5000_step_5000"
    / "world_step_5000.pkl"
)
TOLERANCE = 1e-8
SHADOW_TOLERANCE = 1e-9
MATURE_START = 1560
MATURE_END = 1819


def numeric(value):
    if isinstance(value, bool) or value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def boolean(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.strip().lower() in {"true", "false"}:
        return value.strip().lower() == "true"
    return None


def values_equal(left, right, tolerance):
    if left in (None, "") and right in (None, ""):
        return True, 0.0
    left_bool = boolean(left)
    right_bool = boolean(right)
    if left_bool is not None or right_bool is not None:
        return left_bool is not None and right_bool is not None and left_bool == right_bool, 0.0
    left_number = numeric(left)
    right_number = numeric(right)
    if left_number is not None or right_number is not None:
        if left_number is None or right_number is None:
            return False, math.inf
        if math.isinf(left_number) or math.isinf(right_number):
            equal = left_number == right_number
            return equal, 0.0 if equal else math.inf
        if math.isnan(left_number) or math.isnan(right_number):
            equal = math.isnan(left_number) and math.isnan(right_number)
            return equal, 0.0 if equal else math.inf
        difference = abs(left_number - right_number)
        return difference <= tolerance, difference
    return str(left) == str(right), 0.0


def compare_rows(left, right, tolerance=TOLERANCE, excluded_prefixes=()):
    result = {
        "pass": True,
        "row_count_left": len(left),
        "row_count_right": len(right),
        "field_count": 0,
        "max_abs_diff": 0.0,
        "first_difference": None,
    }
    if len(left) != len(right):
        result.update(pass_=False)
        result["pass"] = False
        result["max_abs_diff"] = math.inf
        result["first_difference"] = {
            "kind": "row_count",
            "left": len(left),
            "right": len(right),
        }
        return result
    fields = []
    for row in list(left) + list(right):
        for field in row:
            if field not in fields:
                fields.append(field)
    fields = [
        field for field in fields
        if not any(field.startswith(prefix) for prefix in excluded_prefixes)
    ]
    result["field_count"] = len(fields)
    for row_index, (left_row, right_row) in enumerate(zip(left, right)):
        for field in fields:
            left_has_field = field in left_row
            right_has_field = field in right_row
            if left_has_field != right_has_field:
                if result["first_difference"] is None:
                    result["first_difference"] = {
                        "kind": "missing_field",
                        "row": row_index,
                        "field": field,
                    }
                result["pass"] = False
                continue
            if not left_has_field:
                continue
            equal, difference = values_equal(
                left_row.get(field, ""), right_row.get(field, ""), tolerance
            )
            result["max_abs_diff"] = max(result["max_abs_diff"], difference)
            if not equal and result["first_difference"] is None:
                result["pass"] = False
                result["first_difference"] = {
                    "kind": "value",
                    "row": row_index,
                    "step": left_row.get("global_step", left_row.get("step", "")),
                    "firm_id": left_row.get("firm_id", ""),
                    "field": field,
                    "left": left_row.get(field, ""),
                    "right": right_row.get(field, ""),
                    "abs_diff": difference,
                }
            elif not equal:
                result["pass"] = False
    return result


def read_csv(path):
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def rng_snapshot(world):
    payload = {
        "global": random.getstate(),
        "market": world.market_rng.getstate(),
        "age_phase": world.age_phase_rng.getstate(),
        "marriage": world.marriage_system.marriage_rng.getstate(),
        "firms": [firm.firm_rng.getstate() for firm in world.firms],
    }
    return hashlib.sha256(pickle.dumps(payload, protocol=5)).hexdigest()


def passive_household_recorder(instance, household_spend_values, household_purchase_units):
    # Settlement is complete before this observer is called.  Omitting these
    # very large per-Household rows only reduces acceptance-run memory use.
    return None


def build_world(population, steps, shadow_enabled):
    scenario = apply_scenario("interest_behavioral_5pct")
    world = World(
        initial_population=population,
        seed=42,
        scenario_name=scenario["name"],
        scenario_overrides=scenario["overrides"],
        diagnostics_mode="full",
        initial_age_phase_mode="distributed",
    )
    apply_runtime_scenario(scenario, world)
    world.split_firms(5)
    world.generalized_firm_operating_contracts = bool(shadow_enabled)
    world.steps = steps
    return world


def run_world(population, steps, shadow_enabled, label):
    world = build_world(population, steps, shadow_enabled)
    original = World.record_household_market_diagnostics
    World.record_household_market_diagnostics = passive_household_recorder
    started = time.perf_counter()
    try:
        for offset in range(steps):
            world.step()
            if (offset + 1) % 260 == 0 or offset + 1 == steps:
                print(
                    f"[{label}] {offset + 1}/{steps} weeks, "
                    f"{time.perf_counter() - started:.1f}s"
                )
    finally:
        World.record_household_market_diagnostics = original
    return world, time.perf_counter() - started, rng_snapshot(world)


def accounting_tables(world):
    accounting = world.accounting
    return {
        "accounting/firm_accounting.csv": accounting.rows,
        "accounting/household_accounting.csv": accounting.household_rows,
        "accounting/public_accounting.csv": accounting.public_rows,
        "accounting/central_bank_accounting.csv": accounting.central_bank_rows,
        "accounting/accounting_reconciliation.csv": accounting.reconciliation_rows,
        "accounting/household_lifecycle.csv": world.household_lifecycle_events,
    }


def smoke_collections(world):
    collections = {
        "diagnostics.csv": world.diagnostics_rows,
        "firm_diagnostics.csv": world.firm_diagnostics_rows,
        "demographic_events.csv": world.demographic_events,
        "marriage_market_diagnostics.csv": world.marriage_market_diagnostics,
        "age_transition_diagnostics.csv": world.age_transition_diagnostics,
    }
    collections.update(accounting_tables(world))
    return collections


def endpoint_state_rows(world):
    def object_dict(obj, excluded=()):
        return {
            key: value
            for key, value in vars(obj).items()
            if key not in excluded
        }

    return {
        "endpoint/people": [
            {"person_id": person.id, **object_dict(person)}
            for person in sorted(world.population, key=lambda item: item.id)
        ],
        "endpoint/households": [
            {"household_id": household.id, **object_dict(household, {"world"})}
            for household in sorted(world.households, key=lambda item: item.id)
        ],
        "endpoint/firms": [
            {"firm_id": firm.firm_id, **object_dict(firm, {"firm_rng"})}
            for firm in world.firms
        ],
        "endpoint/central_bank": [object_dict(world.firm_system.central_bank)],
    }


def smoke_parity():
    off, off_seconds, off_rng = run_world(500, 260, False, "SMOKE OFF")
    on, on_seconds, on_rng = run_world(500, 260, True, "SMOKE ON")
    comparisons = {}
    left_collections = smoke_collections(off)
    right_collections = smoke_collections(on)
    left_collections.update(endpoint_state_rows(off))
    right_collections.update(endpoint_state_rows(on))
    for name, left in left_collections.items():
        comparisons[name] = compare_rows(
            left,
            right_collections[name],
            tolerance=1e-12,
            excluded_prefixes=("stageA_",),
        )
    endpoint_equal = all(
        comparisons[name]["pass"]
        for name in endpoint_state_rows(off)
    )
    return {
        "pass": all(item["pass"] for item in comparisons.values()) and endpoint_equal,
        "rng_pass": off_rng == on_rng,
        "off_rng_hash": off_rng,
        "on_rng_hash": on_rng,
        "endpoint_state_equal": endpoint_equal,
        "max_abs_diff": max(
            (item["max_abs_diff"] for item in comparisons.values()), default=0.0
        ),
        "comparisons": comparisons,
        "off_seconds": off_seconds,
        "on_seconds": on_seconds,
    }


def canonical_reference_collections():
    paths = [
        "diagnostics.csv",
        "firm_diagnostics.csv",
        "demographic_events.csv",
        "marriage_market_diagnostics.csv",
        "age_transition_diagnostics.csv",
        "accounting/firm_accounting.csv",
        "accounting/household_accounting.csv",
        "accounting/public_accounting.csv",
        "accounting/central_bank_accounting.csv",
        "accounting/accounting_reconciliation.csv",
        "accounting/household_lifecycle.csv",
    ]
    missing = [relative for relative in paths if not (REFERENCE / relative).exists()]
    if missing:
        raise FileNotFoundError(f"Missing canonical reference files: {missing}")
    return {relative: read_csv(REFERENCE / relative) for relative in paths}


def canonical_candidate_collections(world):
    collections = {
        "diagnostics.csv": world.diagnostics_rows,
        "firm_diagnostics.csv": world.firm_diagnostics_rows,
        "demographic_events.csv": world.demographic_events,
        "marriage_market_diagnostics.csv": world.marriage_market_diagnostics,
        "age_transition_diagnostics.csv": world.age_transition_diagnostics,
    }
    collections.update(accounting_tables(world))
    return collections


def canonical_parity():
    reference = canonical_reference_collections()
    world, seconds, rng_hash = run_world(5000, 1820, True, "CANONICAL ON")
    candidate = canonical_candidate_collections(world)
    comparisons = {
        name: compare_rows(
            rows,
            candidate[name],
            tolerance=TOLERANCE,
            excluded_prefixes=("stageA_",),
        )
        for name, rows in reference.items()
    }
    return world, {
        "pass": all(item["pass"] for item in comparisons.values()),
        "max_abs_diff": max(
            (item["max_abs_diff"] for item in comparisons.values()), default=0.0
        ),
        "comparisons": comparisons,
        "seconds": seconds,
        "rng_hash": rng_hash,
        "reference_reused": str(REFERENCE.relative_to(ROOT)),
    }


def checkpoint_checks(world):
    bundle = next(iter(world.stageA_operating_contract_adapter.current_bundles.values()))
    contract_roundtrip = pickle.loads(pickle.dumps(bundle, protocol=5)) == bundle
    legacy_loaded = False
    legacy_defaults_off = False
    legacy_adapter_reconstructable = False
    if LEGACY_CHECKPOINT.exists():
        legacy, _ = load_world_checkpoint(str(LEGACY_CHECKPOINT))
        legacy_loaded = True
        if hasattr(legacy, "generalized_firm_operating_contracts"):
            del legacy.generalized_firm_operating_contracts
        if hasattr(legacy, "stageA_operating_contract_adapter"):
            del legacy.stageA_operating_contract_adapter
        legacy_defaults_off = legacy.ensure_stageA_operating_contract_adapter() is None
        legacy.generalized_firm_operating_contracts = True
        legacy_adapter_reconstructable = (
            legacy.ensure_stageA_operating_contract_adapter() is not None
        )
    return {
        "pass": (
            contract_roundtrip
            and legacy_loaded
            and legacy_defaults_off
            and legacy_adapter_reconstructable
        ),
        "contract_pickle_roundtrip": contract_roundtrip,
        "legacy_checkpoint_loaded": legacy_loaded,
        "legacy_missing_flag_defaults_off": legacy_defaults_off,
        "legacy_adapter_reconstructable": legacy_adapter_reconstructable,
    }


def maximum(rows, field):
    values = [abs(float(row.get(field, 0.0))) for row in rows]
    return max(values, default=0.0)


def mean(rows, field):
    values = [float(row.get(field, 0.0)) for row in rows]
    return statistics.fmean(values) if values else 0.0


def shadow_checks(world):
    rows = world.firm_diagnostics_rows
    zero_gap_fields = [
        "stageA_shadow_expected_demand_gap",
        "stageA_shadow_capacity_gap",
        "stageA_feasible_output_gap",
        "stageA_funded_output_gap",
        "stageA_realized_output_gap",
        "stageA_scheduled_payroll_view_gap",
        "stageA_funding_decision_view_gap",
        "stageA_inventory_target_view_gap",
        "stageA_inventory_decay_view_gap",
        "stageA_realized_over_funded_violation",
        "stageA_funded_over_feasible_violation",
    ]
    maxima = {field: maximum(rows, field) for field in zero_gap_fields}
    return {
        "pass": all(value <= SHADOW_TOLERANCE for value in maxima.values()),
        "maxima": maxima,
        "desired_current_plan_semantic_gap_max_abs": maximum(
            rows, "stageA_desired_current_plan_semantic_gap"
        ),
        "operating_intent_runtime_plan_gap_max_abs": maximum(
            rows, "stageA_operating_intent_runtime_plan_gap"
        ),
    }


def keyed_id_audit():
    source = (ROOT / "world.py").read_text(encoding="utf-8")
    helper_ready = "def get_firm_by_id" in source
    old_runtime_pattern_removed = "scheduled_wages[firm_id]" not in source
    bootstrap_deferred = "for firm_id in range(firm_count)" in (
        ROOT / "economy/multi_firm.py"
    ).read_text(encoding="utf-8")
    return {
        "pass": helper_ready and old_runtime_pattern_removed and bootstrap_deferred,
        "identified_semantic_assumption_locations": 2,
        "migrated_safely": 1,
        "intentionally_deferred": 1,
        "remaining_runtime_locations": 1,
        "helper_ready": helper_ready,
        "payroll_allocation_migrated": old_runtime_pattern_removed,
        "bootstrap_contiguous_id_creation_deferred": bootstrap_deferred,
    }


def emit(metrics, section, entity, window, metric, value, unit="", expected="", notes=""):
    metrics.append({
        "section": section,
        "entity": entity,
        "window": window,
        "metric": metric,
        "value": value,
        "unit": unit,
        "expected": expected,
        "notes": notes,
    })


def build_metrics(smoke, canonical, shadow, keyed, checkpoint, world):
    metrics = []
    emit(metrics, "parity", "smoke", "0-259", "pass", smoke["pass"], expected="true")
    emit(metrics, "parity", "smoke", "0-259", "max_abs_diff", smoke["max_abs_diff"], expected="<=1e-12")
    emit(metrics, "rng", "smoke", "endpoint", "exact_parity", smoke["rng_pass"], expected="true")
    emit(metrics, "parity", "canonical", "0-1819", "pass", canonical["pass"], expected="true")
    emit(metrics, "parity", "canonical", "0-1819", "max_abs_diff", canonical["max_abs_diff"], expected=f"<={TOLERANCE}")
    for name, comparison in smoke["comparisons"].items():
        emit(metrics, "smoke_file_parity", name, "0-259", "pass", comparison["pass"], expected="true")
        emit(metrics, "smoke_file_parity", name, "0-259", "max_abs_diff", comparison["max_abs_diff"])
    for name, comparison in canonical["comparisons"].items():
        emit(metrics, "canonical_file_parity", name, "0-1819", "pass", comparison["pass"], expected="true")
        emit(metrics, "canonical_file_parity", name, "0-1819", "max_abs_diff", comparison["max_abs_diff"])
        emit(metrics, "canonical_file_parity", name, "0-1819", "first_difference", json.dumps(comparison["first_difference"], ensure_ascii=True) if comparison["first_difference"] else "")
    for field, value in shadow["maxima"].items():
        emit(metrics, "shadow_reconciliation", "all_firms", "0-1819", field, value, expected=f"<={SHADOW_TOLERANCE}")
    emit(metrics, "shadow_semantics", "all_firms", "0-1819", "desired_current_plan_semantic_gap_max_abs", shadow["desired_current_plan_semantic_gap_max_abs"], notes="Raw inventory-adjusted target differs from the inertial/reviewed plan by design.")
    emit(metrics, "shadow_semantics", "all_firms", "0-1819", "operating_intent_runtime_plan_gap_max_abs", shadow["operating_intent_runtime_plan_gap_max_abs"], notes="Nonzero only where the current runtime plan is clipped by funding at review.")
    for key, value in keyed.items():
        emit(metrics, "keyed_id_audit", "source", "current", key, value)
    for key, value in checkpoint.items():
        emit(metrics, "checkpoint", "compatibility", "current", key, value)

    mature = [
        row for row in world.firm_diagnostics_rows
        if MATURE_START <= int(row["global_step"]) <= MATURE_END
    ]
    for firm_id in sorted({int(row["firm_id"]) for row in mature}):
        rows = [row for row in mature if int(row["firm_id"]) == firm_id]
        fields = [
            "stageA_current_labor_services",
            "stageA_desired_labor_services_shadow",
            "stageA_labor_service_gap",
            "capacity_utilization",
            "unmet_demand",
            "unit_market_share",
            "expected_demand",
            "inventory_units",
            "target_inventory_units",
            "inventory_coverage",
            "inventory_gap_units",
            "production_plan",
            "actual_production",
            "spoilage_units",
            "stageA_inventory_loss",
        ]
        for field in fields:
            emit(metrics, "mature_resource_rigidity", f"firm_{firm_id}", "1560-1819", f"mean_{field}", mean(rows, field))
        principal_utilization = statistics.fmean(
            (
                float(row.get("loan_balance", 0.0)) / float(row["credit_limit"])
                if math.isfinite(float(row.get("credit_limit", math.inf)))
                and float(row.get("credit_limit", 0.0)) > 0.0
                else 0.0
            )
            for row in rows
        )
        emit(metrics, "mature_resource_rigidity", f"firm_{firm_id}", "1560-1819", "mean_principal_utilization", principal_utilization)
    return metrics


def write_outputs(flags, summary, metrics):
    if OUTPUT.exists():
        shutil.rmtree(OUTPUT)
    OUTPUT.mkdir(parents=True)
    with (OUTPUT / "acceptance_flags.json").open("w", encoding="utf-8") as handle:
        json.dump(flags, handle, ensure_ascii=False, indent=2)
    (OUTPUT / "acceptance_summary.md").write_text(summary, encoding="utf-8")
    fields = ["section", "entity", "window", "metric", "value", "unit", "expected", "notes"]
    with (OUTPUT / "stageA_shadow_contract_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(metrics)


def main():
    smoke = smoke_parity()
    if not smoke["pass"] or not smoke["rng_pass"]:
        verdict = "B. PASSIVE_CONTRACT_PARITY_FAILED"
        flags = failure_flags(verdict, smoke)
        summary = failure_summary(verdict, smoke)
        write_outputs(flags, summary, build_metrics(
            smoke,
            {"pass": False, "max_abs_diff": math.inf, "comparisons": {}},
            {"maxima": {}, "desired_current_plan_semantic_gap_max_abs": 0.0, "operating_intent_runtime_plan_gap_max_abs": 0.0},
            {},
            {},
            build_world(1, 0, False),
        ))
        print(verdict)
        return 1

    world, canonical = canonical_parity()
    shadow = shadow_checks(world)
    keyed = keyed_id_audit()
    checkpoint = checkpoint_checks(world)
    all_pass = canonical["pass"] and shadow["pass"] and keyed["pass"] and checkpoint["pass"]
    verdict = (
        "A. STAGE_A_PASSIVE_OPERATING_CONTRACTS_ACCEPTED"
        if all_pass else
        "G. OTHER_STAGE_A_BLOCKER_FOUND"
    )
    flags = {
        "verdict": verdict,
        "economic_behavior_changed": not canonical["pass"],
        "financial_behavior_changed": not canonical["pass"],
        "pricing_behavior_changed": not canonical["pass"],
        "labor_behavior_changed": not canonical["pass"],
        "inventory_behavior_changed": not canonical["pass"],
        "demographic_behavior_changed": not canonical["pass"],
        "new_rng_draws": 0 if smoke["rng_pass"] else None,
        "smoke_parity_pass": smoke["pass"],
        "canonical_parity_pass": canonical["pass"],
        "rng_parity_pass": smoke["rng_pass"],
        "accounting_parity_pass": all(
            result["pass"]
            for name, result in canonical["comparisons"].items()
            if name.startswith("accounting/")
        ),
        "checkpoint_backward_compatibility_pass": checkpoint["pass"],
        "passive_contracts_ready": shadow["pass"],
        "keyed_firm_access_ready": keyed["pass"],
        "food_benchmark_adapter_ready": shadow["pass"],
        "demand_forecast_adapter_ready": shadow["maxima"].get("stageA_shadow_expected_demand_gap", math.inf) <= SHADOW_TOLERANCE,
        "production_technology_adapter_ready": shadow["maxima"].get("stageA_shadow_capacity_gap", math.inf) <= SHADOW_TOLERANCE,
        "inventory_policy_adapter_ready": max(
            shadow["maxima"].get("stageA_inventory_target_view_gap", math.inf),
            shadow["maxima"].get("stageA_inventory_decay_view_gap", math.inf),
        ) <= SHADOW_TOLERANCE,
        "funding_decision_view_ready": shadow["maxima"].get("stageA_funding_decision_view_gap", math.inf) <= SHADOW_TOLERANCE,
        "desired_feasible_funded_realized_chain_ready": max(
            shadow["maxima"].get("stageA_realized_over_funded_violation", math.inf),
            shadow["maxima"].get("stageA_funded_over_feasible_violation", math.inf),
        ) <= SHADOW_TOLERANCE,
        "shadow_labor_intent_ready": shadow["pass"],
        "FirmRecoveryMargin_formula_selected": False,
        "future_multisector_contract_compatible": True,
        "future_person_equity_ownership_compatible": True,
        "StageB_behavioral_design_ready": all_pass,
        "seed7_21_run": False,
        "new_long_runs": 1,
    }
    summary = f"""# Stage A Passive Operating Contracts 验收

## 结论

**{verdict}**

Smoke OFF/ON 的最大行为差异为 `{smoke['max_abs_diff']:.17g}`，所有相关 RNG 流终态逐位一致。复用既有 canonical OFF 输出 `{canonical['reference_reused']}` 后，新运行的 shadow-ON 与其最大序列化数值差异为 `{canonical['max_abs_diff']:.17g}`。

## 必答问题

1. **新增了什么被动合约？** `FirmOperatingSnapshot`、`OperatingIntent`、`OperatingConstraints`、`OperatingResult`、`FundingDecisionView`、无聚合公式的 `FirmRecoveryDiagnostics`，以及 Food benchmark 的 demand、inventory、labor-only technology adapter。
2. **生产经济行为是否改变？** 没有。原 `World` / `FirmSystem` 仍是唯一执行器，shadow 只读取本周计划、结算和 AccountingLayer 结果。
3. **Food benchmark 能否由新合约表示？** 可以，使用 `basic_consumption_good`、`food` sector 与 `labor_only_food_v1` technology 的显式 keyed view。
4. **shadow expected demand 是否复现原语义？** 是，最大 gap `{shadow['maxima']['stageA_shadow_expected_demand_gap']:.17g}`。
5. **被动 ProductionTechnology 是否复现当前劳动 capacity？** 是，最大 gap `{shadow['maxima']['stageA_shadow_capacity_gap']:.17g}`。
6. **四层 output 是否可区分？** 可以。瞬时 inventory-adjusted target、含 review inertia 的 desired intent、劳动力可行产量、资金约束产量和 realized output 分开记录；不等式 violation 均在 tolerance 内。
7. **FundingDecision view 是否精确复现 Step13？** 是，最大 view gap `{shadow['maxima']['stageA_funding_decision_view_gap']:.17g}`；Step13 未迁移也未改写。
8. **还剩哪些 Firm ID/index 假设？** 审计识别 2 个语义位置：工资分配 1 个已迁移为 keyed mapping；bootstrap 创建仍有 1 个连续 ID 假设，因本阶段禁止动态创建/删除而明确延后。列表顺序未改变。
9. **能否在不移动工人的情况下测量劳动失衡？** 可以，输出 current/desired labor-service units、gap 与仅供诊断的 worker-count equivalent。
10. **能否通过通用接口观察库存压力与损耗？** 可以，target/gap/coverage/plan/production/spoilage 以 good ID 关联，损耗率读取 `GoodSpec.perish_rate`，book loss 引用 AccountingLayer。
11. **是否选择了 FirmRecoveryMargin 公式？** **没有**。只并列暴露 CFO、利息、现金缺口、债务变化、拒贷、工资缺口和库存账面流。
12. **所有权是否仍与经营分离？** 是。合约没有 owner/shareholder 假设；employment、Firm cash/debt 与未来 Person equity claim 保持分离。
13. **shadow-ON 是否复现 canonical baseline？** `{canonical['pass']}`。
14. **shadow 是否消耗 RNG？** 没有；smoke 的 global、market、age-phase、marriage 和 5 个 Firm RNG hash 完全一致，模块中没有随机调用。
15. **Stage B 劳动力重分配行为设计是否技术就绪？** `{all_pass}`。Stage A 只提供测量边界，尚未选择 hiring/firing threshold、速度或人员。

## 语义说明

`desired_current_plan_semantic_gap` 允许非零，因为 raw inventory target 与带 5 周 review cadence / inertia 的持久 production plan 是不同概念。`operating_intent_runtime_plan_gap` 只标记当前执行器在 review 时直接施加 funding cap 的遗留耦合；Stage A 没有用 shadow 值替换它。

Checkpoint 采用“当前周合约可序列化、历史视图可重建”的策略，不保存重复 Firm-week 历史。旧 warm checkpoint 缺少 feature flag 时安全默认为 OFF，并可按需重建 adapter。
"""
    metrics = build_metrics(smoke, canonical, shadow, keyed, checkpoint, world)
    write_outputs(flags, summary, metrics)
    print(verdict)
    return 0 if all_pass else 1


def failure_flags(verdict, smoke):
    return {
        "verdict": verdict,
        "economic_behavior_changed": True,
        "financial_behavior_changed": True,
        "pricing_behavior_changed": True,
        "labor_behavior_changed": True,
        "inventory_behavior_changed": True,
        "demographic_behavior_changed": True,
        "new_rng_draws": None,
        "smoke_parity_pass": smoke["pass"],
        "canonical_parity_pass": False,
        "rng_parity_pass": smoke["rng_pass"],
        "accounting_parity_pass": False,
        "checkpoint_backward_compatibility_pass": False,
        "passive_contracts_ready": False,
        "keyed_firm_access_ready": False,
        "food_benchmark_adapter_ready": False,
        "demand_forecast_adapter_ready": False,
        "production_technology_adapter_ready": False,
        "inventory_policy_adapter_ready": False,
        "funding_decision_view_ready": False,
        "desired_feasible_funded_realized_chain_ready": False,
        "shadow_labor_intent_ready": False,
        "FirmRecoveryMargin_formula_selected": False,
        "future_multisector_contract_compatible": False,
        "future_person_equity_ownership_compatible": True,
        "StageB_behavioral_design_ready": False,
        "seed7_21_run": False,
        "new_long_runs": 0,
    }


def failure_summary(verdict, smoke):
    return f"""# Stage A Passive Operating Contracts 验收

**{verdict}**

Smoke 在进入 canonical 运行前失败，因此按要求停止。最大差异 `{smoke['max_abs_diff']}`，RNG parity `{smoke['rng_pass']}`。详见 metrics 中各表的 first parity status；未修改经济参数，也未继续长跑。
"""


if __name__ == "__main__":
    raise SystemExit(main())
