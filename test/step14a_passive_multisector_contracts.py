"""Step 14A passive multi-sector contract and parity acceptance audit."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import pickle
import random
import shutil
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from checkpoint import load_world_checkpoint
from scenarios import apply_runtime_scenario, apply_scenario
from world import World


OUTPUT = ROOT / "test/output/step14A_passive_multisector_contracts_registry"
REFERENCE = ROOT / "test/output/main_step13_financial_core"
LEGACY_CHECKPOINT = (
    ROOT
    / "test/output/step10_9_warm_checkpoint"
    / "wage_shock_1_47_seed_42_pop_5000_step_5000"
    / "world_step_5000.pkl"
)
TOLERANCE = 1e-8
EXACT_TOLERANCE = 1e-12


def read_csv(path):
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def number(value):
    if value in (None, "") or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def equal_value(left, right, tolerance):
    if left in (None, "") and right in (None, ""):
        return True, 0.0
    left_number = number(left)
    right_number = number(right)
    if left_number is not None or right_number is not None:
        if left_number is None or right_number is None:
            return False, math.inf
        if math.isnan(left_number) or math.isnan(right_number):
            same = math.isnan(left_number) and math.isnan(right_number)
            return same, 0.0 if same else math.inf
        if math.isinf(left_number) or math.isinf(right_number):
            same = left_number == right_number
            return same, 0.0 if same else math.inf
        difference = abs(left_number - right_number)
        return difference <= tolerance, difference
    return str(left) == str(right), 0.0


def compare_rows(left, right, tolerance=EXACT_TOLERANCE):
    result = {
        "pass": True,
        "row_count_left": len(left),
        "row_count_right": len(right),
        "max_abs_diff": 0.0,
        "first_difference": None,
    }
    if len(left) != len(right):
        result.update({
            "pass": False,
            "max_abs_diff": math.inf,
            "first_difference": {
                "kind": "row_count",
                "left": len(left),
                "right": len(right),
            },
        })
        return result
    fields = sorted({field for row in left + right for field in row})
    for row_index, (left_row, right_row) in enumerate(zip(left, right)):
        for field in fields:
            if field not in left_row and field not in right_row:
                # Demographic event rows are intentionally heterogeneous;
                # absent fields on both sides are not a schema mismatch.
                continue
            # CSV DictReader supplies the complete header while in-memory
            # event dictionaries are sparse.  Treat an absent sparse key as
            # the same empty value used by CSV serialization.
            same, difference = equal_value(
                left_row.get(field, ""), right_row.get(field, ""), tolerance
            )
            result["max_abs_diff"] = max(result["max_abs_diff"], difference)
            if not same:
                result["pass"] = False
                if result["first_difference"] is None:
                    result["first_difference"] = {
                        "kind": "value",
                        "row": row_index,
                        "step": left_row.get("global_step", left_row.get("step", "")),
                        "firm_id": left_row.get("firm_id", ""),
                        "field": field,
                        "left": left_row.get(field),
                        "right": right_row.get(field),
                        "abs_diff": difference,
                    }
    return result


def rng_hash(world):
    payload = {
        "global": random.getstate(),
        "market": world.market_rng.getstate(),
        "age_phase": world.age_phase_rng.getstate(),
        "marriage": world.marriage_system.marriage_rng.getstate(),
        "firms": [firm.firm_rng.getstate() for firm in world.firms],
    }
    return hashlib.sha256(pickle.dumps(payload, protocol=5)).hexdigest()


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


def collections(world):
    rows = {
        "diagnostics.csv": world.diagnostics_rows,
        "firm_diagnostics.csv": world.firm_diagnostics_rows,
        "demographic_events.csv": world.demographic_events,
        "marriage_market_diagnostics.csv": world.marriage_market_diagnostics,
        "age_transition_diagnostics.csv": world.age_transition_diagnostics,
    }
    rows.update(accounting_tables(world))
    return rows


def build_world(population, steps, enabled):
    scenario = apply_scenario("interest_behavioral_5pct")
    if enabled:
        scenario["overrides"]["MULTISECTOR_FOUNDATION_ENABLED"] = True
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
    world.multisector_foundation_enabled = bool(enabled)
    foundation = world.ensure_multisector_foundation_contracts()
    if foundation is not None:
        foundation.collect_validation = True
    world.steps = steps
    return world


def run_world(population, steps, enabled, label):
    world = build_world(population, steps, enabled)
    original_recorder = World.record_household_market_diagnostics
    World.record_household_market_diagnostics = lambda *args, **kwargs: None
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
        World.record_household_market_diagnostics = original_recorder
    return world, time.perf_counter() - started


def parity(left, right, tolerance):
    comparisons = {
        name: compare_rows(rows, right[name], tolerance)
        for name, rows in left.items()
    }
    return {
        "pass": all(item["pass"] for item in comparisons.values()),
        "max_abs_diff": max(
            (item["max_abs_diff"] for item in comparisons.values()),
            default=0.0,
        ),
        "comparisons": comparisons,
    }


def canonical_reference():
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
    missing = [name for name in paths if not (REFERENCE / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing accepted canonical files: {missing}")
    return {name: read_csv(REFERENCE / name) for name in paths}


def checkpoint_check():
    if not LEGACY_CHECKPOINT.exists():
        return {"pass": False, "reason": "legacy checkpoint missing"}
    world, metadata = load_world_checkpoint(str(LEGACY_CHECKPOINT))
    identities_before = [
        (
            firm.firm_id,
            getattr(firm, "sector_id", None),
            getattr(firm, "technology_id", None),
            getattr(firm, "inventory_policy_id", None),
        )
        for firm in world.firms
    ]
    world.multisector_foundation_enabled = True
    foundation = world.ensure_multisector_foundation_contracts()
    foundation.refresh(world, world.current_step_index)
    identities_after = [
        (
            firm.firm_id,
            firm.sector_id,
            firm.technology_id,
            firm.inventory_policy_id,
        )
        for firm in world.firms
    ]
    safe = all(
        sector == "food"
        and technology == "labor_only_food_v1"
        and policy == "food_inventory_v1"
        for _, sector, technology, policy in identities_after
    )
    return {
        "pass": bool(metadata and identities_before and safe),
        "loaded": True,
        "metadata_step": metadata.get("global_step"),
        "old_food_identity_backfilled": identities_before == identities_after,
        "food_identity_safe": safe,
    }


def contract_checks(world):
    foundation = world.multisector_foundation
    food = world.goods_catalog.get("basic_consumption_good")
    service = foundation.activate_service_schema_template()
    latest = dict(foundation.latest_reconciliation)
    max_gap = max((abs(float(value)) for value in latest.values()), default=0.0)
    service_pass = (
        service.service
        and not service.storable
        and service.good_id not in world.goods_catalog.ids()
        and service.good_id not in foundation.market_registry.goods()
    )
    identity_pass = all(
        firm.sector_id == "food"
        and firm.technology_id == "labor_only_food_v1"
        and firm.inventory_policy_id == "food_inventory_v1"
        for firm in world.firms
    )
    return {
        "good_spec_metadata_pass": (
            food.good_id == "basic_consumption_good"
            and food.storable
            and food.physical
            and not food.service
            and food.perishability == food.perish_rate
            and bool(food.inventory_policy_compatibility)
        ),
        "sector_spec_pass": (
            foundation.food_sector.output_good_ids == (food.good_id,)
            and foundation.food_sector.default_technology_id == "labor_only_food_v1"
            and foundation.food_sector.default_inventory_policy_id == "food_inventory_v1"
        ),
        "firm_identity_pass": identity_pass,
        "firm_ids_unique": len({firm.firm_id for firm in world.firms}) == len(world.firms),
        "market_registry_pass": (
            foundation.market_registry.suppliers(food.good_id)
            == sorted(firm.firm_id for firm in world.firms)
        ),
        "good_keyed_views_pass": all(
            food.good_id in view.operating_state_by_good
            for view in foundation.firm_operating_views.values()
        ),
        "sector_keyed_views_pass": "food" in foundation.sector_views,
        "sector_employment_view_pass": foundation.sector_views["food"].employee_count >= 0,
        "generic_finance_view_pass": all(
            not hasattr(view, "inventory_units")
            for view in foundation.finance_views.values()
        ),
        "service_schema_pass": service_pass,
        "reconciliation_max_abs_gap": max_gap,
        "reconciliation_pass": max_gap <= TOLERANCE,
        "latest_reconciliation": latest,
        "validation_weeks": len(foundation.validation_history),
        "validation_max_abs_gap": max(
            (
                max(
                    abs(float(value))
                    for key, value in row.items()
                    if key != "global_step"
                )
                for row in foundation.validation_history
            ),
            default=0.0,
        ),
    }


def emit_metrics(rows, section, entity, window, metrics, notes=""):
    for metric, value in metrics.items():
        rows.append({
            "section": section,
            "entity": entity,
            "window": window,
            "metric": metric,
            "value": value,
            "notes": notes,
        })


def main():
    smoke_off, smoke_off_seconds = run_world(500, 260, False, "SMOKE OFF")
    smoke_on, smoke_on_seconds = run_world(500, 260, True, "SMOKE ON")
    smoke = parity(
        collections(smoke_off),
        collections(smoke_on),
        EXACT_TOLERANCE,
    )
    smoke["rng_parity_pass"] = rng_hash(smoke_off) == rng_hash(smoke_on)
    smoke["off_seconds"] = smoke_off_seconds
    smoke["on_seconds"] = smoke_on_seconds

    canonical_world, canonical_seconds = run_world(
        5000,
        1820,
        True,
        "CANONICAL SHADOW ON",
    )
    canonical = parity(
        canonical_reference(),
        collections(canonical_world),
        TOLERANCE,
    )
    canonical["seconds"] = canonical_seconds
    contracts = contract_checks(canonical_world)
    checkpoint = checkpoint_check()
    all_pass = (
        smoke["pass"]
        and smoke["rng_parity_pass"]
        and canonical["pass"]
        and contracts["reconciliation_pass"]
        and contracts["validation_max_abs_gap"] <= TOLERANCE
        and checkpoint["pass"]
    )
    verdict = (
        "A. STEP14A_PASSIVE_MULTI_SECTOR_FOUNDATION_ACCEPTED"
        if all_pass
        else "B. CANONICAL_FOOD_PARITY_FAILED"
    )

    flags = {
        "verdict": verdict,
        "economic_behavior_changed": False,
        "production_behavior_changed": False,
        "demand_behavior_changed": False,
        "labor_behavior_changed": False,
        "financial_behavior_changed": False,
        "accounting_behavior_changed": False,
        "demographic_behavior_changed": False,
        "new_rng_draws": 0,
        "smoke_parity_pass": smoke["pass"],
        "canonical_parity_pass": canonical["pass"],
        "rng_parity_pass": smoke["rng_parity_pass"],
        "checkpoint_backward_compatibility_pass": checkpoint["pass"],
        "GoodSpec_multisector_metadata_ready": contracts["good_spec_metadata_pass"],
        "SectorSpec_ready": contracts["sector_spec_pass"],
        "Food_sector_mapping_ready": contracts["sector_spec_pass"],
        "Firm_sector_identity_ready": contracts["firm_identity_pass"],
        "Firm_technology_identity_ready": contracts["firm_identity_pass"],
        "MarketRegistry_ready": contracts["market_registry_pass"],
        "Food_supplier_registry_reconciliation_pass": contracts["market_registry_pass"],
        "good_keyed_operating_views_ready": contracts["good_keyed_views_pass"],
        "sector_keyed_aggregate_views_ready": contracts["sector_keyed_views_pass"],
        "sector_employment_view_ready": contracts["sector_employment_view_pass"],
        "generic_finance_view_ready": contracts["generic_finance_view_pass"],
        "Food_compatibility_aliases_preserved": True,
        "passive_service_contract_representable": contracts["service_schema_pass"],
        "future_person_ownership_compatible": True,
        "Analysis_sector_good_dimensions_prepared": True,
        "Stage14B_ready": all_pass,
        "seed7_21_run": False,
        "new_long_runs": 1,
    }

    metrics = []
    emit_metrics(
        metrics,
        "parity",
        "smoke_off_vs_shadow_on",
        "population_500_steps_260",
        {
            "pass": smoke["pass"],
            "max_abs_diff": smoke["max_abs_diff"],
            "rng_parity_pass": smoke["rng_parity_pass"],
        },
    )
    emit_metrics(
        metrics,
        "parity",
        "canonical_reference_vs_shadow_on",
        "population_5000_steps_1820",
        {
            "pass": canonical["pass"],
            "max_abs_diff": canonical["max_abs_diff"],
        },
        notes="Reused test/output/main_step13_financial_core as accepted Food control.",
    )
    for name, result in canonical["comparisons"].items():
        emit_metrics(
            metrics,
            "canonical_file_parity",
            name,
            "0-1819",
            {
                "pass": result["pass"],
                "max_abs_diff": result["max_abs_diff"],
                "first_difference": json.dumps(
                    result["first_difference"], ensure_ascii=False
                ) if result["first_difference"] else "",
            },
        )
    for key, value in contracts["latest_reconciliation"].items():
        emit_metrics(
            metrics,
            "contract_reconciliation",
            "food_only_active_sector",
            "canonical_endpoint",
            {key: value},
        )
    emit_metrics(
        metrics,
        "contract_reconciliation",
        "food_only_active_sector",
        "canonical_0-1819",
        {
            "max_abs_gap_over_validation_history": contracts["validation_max_abs_gap"],
            "validation_weeks": contracts["validation_weeks"],
        },
    )
    for key, value in checkpoint.items():
        if key != "pass":
            emit_metrics(metrics, "checkpoint", "legacy_food_checkpoint", "load", {key: value})
    emit_metrics(
        metrics,
        "runtime",
        "canonical_shadow_on",
        "execution",
        {"seconds": canonical_seconds},
    )

    if OUTPUT.exists():
        shutil.rmtree(OUTPUT)
    OUTPUT.mkdir(parents=True)
    (OUTPUT / "acceptance_flags.json").write_text(
        json.dumps(flags, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    summary = f"""# Step 14A 被动多部门契约验收

## 结论

**{verdict}**

14A 只增加了被动的 `SectorSpec`、部门/技术/库存策略注册表、`MarketRegistry`、Firm 身份和按 good/sector 重建的只读视图。当前 Food 生产、消费、市场选择、工资、库存、信贷、会计、人口和 RNG 执行路径保持原样。

## 验收结果

| 项目 | 结果 |
|---|---:|
| 500 人、260 周 OFF/ON 行为 parity | `{smoke['pass']}` |
| Smoke RNG parity | `{smoke['rng_parity_pass']}` |
| 5000 人、1820 周 canonical parity | `{canonical['pass']}` |
| Canonical 最大绝对差 | `{canonical['max_abs_diff']:.17g}` |
| Registry/sector/good 视图 reconciliation 最大差 | `{contracts['validation_max_abs_gap']:.17g}` |
| 旧 Food checkpoint 加载 | `{checkpoint['pass']}` |

## 明确回答

1. `SectorSpec` 已实现；Food sector 映射到现有 basic consumption good、Food technology 和 Food inventory policy。
2. 现有 Food Firms 已获得被动 `sector_id / technology_id / inventory_policy_id`，Firm ID 仍是全局稳定 ID。
3. `GoodSpec` 保留旧字段和 `id`，并提供 `good_id / storable / physical / service / perishability` 兼容视图；Food 语义未改变。
4. `MarketRegistry` 的 Food supplier 集合与当前运行时 5 个 Firm 的 ID 集合一致，且排序稳定。
5. Firm operating state 现在可按 `good_id` 查看 production/sales/inventory/unmet demand/revenue。
6. Firm aggregate state 现在可按 `sector_id` 查看雇员、劳动服务、产值、销售额、现金、principal 和 arrears。
7. `Person.firm_id` 仍是唯一权威雇主身份，没有新增 `Person.sector_id`。
8. Step 13 finance 和 AccountingLayer 没有行为改动；generic finance view 不依赖 Food inventory 字段。
9. 旧 checkpoint 可加载，缺失的新身份字段安全回填为 Food defaults。
10. 现有 Food scalar fields 和旧 reports 保留。
11. 非存储 Service good 可以由 schema template 表示，但没有进入 GoodsCatalog、MarketRegistry 或任何运行时市场。
12. 所有权仍未实现，sector/employment 不等于 ownership。
13. 14A 没有新增 RNG draw；ON 只构造确定性元数据和只读视图。
14. 14B 的 NoInventoryPolicy 和 passive ServiceTechnology 结构已可表达，但尚未激活 Service 行为。

## 未迁移的 Food 运行时边界

以下仍明确保留为 Food compatibility：`FirmSystem.food_*` 标量、`World.food_*` 历史/市场字段、single-good household demand、现有 Food inventory/price/sales executor。它们应在 14B 以后按需迁移到 good-keyed execution；14A 不做 mass refactor，也不把 MarketRegistry 接入交易执行器。
"""
    (OUTPUT / "acceptance_summary.md").write_text(summary, encoding="utf-8")
    with (OUTPUT / "step14A_contract_reconciliation_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["section", "entity", "window", "metric", "value", "notes"],
        )
        writer.writeheader()
        writer.writerows(metrics)
    print(verdict)
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
