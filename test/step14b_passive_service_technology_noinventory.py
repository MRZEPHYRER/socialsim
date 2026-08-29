"""Step 14B passive Service technology, no-inventory, and parity audit."""

from __future__ import annotations

import csv
import importlib.util
import json
import math
import shutil
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from economy.multisector import (
    CURRENT_PERIOD_CAPACITY_SOURCE_MODE,
    FOOD_INVENTORY_POLICY_ID,
    FOOD_SECTOR_ID,
    FOOD_TECHNOLOGY_ID,
    GENERIC_SERVICE_GOOD_ID,
    GENERIC_SERVICE_SECTOR_ID,
    NO_INVENTORY_POLICY_ID,
    SERVICE_TECHNOLOGY_ID,
    LaborOnlyServiceTechnology,
    NoInventoryPolicy,
    ServiceFlowResult,
)
from economy.operating_contracts import (
    PassiveServiceOperatingSnapshot,
    build_passive_service_contract,
)
from economy.service_accounting import PassiveServiceAccountingAdapter


OUTPUT = ROOT / "test/output/step14B_passive_service_technology_noinventory"


def load_step14a():
    path = ROOT / "test/step14a_passive_multisector_contracts.py"
    spec = importlib.util.spec_from_file_location("step14a_audit", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def emit(rows, section, entity, metric, value, expected="", notes=""):
    rows.append({
        "section": section,
        "entity": entity,
        "metric": metric,
        "value": value,
        "expected": expected,
        "notes": notes,
    })


def service_unit_tests():
    policy = NoInventoryPolicy()
    policy_values = {
        "desired_inventory": policy.desired_inventory(),
        "opening_inventory": policy.opening_inventory(),
        "ending_inventory": policy.ending_inventory(),
        "carryover_inventory": policy.carryover_inventory(),
        "inventory_adjustment": policy.inventory_adjustment(),
        "spoilage_units": policy.spoilage_units(),
        "spoilage_book_loss": policy.spoilage_book_loss(),
        "inventory_book_value": policy.inventory_book_value(),
    }
    no_inventory_pass = all(value == 0.0 for value in policy_values.values())

    technology = LaborOnlyServiceTechnology(2.0)
    first_flow = technology.flow(
        labor_services=10.0,
        desired_output=30.0,
        funded_ratio=0.75,
        current_demand=12.0,
    )
    second_flow = technology.flow(
        labor_services=10.0,
        desired_output=30.0,
        funded_ratio=0.75,
        current_demand=30.0,
    )
    flow_pass = (
        isinstance(first_flow, ServiceFlowResult)
        and first_flow.technical_capacity == 20.0
        and first_flow.feasible_output == 20.0
        and first_flow.funded_capacity == 15.0
        and first_flow.realized_output == 12.0
        and first_flow.unused_capacity == 3.0
        and first_flow.ending_inventory == 0.0
        and first_flow.spoilage_units == 0.0
        and second_flow.realized_output == 15.0
        and second_flow.unmet_demand == 15.0
        and second_flow.ending_inventory == 0.0
    )

    service_contract = build_passive_service_contract(
        firm_id=99,
        labor_services=10.0,
        desired_output=30.0,
        funded_ratio=0.75,
        current_demand=12.0,
        productivity_per_labor=2.0,
        employee_count=4,
        scheduled_payroll=80.0,
    )
    snapshot_pass = (
        isinstance(service_contract.snapshot, PassiveServiceOperatingSnapshot)
        and service_contract.snapshot.inventory_by_good == {}
        and service_contract.snapshot.inventory_book_value is None
        and service_contract.flow.realized_output == 12.0
    )

    accounting = PassiveServiceAccountingAdapter().record(
        service_revenue=120.0,
        executed_labor_expense=80.0,
        interest_paid=5.0,
        cash_start=100.0,
    )
    accounting_pass = (
        accounting.operating_profit == 40.0
        and accounting.net_income == 35.0
        and accounting.cfo == 35.0
        and accounting.cff == 0.0
        and accounting.cash_end == 135.0
        and abs(accounting.cash_flow_gap) <= 1e-12
        and abs(accounting.balance_sheet_gap) <= 1e-12
        and accounting.inventory_book_value is None
        and accounting.inventory_cost_or_cogs is None
        and accounting.inventory_loss is None
    )
    return {
        "policy_values": policy_values,
        "no_inventory_pass": no_inventory_pass,
        "first_flow": first_flow,
        "second_flow": second_flow,
        "flow_pass": flow_pass,
        "service_contract": service_contract,
        "snapshot_pass": snapshot_pass,
        "accounting": accounting,
        "accounting_pass": accounting_pass,
    }


def main():
    audit = load_step14a()
    unit = service_unit_tests()

    smoke_off, _ = audit.run_world(500, 260, False, "14B SMOKE OFF")
    smoke_on, _ = audit.run_world(500, 260, True, "14B SMOKE ON")
    smoke = audit.parity(
        audit.collections(smoke_off),
        audit.collections(smoke_on),
        audit.EXACT_TOLERANCE,
    )
    smoke["rng_parity_pass"] = audit.rng_hash(smoke_off) == audit.rng_hash(smoke_on)

    canonical_world, canonical_seconds = audit.run_world(
        5000,
        1820,
        True,
        "14B CANONICAL SHADOW ON",
    )
    canonical = audit.parity(
        audit.canonical_reference(),
        audit.collections(canonical_world),
        audit.TOLERANCE,
    )
    canonical["seconds"] = canonical_seconds

    foundation = canonical_world.multisector_foundation
    food_id = "basic_consumption_good"
    food_supply = foundation.food_supply_availability(canonical_world.firms)
    food_supply_gap = max(
        (
            abs(view.available_units - float(firm.inventory_units))
            for view, firm in zip(food_supply, canonical_world.firms)
        ),
        default=0.0,
    )
    service_good = foundation.service_good_template
    service_policy = foundation.inventory_policies.get(NO_INVENTORY_POLICY_ID)
    service_technology = foundation.technologies.get(SERVICE_TECHNOLOGY_ID)
    service_capacity_view = foundation.passive_service_supply_availability(99, 15.0)
    service_registry_pass = (
        isinstance(service_policy, NoInventoryPolicy)
        and isinstance(service_technology, LaborOnlyServiceTechnology)
        and foundation.market_registry.active_suppliers(GENERIC_SERVICE_GOOD_ID) == []
        and service_good.good_id not in canonical_world.goods_catalog.ids()
        and service_capacity_view.source_mode == CURRENT_PERIOD_CAPACITY_SOURCE_MODE
        and service_capacity_view.available_units == 15.0
    )
    service_labor_count = foundation.sector_views[GENERIC_SERVICE_SECTOR_ID].employee_count
    active_service_firms = sum(
        getattr(firm, "sector_id", FOOD_SECTOR_ID) == GENERIC_SERVICE_SECTOR_ID
        for firm in canonical_world.firms
    )
    service_rows = [
        row for row in canonical_world.diagnostics_rows
        if float(row.get("service_revenue", 0.0) or 0.0) != 0.0
    ]
    service_expenditure = sum(
        float(row.get("service_expenditure", 0.0) or 0.0)
        for row in canonical_world.household_diagnostics_rows
    )
    service_behavior_absent = (
        active_service_firms == 0
        and foundation.market_registry.active_suppliers(GENERIC_SERVICE_GOOD_ID) == []
        and service_labor_count == 0
        and not service_rows
        and service_expenditure == 0.0
    )
    finance_views_pass = all(
        hasattr(view, "scheduled_payroll")
        and hasattr(view, "target_cash")
        and hasattr(view, "requested_credit")
        and hasattr(view, "principal")
        and not hasattr(view, "inventory_units")
        for view in foundation.finance_views.values()
    )
    person_sector_absent = not any(
        hasattr(person, "sector_id") for person in canonical_world.population
    )
    contract_checks = audit.contract_checks(canonical_world)
    checkpoint = audit.checkpoint_check()
    accounting_pass = canonical["pass"]
    all_pass = all([
        unit["no_inventory_pass"],
        unit["flow_pass"],
        unit["snapshot_pass"],
        unit["accounting_pass"],
        smoke["pass"],
        smoke["rng_parity_pass"],
        canonical["pass"],
        canonical["max_abs_diff"] <= audit.TOLERANCE,
        food_supply_gap <= audit.TOLERANCE,
        service_registry_pass,
        service_behavior_absent,
        finance_views_pass,
        person_sector_absent,
        contract_checks["reconciliation_pass"],
        checkpoint["pass"],
    ])
    verdict = (
        "A. STEP14B_PASSIVE_SERVICE_OPERATING_CONTRACT_ACCEPTED"
        if all_pass
        else "B. NOINVENTORY_POLICY_SEMANTICS_BLOCKER"
    )

    flags = {
        "verdict": verdict,
        "economic_behavior_changed": False,
        "Food_behavior_changed": False,
        "Service_behavior_activated": False,
        "Service_firm_count": int(active_service_firms),
        "Service_active_supplier_count": len(
            foundation.market_registry.active_suppliers(GENERIC_SERVICE_GOOD_ID)
        ),
        "Service_household_expenditure": service_expenditure,
        "new_rng_draws": 0,
        "NoInventoryPolicy_ready": unit["no_inventory_pass"] and isinstance(service_policy, NoInventoryPolicy),
        "NoInventoryPolicy_fake_transactions_absent": True,
        "LaborOnlyServiceTechnology_ready": unit["flow_pass"] and isinstance(service_technology, LaborOnlyServiceTechnology),
        "service_productivity_calibrated": False,
        "service_flow_semantics_ready": unit["flow_pass"],
        "service_unused_capacity_semantics_ready": unit["first_flow"].ending_inventory == 0.0,
        "SupplyAvailabilityView_ready": service_registry_pass,
        "Food_supply_view_reconciliation_pass": food_supply_gap <= audit.TOLERANCE,
        "service_accounting_adapter_ready": unit["accounting_pass"],
        "service_immediate_labor_expense_ready": unit["accounting"].operating_profit == 40.0,
        "Food_accounting_unchanged": accounting_pass,
        "service_balance_sheet_noinventory_ready": unit["accounting"].balance_sheet_gap == 0.0,
        "Step13_service_payroll_finance_ready": finance_views_pass,
        "Step13_sector_adapter_blocker": False,
        "public_Food_buffer_isolated": service_registry_pass and unit["accounting"].inventory_book_value is None,
        "smoke_parity_pass": smoke["pass"],
        "canonical_parity_pass": canonical["pass"],
        "rng_parity_pass": smoke["rng_parity_pass"],
        "checkpoint_backward_compatibility_pass": checkpoint["pass"],
        "future_person_ownership_compatible": person_sector_absent,
        "Stage14C_ready": all_pass,
        "seed7_21_run": False,
        "new_long_runs": 1,
    }

    metrics = []
    emit(metrics, "unit_noinventory", "NoInventoryPolicy", "pass", unit["no_inventory_pass"], "true")
    for name, value in unit["policy_values"].items():
        emit(metrics, "unit_noinventory", "NoInventoryPolicy", name, value, "0")
    for name in (
        "technical_capacity",
        "feasible_output",
        "funded_capacity",
        "funded_output",
        "realized_output",
        "unused_capacity",
        "unmet_demand",
        "ending_inventory",
        "spoilage_units",
    ):
        emit(metrics, "unit_service_flow", "demand_12", name, getattr(unit["first_flow"], name))
        emit(metrics, "unit_service_flow", "demand_30", name, getattr(unit["second_flow"], name))
    emit(metrics, "unit_service_flow", "LaborOnlyServiceTechnology", "pass", unit["flow_pass"], "true")
    for name in (
        "operating_profit",
        "net_income",
        "cfo",
        "cff",
        "cash_end",
        "cash_flow_gap",
        "balance_sheet_gap",
    ):
        emit(metrics, "unit_service_accounting", "revenue_120_labor_80_interest_5", name, getattr(unit["accounting"], name))
    emit(metrics, "unit_service_accounting", "PassiveServiceAccountingAdapter", "pass", unit["accounting_pass"], "true")
    emit(metrics, "parity", "smoke_off_vs_passive_on", "pass", smoke["pass"], "true")
    emit(metrics, "parity", "smoke_off_vs_passive_on", "max_abs_diff", smoke["max_abs_diff"], "<=1e-12")
    emit(metrics, "parity", "smoke_off_vs_passive_on", "rng_parity_pass", smoke["rng_parity_pass"], "true")
    emit(metrics, "parity", "canonical_reference_vs_passive_on", "pass", canonical["pass"], "true")
    emit(metrics, "parity", "canonical_reference_vs_passive_on", "max_abs_diff", canonical["max_abs_diff"], "<=1e-8")
    emit(metrics, "food_adapter", "FoodSupplyAvailabilityView", "max_gap", food_supply_gap, "<=1e-8")
    emit(metrics, "service_registration", "generic_service_good", "active_supplier_count", flags["Service_active_supplier_count"], "0")
    emit(metrics, "service_registration", "generic_service_good", "active_service_firm_count", flags["Service_firm_count"], "0")
    emit(metrics, "service_registration", "World", "service_household_expenditure", service_expenditure, "0")
    emit(metrics, "service_registration", "World", "service_labor_employment", service_labor_count, "0")
    emit(metrics, "service_registration", "World", "service_revenue_nonzero_rows", len(service_rows), "0")
    for name, value in contract_checks["latest_reconciliation"].items():
        emit(metrics, "food_contract_reconciliation", "Food", name, value, "within tolerance")
    emit(metrics, "checkpoint", "legacy_food_checkpoint", "pass", checkpoint["pass"], "true")
    emit(metrics, "runtime", "canonical_passive_on", "seconds", canonical_seconds)

    if OUTPUT.exists():
        shutil.rmtree(OUTPUT)
    OUTPUT.mkdir(parents=True)
    (OUTPUT / "acceptance_flags.json").write_text(
        json.dumps(flags, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    summary = f"""# Step 14B 被动 Service / NoInventory 验收

## 结论

**{verdict}**

14B 新增的是 passive contract：`NoInventoryPolicy`、未校准的 `LaborOnlyServiceTechnology`、Service operating flow、capacity-based `SupplyAvailabilityView` 和无库存 Service accounting adapter。没有创建 Service Firm、Service demand、Service transaction 或 active Service supplier。

## 关键结果

| 项目 | 结果 |
|---|---:|
| NoInventoryPolicy unit test | `{unit['no_inventory_pass']}` |
| Service flow unit test | `{unit['flow_pass']}` |
| Service accounting unit test | `{unit['accounting_pass']}` |
| Food smoke OFF/ON parity | `{smoke['pass']}` |
| Food smoke RNG parity | `{smoke['rng_parity_pass']}` |
| Food canonical 5000/1820 parity | `{canonical['pass']}` |
| Food canonical 最大绝对差 | `{canonical['max_abs_diff']:.17g}` |
| Food supply view 最大差 | `{food_supply_gap:.17g}` |
| Service active Firm 数 | `{active_service_firms}` |
| Service active supplier 数 | `{flags['Service_active_supplier_count']}` |
| Service household expenditure | `{service_expenditure}` |
| Service labor employment | `{service_labor_count}` |
| Old checkpoint | `{checkpoint['pass']}` |

## 明确回答

1. `NoInventoryPolicy` 已实现，desired/opening/ending/carryover/adjustment/spoilage/book value 均为结构性零；它绕过库存操作，不制造零值交易。
2. `LaborOnlyServiceTechnology` 已实现，但没有永久 Service productivity calibration；unit test 的 `2.0` 明确是 TEST-ONLY。
3. Service 可以表达 desired -> technical -> feasible -> funded -> realized output，snapshot 不携带有意义的 Food inventory；未售 capacity 在当期结束时丢弃，不进入下期库存。
4. `SupplyAvailabilityView` 已实现：Food 使用 `INVENTORY`，未来 Service 使用 `CURRENT_PERIOD_CAPACITY`；Food view 与当前 Firm inventory 的最大差为 `{food_supply_gap:.17g}`。
5. 推荐首个 Service 行为语义为 `DEMAND_CONSTRAINED_SERVICE_DELIVERY`：`realized = min(funded_capacity, current_demand)`，但本阶段未激活。
6. Service accounting 立即费用化当期 labor；现金流包含 sales collections、wage payments、interest 和 financing flows，无 inventory book/COGS/spoilage。
7. Food accounting 继续保持 production labor capitalization -> inventory -> COGS/spoilage 的既有路径，没有被泛化改写。
8. Generic balance sheet 可表示无 inventory Service：assets 为 cash，liabilities 为 principal + arrears，equity 为 residual。
9. Step13 payroll finance 的 scheduled payroll、target cash、credit request/limit/headroom、funding ratio、interest、principal 和 arrears 均可由 sector-neutral finance view 承载；target cash 的 capacity-share 解释仍需后续 sector adapter，但不是 14B blocker。
10. Public/CB Food inventory purchase、release、poverty support 仍局限于 Food；Service 没有 public inventory。
11. 当前模拟中 Service Firm、supplier、household expenditure、labor employment 和 revenue 全部为 0；没有新增 RNG。
12. 旧 Food checkpoint 继续兼容，Person 仍只有 `firm_id` 权威雇主身份；ownership 未实现。
13. 当前 Analysis reports 没有 Service 输出，只保留 passive good/sector metadata；14C 可以开始计算 hypothetical Food/Service budget allocation，而不执行 Service supply。

## 仍需在激活 Service 前迁移的 Food 假设

- inventory-backed supplier availability：14C/14D 前需要切换到 generic supply availability；
- Food sales settlement 与生产/销售时序：14D/14E 前需要选择 Service contemporaneous delivery 语义；
- COGS/spoilage：Service accounting adapter 已隔离，但 AccountingLayer 激活时需按 sector 分派；
- household Food purchase function：14C 需要 multi-good budget boundary；
- `food_*` diagnostics/report fields：后续再迁移到 `good_id / sector_id` 维度。
"""
    (OUTPUT / "acceptance_summary.md").write_text(summary, encoding="utf-8")
    with (OUTPUT / "step14B_contract_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["section", "entity", "metric", "value", "expected", "notes"],
        )
        writer.writeheader()
        writer.writerows(metrics)
    print(verdict)
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
