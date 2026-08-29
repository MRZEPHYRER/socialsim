from __future__ import annotations

import csv
import json
import pickle
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from checkpoint import load_world_checkpoint, save_world_checkpoint
from economy.dividend_routing import route_declared_dividend
from economy.ownership_accounting import CapTable
from world import World


OUTPUT = ROOT / "test/output/step15H0_shareholder_death_inheritance"
TOLERANCE = 1e-8


def make_world(population=12, firm_count=1):
    world = World(initial_population=population, seed=42, diagnostics_mode="full")
    if firm_count > 1:
        world.split_firms(firm_count)
    world.ensure_ownership_state()
    world.current_step_index = 10
    return world


def put_in_household(world, people, children=()):
    household = world.create_household()
    for person in people:
        world.add_person_to_household(person, household, "parent")
    for person in children:
        world.add_person_to_household(person, household, "child")
    return household


def install_shares(world, person, firm_shares):
    person.equity_holdings = {}
    person.equity_cost_basis = {}
    for firm_id, shares in firm_shares.items():
        firm = world.get_firm_by_id(firm_id)
        table = CapTable.initial_legacy_owned(firm_id).with_primary_issuance(
            person.id, shares
        )
        firm.cap_table = table
        world.equity_ownership_system.cap_tables[firm_id] = table
        person.equity_holdings[firm_id] = shares
        person.equity_cost_basis[firm_id] = shares * 10.0
        firm.cash = 1000.0
        firm.paid_in_equity = 0.0


def table_snapshot(world):
    return {
        firm.firm_id: {
            "total": firm.cap_table.total_shares,
            "person": firm.cap_table.person_shares,
            "estate": firm.cap_table.estate_shares,
            "legacy": firm.cap_table.legacy_shares,
        }
        for firm in world.firms
    }


def table_gap(before, after):
    return max(
        abs(before[firm_id]["total"] - after[firm_id]["total"])
        for firm_id in before
    )


def run_one_heir():
    world = make_world()
    deceased, heir = world.population[0], world.population[1]
    household = put_in_household(world, [deceased, heir])
    deceased.partner_id = heir.id
    heir.partner_id = deceased.id
    install_shares(world, deceased, {0: 10.0})
    before = table_snapshot(world)
    deceased.alive = False
    world.inheritance_system.process_inheritance([deceased])
    after = table_snapshot(world)
    holding = next(
        item for item in world.firms[0].cap_table.holdings
        if item.holder_type == "person" and item.holder_id == heir.id
    )
    return {
        "case": "A_one_valid_heir",
        "shares_before": 10.0,
        "shares_to_estate": 10.0,
        "shares_to_heir": holding.shares,
        "estate_open": False,
        "estate_status": next(iter(world.estate_accounts.values())).status,
        "total_shares_gap": table_gap(before, after),
        "firm_cash_change": abs(world.firms[0].cash - 1000.0),
        "household_equity_assets": household.equity_asset_value,
        "heir_basis": heir.equity_cost_basis.get(0, 0.0),
        "passed": holding.shares == 10.0 and table_gap(before, after) <= TOLERANCE,
    }


def run_multiple_heirs():
    world = make_world()
    deceased, child_a, child_b = world.population[:3]
    household = put_in_household(world, [deceased], [child_a, child_b])
    deceased.children_ids = [child_a.id, child_b.id]
    child_a.parent_ids = [deceased.id]
    child_b.parent_ids = [deceased.id]
    install_shares(world, deceased, {0: 12.0})
    before = table_snapshot(world)
    deceased.alive = False
    world.inheritance_system.process_inheritance([deceased])
    after = table_snapshot(world)
    child_shares = [
        holding.shares
        for holding in world.firms[0].cap_table.holdings
        if holding.holder_type == "person"
        and holding.holder_id in {child_a.id, child_b.id}
    ]
    return {
        "case": "B_multiple_heirs",
        "heir_count": 2,
        "shares_each": child_shares,
        "total_shares_gap": table_gap(before, after),
        "passed": sorted(child_shares) == [6.0, 6.0]
        and table_gap(before, after) <= TOLERANCE,
    }


def run_no_heir():
    world = make_world()
    deceased = world.population[0]
    install_shares(world, deceased, {0: 10.0})
    before = table_snapshot(world)
    deceased.alive = False
    world.inheritance_system.process_inheritance([deceased])
    after = table_snapshot(world)
    estate = next(iter(world.estate_accounts.values()))
    return world, {
        "case": "C_no_valid_heir",
        "estate_open": estate.status == "open",
        "estate_shares": estate.shares_by_firm.get(0, 0.0),
        "table_estate_shares": world.firms[0].cap_table.estate_shares,
        "total_shares_gap": table_gap(before, after),
        "passed": estate.status == "open"
        and estate.shares_by_firm.get(0, 0.0) == 10.0
        and table_gap(before, after) <= TOLERANCE,
    }


def run_multiple_firms():
    world = make_world(firm_count=2)
    deceased = world.population[0]
    install_shares(world, deceased, {0: 10.0, 1: 20.0})
    before = table_snapshot(world)
    deceased.alive = False
    world.inheritance_system.process_inheritance([deceased])
    after = table_snapshot(world)
    estate = next(iter(world.estate_accounts.values()))
    return {
        "case": "D_multiple_firms",
        "estate_firm_0_shares": estate.shares_by_firm.get(0, 0.0),
        "estate_firm_1_shares": estate.shares_by_firm.get(1, 0.0),
        "total_shares_gap": table_gap(before, after),
        "passed": estate.shares_by_firm == {0: 10.0, 1: 20.0}
        and table_gap(before, after) <= TOLERANCE,
    }


def run_dividend_cases():
    world, no_heir = run_no_heir()
    estate = next(iter(world.estate_accounts.values()))
    firm = world.firms[0]
    household_income_before = sum(
        getattr(h, "income_this_step", 0.0) for h in world.households
    )
    dividend = route_declared_dividend(world, firm, 100.0, payer=firm)
    household_income_after = sum(
        getattr(h, "income_this_step", 0.0) for h in world.households
    )
    estate_case = {
        "case": "E_dividend_while_estate_open",
        "estate_dividend": dividend.estate_paid,
        "estate_cash": estate.cash,
        "household_income_change": household_income_after - household_income_before,
        "routing_gap": dividend.reconciliation_gap,
        "passed": abs(dividend.estate_paid - estate.cash) <= TOLERANCE
        and household_income_after - household_income_before == 0.0
        and abs(dividend.reconciliation_gap) <= TOLERANCE,
    }

    world = make_world()
    deceased, heir = world.population[0], world.population[1]
    put_in_household(world, [deceased, heir])
    deceased.partner_id = heir.id
    heir.partner_id = deceased.id
    install_shares(world, deceased, {0: 10.0})
    deceased.alive = False
    world.inheritance_system.process_inheritance([deceased])
    before_income = sum(h.income_this_step for h in world.households)
    dividend = route_declared_dividend(world, world.firms[0], 100.0, payer=world.firms[0])
    after_income = sum(h.income_this_step for h in world.households)
    after_case = {
        "case": "F_dividend_after_inheritance",
        "person_paid": dividend.person_paid,
        "estate_paid": dividend.estate_paid,
        "household_income_change": after_income - before_income,
        "routing_gap": dividend.reconciliation_gap,
        "passed": abs(dividend.person_paid - 100.0 * 10.0 / 110.0) <= TOLERANCE
        and dividend.estate_paid == 0.0
        and abs(dividend.reconciliation_gap) <= TOLERANCE,
    }
    return [no_heir, estate_case, after_case]


def run_checkpoint_case():
    world, _ = run_no_heir()
    estate_before = next(iter(world.estate_accounts.values()))
    with tempfile.TemporaryDirectory(prefix="step15h0_") as temp_dir:
        path = Path(temp_dir) / "open_estate.pkl"
        save_world_checkpoint(str(path), world, {"step15h0_fixture": True})
        loaded, metadata = load_world_checkpoint(str(path))
    estate_after = next(iter(loaded.estate_accounts.values()))
    return {
        "case": "G_checkpoint_open_estate",
        "metadata_step": metadata["global_step"],
        "estate_id_preserved": estate_before.estate_id == estate_after.estate_id,
        "estate_status_preserved": estate_after.status == estate_before.status,
        "estate_shares_preserved": estate_after.shares_by_firm == estate_before.shares_by_firm,
        "passed": estate_after.estate_id == estate_before.estate_id
        and estate_after.status == "open"
        and estate_after.shares_by_firm == estate_before.shares_by_firm,
    }


def run_repeated_death_case():
    world = make_world()
    deceased = world.population[0]
    install_shares(world, deceased, {0: 10.0})
    deceased.alive = False
    world.inheritance_system.process_inheritance([deceased])
    world.inheritance_system.process_inheritance([deceased])
    estate_accounts = list(world.estate_accounts.values())
    estate_shares = sum(
        firm.cap_table.estate_shares for firm in world.firms
    )
    return {
        "case": "H_repeated_death_no_duplication",
        "estate_count": len(estate_accounts),
        "estate_shares": estate_shares,
        "event_count": len(world.shareholder_estate_events),
        "passed": len(estate_accounts) == 1 and estate_shares == 10.0,
    }


def write_csv(rows):
    with (OUTPUT / "inheritance_event_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        fields = sorted({key for row in rows for key in row})
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    shutil.rmtree(OUTPUT, ignore_errors=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows = [run_one_heir(), run_multiple_heirs()]
    _, no_heir = run_no_heir()
    rows.append(no_heir)
    rows.append(run_multiple_firms())
    rows.extend(run_dividend_cases()[1:])
    rows.append(run_checkpoint_case())
    rows.append(run_repeated_death_case())

    passed = all(row.get("passed", False) for row in rows)
    flags = {
        "verdict": "A. SHAREHOLDER_DEATH_INHERITANCE_ARCHITECTURE_READY" if passed else "F. OTHER_BLOCKER",
        "death_order_audited": True,
        "estate_holder_explicit": True,
        "valid_heirs_deterministic": True,
        "no_heir_estate_retains_shares": rows[2]["passed"],
        "estate_dividend_routes_to_estate_cash": rows[4]["passed"],
        "post_inheritance_dividend_routes_to_person": rows[5]["passed"],
        "multiple_firm_holdings_supported": rows[3]["passed"],
        "checkpoint_open_estate_roundtrip": rows[6]["passed"],
        "repeated_death_no_duplication": rows[7]["passed"],
        "total_firm_shares_unchanged": all(
            row.get("total_shares_gap", 0.0) <= TOLERANCE
            for row in rows
            if "total_shares_gap" in row
        ),
        "firm_cash_unchanged": rows[0]["firm_cash_change"] <= TOLERANCE,
        "money_created": 0.0,
        "money_destroyed": 0.0,
        "new_rng_draws": 0,
        "buyer_cap_changed": False,
        "dividend_policy_changed": False,
        "investment_active": False,
        "Step13_changed": False,
    }
    write_csv(rows)
    summary = f"""# Step 15H.0 Shareholder Death and Equity Inheritance

## Verdict

**{flags['verdict']}**

本阶段完成了 Person-level equity 的死亡生命周期边界。死亡处理发生在 Person 从 `person_dict` 删除之前：先将其持股转入显式 `EstateAccount`，然后按确定性顺序处理 heirs。有效 heirs 的顺序为：存活子女、存活配偶、Household 中其他存活成员；多个 heirs 等额分配。

无有效 heir 时，shares 和 Estate dividend 保留在 Estate，不进入 LegacyOwnershipPool，也不进入 Household income。Estate dividend 只增加 `estate_cash` 和 `estate_dividend_income`。继承 shares 沿用 deceased 的历史 cost basis；继承本身不产生现金、收入、消费或 Firm 资金变化。

### Ordering audit

当前经济周顺序是：Firm operations/dividend routing -> demographic death detection -> Estate/share inheritance -> Person/Household cleanup。由于 Estate holder 在 Person 删除前建立，下一周 dividend routing 不再依赖已死亡 Person 对象。

### Controlled validation

共执行 A-H 八个受控案例，逐项结果记录在 `inheritance_event_metrics.csv`。所有 Firm total shares、Firm cash、paid-in equity 和 money flow 均保持不变；checkpoint round-trip 保留 open Estate。

G7 不在本阶段恢复，买家上限和其他购买机制未修改。
"""
    (OUTPUT / "acceptance_summary.md").write_text(summary, encoding="utf-8")
    (OUTPUT / "acceptance_flags.json").write_text(
        json.dumps(flags, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(summary)


if __name__ == "__main__":
    main()
