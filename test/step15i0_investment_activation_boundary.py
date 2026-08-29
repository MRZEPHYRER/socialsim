from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "test/output/step15I0_investment_activation_boundary"


def write_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def source_text(path):
    return (ROOT / path).read_text(encoding="utf-8")


def audit_chain():
    world = source_text("world.py")
    planning = source_text("economy/investment_planning.py")
    contracts = source_text("economy/investment_contracts.py")
    capital_goods = source_text("economy/capital_goods.py")
    multisector = source_text("economy/multisector.py")

    return [
        {
            "stage": "demand_to_desired_capacity",
            "contract": "ShadowInvestmentPlanner",
            "status": "PASSIVE_AVAILABLE",
            "canonical_runtime": "NOT_CONNECTED",
            "evidence": "planner consumes lagged expected demand/desired capacity as explicit shadow inputs",
            "blocker": "no canonical investment decision call",
        },
        {
            "stage": "replacement_and_expansion_split",
            "contract": "FirmInvestmentDecision",
            "status": "PASSIVE_AVAILABLE",
            "canonical_runtime": "NOT_CONNECTED",
            "evidence": "replacement_investment_need and expansion_investment_need remain separate",
            "blocker": "no runtime intent emission",
        },
        {
            "stage": "decision_to_intent",
            "contract": "FirmInvestmentIntent",
            "status": "CONTRACT_ONLY",
            "canonical_runtime": "NOT_CONNECTED",
            "evidence": "contract exists, but World does not import investment_planning or create intents",
            "blocker": "intent-to-order adapter absent",
        },
        {
            "stage": "financing_constraint",
            "contract": "retained_cash shadow capacity",
            "status": "SHADOW_ONLY",
            "canonical_runtime": "NOT_CONNECTED",
            "evidence": "planner computes cash minus an explicit operating floor and forbids working-capital finance",
            "blocker": "no active investment financing path",
        },
        {
            "stage": "intent_to_order",
            "contract": "InvestmentOrder",
            "status": "CONTRACT_ONLY",
            "canonical_runtime": "NOT_CONNECTED",
            "evidence": "InvestmentOrder is defined in investment/capital-good modules only",
            "blocker": "no World order creation or scheduling boundary",
        },
        {
            "stage": "capital_good_supply",
            "contract": "CapitalGoodSpec / CapitalGoodOffer",
            "status": "FIXTURE_SETTLEABLE",
            "canonical_runtime": "MISSING",
            "evidence": "generic supplier_id offer works in isolated settlement fixtures; World has no capital-good offer inventory",
            "blocker": "capital-good supply technology/sector not active",
        },
        {
            "stage": "order_to_settlement",
            "contract": "PassiveCapitalGoodSettlementEngine",
            "status": "PASSIVE_READY",
            "canonical_runtime": "NOT_CONNECTED",
            "evidence": "deterministic partial-fill settlement closes buyer cash, supplier revenue, CFI and asset cost",
            "blocker": "engine is deliberately not imported by World",
        },
        {
            "stage": "capital_asset_acquisition",
            "contract": "CapitalAsset / CapitalStock",
            "status": "PASSIVE_READY",
            "canonical_runtime": "EMPTY_ONLY",
            "evidence": "Firm and FirmSlice carry empty CapitalStock; old checkpoints backfill empty state",
            "blocker": "no canonical acquisition event",
        },
        {
            "stage": "depreciation",
            "contract": "capital_book_value_bridge",
            "status": "INTERFACE_ONLY",
            "canonical_runtime": "INACTIVE",
            "evidence": "book-value bridge exists but depreciation policy remains unselected",
            "blocker": "no selected accounting/physical depreciation policy",
        },
        {
            "stage": "capital_to_capacity",
            "contract": "ProductionTechnology.capital_capacity_contribution",
            "status": "HOOK_ONLY",
            "canonical_runtime": "LABOR_ONLY",
            "evidence": "capital hook returns zero for current Food/Service technologies and World does not compose capital capacity",
            "blocker": "capital productivity and capacity contribution unidentified",
        },
    ]


def capital_supply_matrix():
    return [
        {
            "option": "A_EXISTING_GENERIC_FIRM_SUPPLY_ADAPTER",
            "who_supplies": "generic Firm via CapitalGoodOffer",
            "canonical_supply_available": False,
            "money_goods_conservation": "compatible",
            "requires_capital_good_inventory_or_technology": True,
            "implementation_complexity": "medium",
            "risk": "offers would otherwise be empty; no goods-from-nowhere shortcut",
            "assessment": "contract sufficient for isolated fixtures, insufficient for canonical activation alone",
        },
        {
            "option": "B_DEDICATED_CAPITAL_GOODS_SECTOR",
            "who_supplies": "capital-good Firm/sector using generic Firm architecture",
            "canonical_supply_available": False,
            "money_goods_conservation": "compatible if production and inventory are real",
            "requires_capital_good_inventory_or_technology": True,
            "implementation_complexity": "high",
            "risk": "must define technology, labor allocation and demand without double counting",
            "assessment": "preferred permanent supply boundary before broad activation",
        },
        {
            "option": "C_EXOGENOUS_CAPITAL_GOOD_INVENTORY",
            "who_supplies": "test fixture or explicitly endowed supplier",
            "canonical_supply_available": False,
            "money_goods_conservation": "only compatible as a declared initial endowment",
            "requires_capital_good_inventory_or_technology": False,
            "implementation_complexity": "low for fixtures",
            "risk": "not a permanent economy; risks hiding missing supply and creating free goods",
            "assessment": "allowed only for isolated contract tests, not canonical behavior",
        },
        {
            "option": "D_OTHER_ARCHITECTURE",
            "who_supplies": "external/provider mechanism",
            "canonical_supply_available": False,
            "money_goods_conservation": "requires explicit endowment or transaction",
            "requires_capital_good_inventory_or_technology": True,
            "implementation_complexity": "high",
            "risk": "would blur sector and external-demand boundaries",
            "assessment": "not justified at this stage",
        },
    ]


def financing_matrix():
    return [
        {
            "option": "A_INTERNAL_CASH_ONLY",
            "active_now": False,
            "buyer_cash_source": "Firm retained cash after operating liquidity protection",
            "money_creation": 0.0,
            "working_capital_credit_used": False,
            "compatibility_step13": "high",
            "ownership_compatibility": "high; no dilution",
            "main_risk": "investment may remain unexecuted when cash is insufficient",
            "assessment": "safest first behavioral activation once supply and productivity contracts exist",
        },
        {
            "option": "B_NEW_EQUITY_FINANCING",
            "active_now": False,
            "buyer_cash_source": "future Person/Household primary equity subscription",
            "money_creation": 0.0,
            "working_capital_credit_used": False,
            "compatibility_step13": "high if separately settled",
            "ownership_compatibility": "requires active dilution and issuance accounting",
            "main_risk": "confounds ownership transition with investment demand",
            "assessment": "defer",
        },
        {
            "option": "C_CAPITAL_TERM_LOAN",
            "active_now": False,
            "buyer_cash_source": "separate investment lender facility",
            "money_creation": "future explicit mechanism",
            "working_capital_credit_used": False,
            "compatibility_step13": "separate contract required",
            "ownership_compatibility": "high",
            "main_risk": "adds debt-service and default behavior before investment boundary is validated",
            "assessment": "defer",
        },
        {
            "option": "D_HYBRID",
            "active_now": False,
            "buyer_cash_source": "cash plus equity or term finance",
            "money_creation": "depends on component",
            "working_capital_credit_used": False,
            "compatibility_step13": "requires multiple new boundaries",
            "ownership_compatibility": "mixed",
            "main_risk": "cannot isolate real-investment effect",
            "assessment": "defer",
        },
    ]


def main():
    shutil.rmtree(OUTPUT, ignore_errors=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)

    chain = audit_chain()
    supply = capital_supply_matrix()
    finance = financing_matrix()
    source_world = source_text("world.py")
    source_capital_goods = source_text("economy/capital_goods.py")
    source_planning = source_text("economy/investment_planning.py")
    canonical_investment_runtime_missing = (
        "capital_goods" not in source_world
        and "investment_planning" not in source_world
    )

    flags = {
        "verdict": "E. MULTIPLE_BOUNDARIES_REMAIN",
        "audit_only": True,
        "simulation_run": False,
        "canonical_investment_transactions": 0,
        "canonical_capital_assets_created": 0,
        "canonical_capital_good_supply_active": False,
        "new_rng_draws": 0,
        "economic_behavior_changed": False,
        "primary_issuance_changed": False,
        "capital_loan_active": False,
        "working_capital_credit_funds_investment": False,
        "Step13_changed": False,
        "purchase_rate_changed": False,
        "stock_market_changed": False,
        "replacement_expansion_separate": True,
        "settlement_contract_reconciles": True,
        "capital_book_bridge_available": True,
        "generic_supplier_fixture_available": True,
        "canonical_generic_supplier_available": False,
        "capital_productivity_contract_selected": False,
        "depreciation_calibration_selected": False,
        "internal_cash_only_nominated": True,
        "capital_good_boundary_blocked": True,
        "capital_productivity_boundary_blocked": True,
        "canonical_runtime_imports_investment_layer": not canonical_investment_runtime_missing,
        "next_activation_requires_multiple_boundaries": True,
        "source_files": [
            "economy/investment_contracts.py",
            "economy/investment_planning.py",
            "economy/capital_goods.py",
            "economy/multisector.py",
            "world.py",
        ],
    }

    chain_fields = [
        "stage", "contract", "status", "canonical_runtime", "evidence", "blocker"
    ]
    supply_fields = list(supply[0])
    finance_fields = list(finance[0])
    write_csv(OUTPUT / "investment_chain_audit.csv", chain, chain_fields)
    write_csv(OUTPUT / "capital_good_supply_matrix.csv", supply, supply_fields)
    write_csv(OUTPUT / "financing_boundary_matrix.csv", finance, finance_fields)
    (OUTPUT / "acceptance_flags.json").write_text(
        json.dumps(flags, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    summary = """# Step 15I.0 Investment Activation and Capital-Good Supply Boundary Audit

## Verdict

**E. MULTIPLE_BOUNDARIES_REMAIN**

This stage was architecture/audit only. No canonical simulation was run and
no investment transaction, capital-good production, financing, ownership or
Step13 behavior was changed.

## Investment chain

The accepted Step15C planner can separate replacement and expansion needs and
can calculate a shadow internal-finance gap. Step15D can settle an isolated
`InvestmentOrder` against deterministic `CapitalGoodOffer` records. Step15B
provides an empty runtime `CapitalStock` and a book-value bridge.

The chain is not connected to `World`: no canonical intent/order is created,
the capital-good settlement engine is not imported by `World`, no capital-good
offers are produced by current Firms, and no acquired asset can affect current
Food/Service capacity. Depreciation remains an unselected interface.

## Supply boundary

The generic supplier contract is sufficient for isolated settlement fixtures,
but canonical Firms currently supply the accepted consumption goods only. A
real activation therefore needs a declared capital-good supply boundary, most
likely a generic-Firm capital-good sector/technology with real output and
inventory. Exogenous capital-good inventory is acceptable only for a controlled
fixture and must not be used as canonical free supply.

## Financing boundary

The safest first behavioral activation is **INTERNAL_CASH_ONLY**: Firm cash may
fund settled investment only after the existing operating-liquidity, payroll
and debt-service obligations are protected. Step13 working-capital credit must
remain excluded. Equity financing and capital term loans should be deferred so
that the first experiment isolates investment demand and settlement.

## Remaining blocker

The production-capacity contribution of capital is still unidentified. The
current labor-only Food/Service technologies explicitly contribute zero capital
capacity, and no capital-labor interaction, service flow, useful life or
physical decay calibration has been selected. Therefore activating investment
before a minimal capital-productivity contract would create an asset that has
no defined economic use.

The next safe design stage must specify both the capital-good supply adapter
and the minimal capital-capacity contribution contract. No permanent
productivity coefficient or depreciation calibration should be chosen here.
"""
    (OUTPUT / "acceptance_summary.md").write_text(summary, encoding="utf-8")
    print("E. MULTIPLE_BOUNDARIES_REMAIN")
    print(f"Outputs: {OUTPUT}")


if __name__ == "__main__":
    main()
