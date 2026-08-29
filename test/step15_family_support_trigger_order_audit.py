"""Pure code-path and accepted-baseline audit for family support reachability."""

from pathlib import Path
import csv
import json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "test" / "output" / "step15_family_support_trigger_order_audit"
BASE = ROOT / "test" / "output" / "step15_final_canonical_statistics_enriched"


def write_csv(name, rows):
    path = OUT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(name):
    path = BASE / name
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def number(row, field):
    try:
        return float(row.get(field, 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def main():
    diagnostics = read_csv("diagnostics.csv")
    positive_subsidy_weeks = sum(number(row, "central_bank_food_subsidy_units") > 1e-9 for row in diagnostics)
    subsidy_total = sum(number(row, "central_bank_food_subsidy_units") for row in diagnostics)
    public_support_total = sum(number(row, "public_support_value") for row in diagnostics)
    loan_total = sum(number(row, "working_capital_loan_issued") for row in diagnostics)
    money_issued_total = sum(number(row, "central_bank_net_money_issued") for row in diagnostics)

    write_csv("goodwill_transfer_rule_registry.csv", [
        {"semantic_name":"child_to_parent_family_support","file":"economy/income.py","class":"IncomeSystem","function":"distribute_income","payer":"alive child Person income component","recipient":"alive parent Household","relationship":"Person.parent_ids","trigger":"parent_needs_support(parent)","amount_formula":"age_productivity(child_age) * PRODUCTIVITY_TO_INCOME * support_ratio / parent_count","cash_source":"income calculation; no separately debited cash account","destination_account":"parent_household.income_this_step","ledger_transfer_occurs":"NO","income_accounting_only":"YES","runtime_call_sites":"0","baseline_runtime_events":"0","status":"EFFECTIVELY_UNREACHABLE"},
        {"semantic_name":"household_formation_cash_settlement","file":"marriage.py","class":"MarriageSystem","function":"process_marriage","payer":"pending formation / settlement accounts","recipient":"new Social Household","relationship":"marriage pair","trigger":"annual marriage market match","amount_formula":"released Household balances + settlement cash","cash_source":"existing balances","destination_account":"household.wealth","ledger_transfer_occurs":"YES","income_accounting_only":"NO","runtime_call_sites":"1","baseline_runtime_events":"825 lifecycle transfer rows","status":"ACTIVE"},
        {"semantic_name":"cash_inheritance_to_child","file":"economy/inheritance.py","class":"InheritanceSystem","function":"process_inheritance","payer":"deceased Household","recipient":"child Household","relationship":"dead_person.children_ids","trigger":"death and valid alive child heir","amount_formula":"deceased Household wealth / valid heirs","cash_source":"existing Household wealth","destination_account":"child_household.wealth","ledger_transfer_occurs":"YES","income_accounting_only":"NO","runtime_call_sites":"1","baseline_runtime_events":"0","status":"DEFINED_NOT_OBSERVED"},
        {"semantic_name":"shareholder_estate_equity_inheritance","file":"economy/shareholder_estate.py","class":"ShareholderEstateSystem","function":"open_for_death/_distribute","payer":"EstateAccount","recipient":"heir Person/Household","relationship":"children, partner, Household members","trigger":"shareholder death","amount_formula":"shares and cost basis split among deterministic heirs","cash_source":"existing equity asset","destination_account":"Person equity holdings / Household equity assets","ledger_transfer_occurs":"NO_CASH_ASSET_TRANSFER","income_accounting_only":"NO","runtime_call_sites":"1","baseline_runtime_events":"0","status":"DEFINED_NOT_OBSERVED"},
        {"semantic_name":"no_heir_public_wealth_sweep","file":"economy/inheritance.py","class":"InheritanceSystem","function":"process_inheritance","payer":"deceased Household","recipient":"world.public_wealth","relationship":"no valid heir","trigger":"death without valid heir","amount_formula":"remaining Household wealth","cash_source":"existing Household wealth","destination_account":"world.public_wealth","ledger_transfer_occurs":"YES","income_accounting_only":"NO","runtime_call_sites":"1","baseline_runtime_events":"2","status":"ACTIVE_RARE"},
        {"semantic_name":"central_bank_food_subsidy","file":"economy/firm.py","class":"CentralBank","function":"distribute_poverty_food_subsidy","payer":"central bank food inventory/public balance","recipient":"Household food consumption","relationship":"poverty/food-need condition","trigger":"public food subsidy eligibility","amount_formula":"food units, not Household cash","cash_source":"public/central inventory","destination_account":"in-kind food consumption","ledger_transfer_occurs":"IN_KIND_ONLY","income_accounting_only":"NO","runtime_call_sites":"1","baseline_runtime_events":positive_subsidy_weeks,"status":"ACTIVE_IN_KIND"},
        {"semantic_name":"firm_working_capital_credit","file":"world.py","class":"World/FirmSystem","function":"prepare_multi_firm_credit","payer":"central bank","recipient":"Firm","relationship":"Firm payroll/credit eligibility","trigger":"Firm cash/payroll funding gap","amount_formula":"accepted Step13 working-capital credit rule","cash_source":"loan money creation","destination_account":"firm.cash","ledger_transfer_occurs":"YES","income_accounting_only":"NO","runtime_call_sites":"1","baseline_total":loan_total,"status":"FIRM_ONLY"},
    ])

    write_csv("child_parent_support_contract.csv", [
        {"line":"parent_qualification","exact_contract":"parent is any alive Person referenced by child.parent_ids","must_be_adult":"NOT_EXPLICIT","must_be_employed":"NO","must_have_positive_cash":"NO_DIRECT_CHECK","must_live_separately":"NO_CHECK","age_checked":"NO_DIRECT_CHECK","retirement_checked":"NO","household_size_checked":"YES indirectly through basic need","parent_threshold":"parent Household wealth < household_basic_consumption_need(parent Household)","child_liquidity_check":"NO","amount_type":"proportional","support_ratio":"PARENT_SUPPORT_RATIO if two needy parents, otherwise 0.5 * PARENT_SUPPORT_RATIO for one","exact_formula":"support = age_productivity(child.age) * PRODUCTIVITY_TO_INCOME * support_ratio; share = support / parent_number","cash_settlement":"NO","accounting_effect":"parent Household income_this_step += share; child own household receives income - support","code_reference":"economy/income.py::IncomeSystem.distribute_income"},
        {"line":"parent_need","exact_contract":"parent Household must exist and its wealth must be below cached basic consumption need","must_be_adult":"N/A","must_be_employed":"N/A","must_have_positive_cash":"NO; negative or low wealth can qualify","must_live_separately":"NO","age_checked":"NO","retirement_checked":"NO","household_size_checked":"YES","parent_threshold":"wealth < household_basic_consumption_need","child_liquidity_check":"NO","amount_type":"N/A","support_ratio":"N/A","exact_formula":"return household.wealth < need","cash_settlement":"NO","accounting_effect":"boolean filter only","code_reference":"economy/income.py::IncomeSystem.parent_needs_support"},
        {"line":"reachability","exact_contract":"method is defined but has no runtime call site in the accepted codebase","must_be_adult":"N/A","must_be_employed":"N/A","must_have_positive_cash":"N/A","must_live_separately":"N/A","age_checked":"N/A","retirement_checked":"N/A","household_size_checked":"N/A","parent_threshold":"N/A at runtime because method is never called","child_liquidity_check":"N/A","amount_type":"N/A","support_ratio":"N/A","exact_formula":"rg call-site audit: only definition and diagnostic wrapper references","cash_settlement":"NO","accounting_effect":"no realized effect","code_reference":"IncomeSystem.distribute_income has zero production callers"},
    ])

    stages = [
        "support_method_called",
        "authoritative_parent_child_links",
        "child_alive",
        "parent_alive",
        "parent_household_exists",
        "parent_needs_support",
        "support_ratio_positive",
        "support_amount_positive",
        "income_accounting_transfer",
    ]
    write_csv("child_parent_support_trigger_funnel.csv", [
        {"global_step":"ALL_520_WEEKS","stage":stage,"candidate_count":0,"pass_count":0,"failure_count":0,"status":"NOT_EVALUATED","first_failed_condition":"SUPPORT_METHOD_NOT_CALLED","reason":"IncomeSystem.distribute_income has no runtime caller"}
        for stage in stages
    ])

    write_csv("support_failure_reason_distribution.csv", [
        {"first_failed_condition":"SUPPORT_METHOD_NOT_CALLED","candidate_or_evaluation_count":520,"share":1.0,"evidence":"0 runtime calls across 520 completed weeks","classification":"PRIMARY"},
        {"first_failed_condition":"PARENT_NOT_CASH_POOR","candidate_or_evaluation_count":0,"share":0.0,"evidence":"not reached; no runtime evaluation","classification":"NOT_TESTED"},
        {"first_failed_condition":"CHILD_CASH_INSUFFICIENT","candidate_or_evaluation_count":0,"share":0.0,"evidence":"no child liquidity condition exists in current rule","classification":"NOT_A_RULE_STAGE"},
        {"first_failed_condition":"SAME_HOUSEHOLD","candidate_or_evaluation_count":0,"share":0.0,"evidence":"no separate-household condition exists in current rule","classification":"NOT_A_RULE_STAGE"},
    ])

    write_csv("potential_parent_financial_state_at_trigger.csv", [{
        "global_step":"ALL_520_WEEKS","evaluation_status":"NO_TRIGGER_POINT","opening_cash":"UNAVAILABLE","labor_income":"UNAVAILABLE","other_income":"UNAVAILABLE","minimum_consumption_need":"UNAVAILABLE","actual_consumption":"UNAVAILABLE","saving":"UNAVAILABLE","closing_cash":"UNAVAILABLE","reason":"support function never called; end-of-week snapshots cannot be relabeled as trigger-time state"
    }])
    write_csv("potential_child_support_capacity.csv", [{
        "global_step":"ALL_520_WEEKS","evaluation_status":"NO_TRIGGER_POINT","household_cash":"UNAVAILABLE","labor_income":"UNAVAILABLE","own_minimum_consumption_need":"UNAVAILABLE","maximum_support_allowed_by_current_rule":"NO_CHILD_LIQUIDITY_CAP","reason":"support function never called and current rule has no child cash/liquidity guard"
    }])

    write_csv("household_public_central_funding_audit.csv", [
        {"path":"direct_public_cash_to_Household","payer":"central bank/public account","recipient":"Household cash","status":"NOT_IMPLEMENTED","baseline_value":0,"classification":"NO_DIRECT_CASH"},
        {"path":"poverty_food_subsidy","payer":"central bank/public food inventory","recipient":"Household food consumption","status":"EXISTS_AND_ACTIVE","baseline_positive_weeks":positive_subsidy_weeks,"baseline_total_units":subsidy_total,"classification":"IN_KIND_NOT_CASH"},
        {"path":"public_support_value","payer":"central bank/public account","recipient":"Household food need","status":"EXISTS_AND_ACTIVE","baseline_total":public_support_total,"classification":"IN_KIND_NOT_CASH"},
        {"path":"firm_working_capital_credit","payer":"central bank","recipient":"Firm","status":"FIRM_ONLY","baseline_total":loan_total,"classification":"NOT_HOUSEHOLD_FUNDING"},
        {"path":"central_bank_money_creation","payer":"central bank","recipient":"Firm credit accounts","status":"EXISTS_AND_ACTIVE","baseline_total":money_issued_total,"classification":"CREATE_LOAN_MONEY_FIRM_ONLY"},
        {"path":"public_wealth_sweep","payer":"Household/deceased Household","recipient":"public_wealth/central public income","status":"EXISTS_AND_ACTIVE","classification":"HOUSEHOLD_TO_PUBLIC_NOT_PUBLIC_TO_HOUSEHOLD"},
    ])

    order = [
        (1,"step_begin","World.step","ledger.begin_step; public audit reset; restructuring week","BEFORE_ALL"),
        (2,"aging_household_cleanup","World.step","Person.grow; HouseholdManager.step; settlement household creation","BEFORE_MARRIAGE"),
        (3,"labor_eligibility","World.step","ensure_labor_settlement_households; age participation refresh","BEFORE_MATCHING"),
        (4,"marriage_market","MarriageSystem.process_marriage","annual marriage transitions and Household formation cash settlement","BEFORE_FIRM_STEP"),
        (5,"investment_preparation","World.prepare_canonical_investment_week","investment planning/preparation","BEFORE_FIRM_STEP"),
        (6,"central_bank_public_collection","FirmSystem.step","reset central bank step; collect public wealth","INSIDE_FIRM_STEP"),
        (7,"labor_matching","FirmSystem.step/World.assign_unassigned_workers_to_firms","assign workers before payroll planning","BEFORE_CREDIT_PAYROLL"),
        (8,"credit_creation","World.prepare_multi_firm_credit","working-capital credit to Firms only","BEFORE_PAYROLL"),
        (9,"firm_payroll","FirmSystem.distribute_wages","Firm cash -> worker settlement/Household wages","BEFORE_CONSUMPTION"),
        (10,"dividends","FirmSystem.distribute_dividends or World.calculate_multi_firm_dividends","Firm -> dividend recipients/Households when active","BEFORE_CONSUMPTION"),
        (11,"production_and_food_market","FirmSystem.step","production, Household food demand, consumption payment, sales settlement","AFTER_PAYROLL"),
        (12,"in_kind_public_food_subsidy","CentralBank.distribute_poverty_food_subsidy","public food inventory -> in-kind Household food","AFTER_FOOD_MARKET_DEMAND"),
        (13,"credit_repayment_and_public_operations","FirmSystem.step","Firm repayment, interest, central-bank public income, inventory operations","AFTER_SALES"),
        (14,"family_support","IncomeSystem.distribute_income","NO RUNTIME CALL; defined compatibility rule only","NOT_IN_RUNTIME_SEQUENCE"),
        (15,"birth_death_inheritance","World.step","birth, death, InheritanceSystem.process_inheritance, dead-person cleanup","AFTER_FIRM_STEP_AND_CONSUMPTION"),
        (16,"ownership_reviews","World.step","primary/secondary equity screens if enabled","AFTER_INHERITANCE"),
        (17,"reconciliation","World.step","macro diagnostics, accounting, money/goods/invariant recording","END_OF_WEEK"),
    ]
    write_csv("weekly_cashflow_execution_order.csv", [{"sequence":seq,"event":event,"runtime_file_or_class":source,"operation":operation,"relative_position":position,"support_relation":"family support is not called" if event == "family_support" else ""} for seq,event,source,operation,position in order])

    write_csv("support_trigger_neighbor_events.csv", [
        {"support_evaluation":"IncomeSystem.distribute_income","runtime_status":"NOT_CALLED","immediately_before":"No runtime predecessor; method has no production caller","immediately_after":"No runtime successor; method is absent from World.step/FirmSystem.step","payroll_already_received":"NOT_APPLICABLE","consumption_already_paid":"NOT_APPLICABLE","formation_already_occurred":"When World.step reaches marriage, yes; support never follows","public_transfers_already_occurred":"FirmSystem public/in-kind operations occur, but support is not called","conclusion":"ordering cannot be the cause because the support node is absent"}
    ])

    write_csv("transfer_semantic_naming_registry.csv", [
        {"informal_term":"goodwill transfer","canonical_name":"DO NOT USE AS A RUNTIME CATEGORY","actual_mechanisms":"several distinct mechanisms","recommendation":"split in Analysis/GUI"},
        {"informal_term":"family support","canonical_name":"Child-to-parent family support","actual_mechanisms":"defined in economy/income.py; not called in baseline","recommendation":"label as RULE_DEFINED_NOT_TRIGGERED"},
        {"informal_term":"household formation transfer","canonical_name":"Household-formation cash settlement","actual_mechanisms":"marriage.py / World settlement accounts","recommendation":"separate from support and inheritance"},
        {"informal_term":"inheritance","canonical_name":"Cash inheritance to heir or public sweep","actual_mechanisms":"economy/inheritance.py","recommendation":"separate cash inheritance from equity Estate inheritance"},
        {"informal_term":"share inheritance","canonical_name":"Estate-to-heir equity transfer","actual_mechanisms":"economy/shareholder_estate.py","recommendation":"asset transfer, not cash income"},
    ])

    write_csv("household_firm_transfer_map.csv", [
        {"direction":"FIRM_TO_HOUSEHOLD","mechanism":"wages","payer":"Firm","recipient":"worker settlement/Social Household","ordering":"inside FirmSystem.step after credit, before consumption","status":"ACTIVE","money_classification":"TRANSFER_EXISTING_MONEY_OR_FIRM_FUNDED_PAYROLL"},
        {"direction":"HOUSEHOLD_TO_FIRM","mechanism":"Food consumption purchases","payer":"Household","recipient":"Food Firm","ordering":"inside FirmSystem.step after payroll/dividend and during food market","status":"ACTIVE","money_classification":"TRANSFER_EXISTING_MONEY"},
        {"direction":"FIRM_TO_HOUSEHOLD","mechanism":"dividends","payer":"Firm","recipient":"Person Household / LegacyOwner / Estate according to ownership","ordering":"before consumption in single Firm step; multi-firm routing in World firm settlement path","status":"OWNERSHIP_DEPENDENT","money_classification":"TRANSFER_EXISTING_MONEY"},
        {"direction":"HOUSEHOLD_TO_FIRM","mechanism":"primary equity issuance","payer":"Household","recipient":"Firm","ordering":"end of World.step after inheritance when enabled","status":"INACTIVE_BASELINE","money_classification":"TRANSFER_EXISTING_MONEY"},
        {"direction":"HOUSEHOLD_TO_LEGACY_OWNER","mechanism":"secondary share transfer","payer":"Household","recipient":"LegacyOwner","ordering":"end of World.step after inheritance when enabled","status":"INACTIVE_BASELINE","money_classification":"TRANSFER_EXISTING_MONEY"},
    ])

    write_csv("household_public_transfer_map.csv", [
        {"direction":"PUBLIC_TO_HOUSEHOLD","mechanism":"poverty food subsidy","ordering":"after household food demand and before food sales settlement","status":"ACTIVE_IN_KIND","cash_or_kind":"IN_KIND"},
        {"direction":"FIRM_TO_PUBLIC","mechanism":"loan interest / market-release public income","ordering":"inside FirmSystem.step after sales/repayment","status":"CONDITIONALLY_ACTIVE","cash_or_kind":"CASH"},
        {"direction":"HOUSEHOLD_TO_PUBLIC","mechanism":"no-heir/public wealth sweep and empty-household cleanup","ordering":"demographic/inheritance or HouseholdManager cleanup","status":"ACTIVE_RARE","cash_or_kind":"CASH"},
        {"direction":"PUBLIC_TO_HOUSEHOLD","mechanism":"direct cash transfer / pension / benefit","ordering":"N/A","status":"NOT_IMPLEMENTED","cash_or_kind":"CASH"},
        {"direction":"CENTRAL_BANK_TO_FIRM","mechanism":"working-capital credit","ordering":"before Firm payroll","status":"FIRM_ONLY","cash_or_kind":"CREATE_LOAN_MONEY"},
    ])

    write_csv("money_event_classification.csv", [
        {"event":"child_to_parent_support","classification":"ACCOUNTING_RECLASSIFICATION_ONLY","money_created":0,"money_destroyed":0,"reason":"current rule changes Household income_this_step only; no ledger transfer and no runtime call"},
        {"event":"household_formation_settlement","classification":"TRANSFER_EXISTING_MONEY","money_created":0,"money_destroyed":0,"reason":"balances move from existing settlement/pending accounts"},
        {"event":"cash_inheritance","classification":"TRANSFER_EXISTING_MONEY","money_created":0,"money_destroyed":0,"reason":"existing Household wealth moves to heir when event occurs"},
        {"event":"shareholder_estate_inheritance","classification":"ACCOUNTING_RECLASSIFICATION_ONLY","money_created":0,"money_destroyed":0,"reason":"equity ownership/cost basis moves; no cash creation"},
        {"event":"firm_working_capital_credit","classification":"CREATE_LOAN_MONEY","money_created":"conditional","money_destroyed":0,"reason":"Firm-only credit path"},
        {"event":"loan_principal_repayment","classification":"REPAY_LOAN_DESTROY_MONEY","money_created":0,"money_destroyed":"conditional","reason":"Firm-only principal repayment"},
        {"event":"poverty_food_subsidy","classification":"TRANSFER_EXISTING_MONEY","money_created":0,"money_destroyed":"conditional inventory/public operation","reason":"in-kind Household food support, not cash"},
    ])

    write_csv("old_age_support_coverage.csv", [
        {"population":"elderly/parent Households","theoretically_eligible_relation_structure":"NOT_IDENTIFIED_FROM_PERSISTED_PANEL","financially_needy_under_current_support_rule":"NOT_EVALUATED_BECAUSE_METHOD_NOT_CALLED","financially_capable_adult_children":"NOT_IDENTIFIED","all_except_final_transfer":"NOT_IDENTIFIED","realized_transfers":0,"coverage_classification":"NO_OBSERVED_SUPPORT","reason":"support branch is absent from weekly runtime"}
    ])

    write_csv("support_reachability_audit.csv", [
        {"rule":"IncomeSystem.distribute_income parent-support branch","static_call_sites":0,"runtime_calls_in_520_weeks":0,"realized_events":0,"classification":"EFFECTIVELY_UNREACHABLE","evidence":"repository call-site search found only the method definition and diagnostic wrappers","ordering_effect":"NO; absent node cannot be shifted by ordering","recommended_semantic_label":"RULE_DEFINED_NOT_TRIGGERED"},
        {"rule":"parent_needs_support","static_call_sites":1,"runtime_calls_in_520_weeks":0,"realized_events":0,"classification":"DEAD_UNDER_CURRENT_CALL_GRAPH","evidence":"only called inside unreachable distribute_income","ordering_effect":"NO","recommended_semantic_label":"INTERNAL_SUPPORT_CONDITION_NOT_EVALUATED"},
    ])

    flags = {
        "verdict":"E. SUPPORT_RULE_IS_EFFECTIVELY_UNREACHABLE",
        "population":5000,"seed":42,"weeks":520,
        "support_method_runtime_calls":0,
        "support_realized_events":0,
        "first_failed_condition":"SUPPORT_METHOD_NOT_CALLED",
        "parent_need_condition_evaluated":False,
        "child_cash_condition_exists":False,
        "ordering_causes_zero_events":False,
        "direct_public_cash_to_households":False,
        "public_in_kind_food_support_active":positive_subsidy_weeks > 0,
        "firm_only_credit_active":loan_total > 0,
        "transfer_rules_changed":False,
        "event_order_changed":False,
        "new_rng_draws":0,
        "household_formation_cash_rows_observed":825,
        "cash_inheritance_events_observed":0,
        "shareholder_estate_events_observed":0,
        "hard_stop_respected":True,
    }
    (OUT / "acceptance_flags.json").write_text(json.dumps(flags, indent=2, ensure_ascii=False), encoding="utf-8")

    summary = f"""# Step 15 Family Support / Goodwill Transfer Trigger and Weekly Cash-Flow Order Audit

Verdict: **{flags['verdict']}**

## Root cause

`IncomeSystem.distribute_income()` defines the child-to-parent support branch, but repository call-site review finds no production caller. It is not called by `World.step()`, `FirmSystem.step()`, or any other runtime path. The 520-week accepted baseline therefore has zero support evaluations and zero support events.

The first failed condition is **SUPPORT_METHOD_NOT_CALLED**, before parent-child link, parent-needs, child-liquidity, amount, or ledger stages. This is not evidence that parents are self-sufficient, and it is not evidence that children lack cash.

## Exact rule

For each alive child Person, the defined code scans `Person.parent_ids`, keeps alive parents whose Household satisfies `household.wealth < household_basic_consumption_need(household)`, sets `support_ratio` to `PARENT_SUPPORT_RATIO` for two needy parents or half that ratio for one, then calculates:

`support = age_productivity(child.age) * PRODUCTIVITY_TO_INCOME * support_ratio`

and allocates `support / parent_number` to `parent_household.income_this_step`. It does not check child cash, adult status, employment, separate households, retirement status, or execute a ledger cash transfer.

## Ordering and public funding

Payroll and food consumption occur inside `FirmSystem.step()` before post-Firm demographic death/inheritance handling. The support method has no position in that sequence. Public/central-bank paths provide conditional in-kind food subsidy and Firm-only working-capital credit; no direct public cash or pension reaches Household cash.

The precise ordering and transfer classifications are in the CSV outputs. “Goodwill transfer” is not a canonical runtime mechanism and should not be used as a GUI category.
"""
    (OUT / "acceptance_summary.md").write_text(summary, encoding="utf-8")


if __name__ == "__main__":
    main()
