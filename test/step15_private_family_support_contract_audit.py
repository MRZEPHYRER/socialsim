"""Diagnostic-only audit of the legacy private family-support contract."""

from pathlib import Path
import csv
import json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "test" / "output" / "step15_private_family_support_contract_audit"
BASE = ROOT / "test" / "output" / "step15_final_canonical_statistics_enriched"


def read_csv(name):
    path = BASE / name
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def num(row, field):
    try:
        return float(row.get(field, 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


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


def main():
    diagnostics = read_csv("diagnostics.csv")
    subsidy_units = sum(num(row, "central_bank_food_subsidy_units") for row in diagnostics)
    subsidy_value = sum(num(row, "public_support_value") for row in diagnostics)
    subsidy_weeks = sum(num(row, "central_bank_food_subsidy_units") > 1e-9 for row in diagnostics)

    write_csv("family_support_legacy_origin_audit.csv", [
        {"component":"IncomeSystem","origin_evidence":"standalone economy/income.py class; no production caller","classification":"PARTIALLY_MIGRATED_MECHANISM","current_role":"legacy income allocation model disconnected from active Firm cash model","Ledger_aware":"NO"},
        {"component":"IncomeSystem.distribute_income","origin_evidence":"resets household income and allocates age-productivity income directly","classification":"LEGACY_PRE_LEDGER_INCOME_MODEL","current_role":"dead compatibility code under current call graph","Ledger_aware":"NO"},
        {"component":"PARENT_SUPPORT_RATIO","origin_evidence":"economy/config.py constant = 0.15","classification":"LEGACY_PRE_LEDGER_INCOME_MODEL","current_role":"formula input only","Ledger_aware":"NO"},
        {"component":"PRODUCTIVITY_TO_INCOME","origin_evidence":"economy/config.py constant = 100.0","classification":"LEGACY_INCOME_PROXY","current_role":"modeled earning-capacity proxy, not actual Household cash","Ledger_aware":"NO"},
        {"component":"active FirmSystem wage path","origin_evidence":"economy/firm.py distribute_wages writes Household.wealth and income_this_step","classification":"ACTIVE_MIGRATED_MECHANISM","current_role":"authoritative wage cash flow","Ledger_aware":"YES"},
    ])

    write_csv("support_income_cash_semantics.csv", [
        {"field":"Household.income_this_step","downstream_use":"household_food_purchase_units / consumption budget","effect_if_support_added_without_cash":"increases affordable_consumption and desired consumption","cash_changed":"NO","classification":"UNFUNDED_PURCHASING_POWER_RISK"},
        {"field":"Household.wealth","downstream_use":"cash balance and consumption payment","effect_if_support_added_without_cash":"does not increase; payment can exceed located cash","cash_changed":"NO","classification":"CASH_INCONSISTENCY"},
        {"field":"Household.saving_this_step","downstream_use":"income_this_step - actual consumption","effect_if_support_added_without_cash":"accounting saving may rise or appear less negative without cash counterpart","cash_changed":"NO","classification":"ACCOUNTING_ONLY_UNLESS_SETTLED"},
        {"field":"Household consumption","downstream_use":"actual_money=min(desired, income + wealth drawdown)","effect_if_support_added_without_cash":"can be funded by fictitious income","cash_changed":"NO","classification":"UNFUNDED_PURCHASING_POWER_RISK"},
        {"field":"Ledger","downstream_use":"authoritative money location and counterpart bridge","effect_if_support_added_without_cash":"no payer/recipient transaction exists","cash_changed":"NO","classification":"LEDGER_GAP"},
    ])

    write_csv("private_support_accounting_contract.csv", [
        {"contract_field":"payer","required_semantics":"child Social Household economic account, not Person object","current_status":"NOT_IMPLEMENTED","reason":"Person has no authoritative cash balance"},
        {"contract_field":"recipient","required_semantics":"parent Social Household wealth/cash account","current_status":"AVAILABLE_RUNTIME","reason":"Household.wealth is authoritative cash"},
        {"contract_field":"transaction","required_semantics":"Ledger.transfer_attrs(child_household.wealth -> parent_household.wealth)","current_status":"NOT_IMPLEMENTED","reason":"legacy function only mutates income_this_step"},
        {"contract_field":"income attribution","required_semantics":"parent Household intergenerational transfer income +X; child Household transfer-out -X","current_status":"NOT_IMPLEMENTED","reason":"no transfer-specific income/outflow fields"},
        {"contract_field":"saving attribution","required_semantics":"child saving includes -X; parent saving includes +X unless consumed","current_status":"NOT_IMPLEMENTED","reason":"requires ordered cash settlement and explicit transfer fields"},
        {"contract_field":"money identity","required_semantics":"child cash decrease = parent cash increase; money creation/destruction = 0","current_status":"DESIGN_REQUIREMENT","reason":"private existing-money transfer"},
        {"contract_field":"isolated fixture","required_semantics":"child cash 100 -> 80; parent cash 10 -> 30; support 20; money unchanged","current_status":"NOT_EXECUTED","reason":"verdict is redesign before activation; hard stop forbids implementation"},
    ])

    write_csv("family_relationship_contract.csv", [
        {"relationship":"Person.parent_ids / children_ids","authority":"runtime Person object","survives_marriage":"YES; marriage changes Household/partner, not parent_ids","survives_household_split":"YES while Persons remain alive","survives_settlement_transition":"YES; settlement_household_id is separate","survives_death":"NO for links to removed Person; World.remove_person_relationship deletes links","safe_for_runtime_transfer":"YES only during live synchronized runtime","historical_persistence":"NOT_AVAILABLE in accepted Household snapshots","risk":"cannot calculate historical coverage from persisted Household-only panels"},
        {"relationship":"Person.household_id","authority":"runtime settlement/social mapping","survives_marriage":"updated to new Household","survives_household_split":"updated","survives_settlement_transition":"social and settlement IDs are separate","survives_death":"cleared/removed with Person lifecycle","safe_for_runtime_transfer":"payer Household must resolve at evaluation time","historical_persistence":"NOT_AVAILABLE in target analysis panel","risk":"payer Household mapping needs same-step lookup"},
        {"relationship":"Household.parents / children","authority":"Household membership lists","survives_marriage":"new Household membership created","survives_household_split":"maintained by Household manager","survives_settlement_transition":"social vs settlement accounts distinct","survives_death":"dead member removed","safe_for_runtime_transfer":"recipient Household resolution available when alive","historical_persistence":"snapshot has counts, not IDs","risk":"coverage cannot be reconstructed from counts"},
    ])

    write_csv("support_need_definition_audit.csv", [
        {"candidate":"A_CURRENT_LEGACY","trigger":"parent Household wealth < household_basic_consumption_need","uses_cash":"YES","uses_income":"NO_DIRECT","uses_minimum_consumption":"YES, as threshold","uses_actual_consumption":"NO","uses_old_age":"NO","uses_retirement":"NO","status":"DEFINED_NOT_SELECTED_FOR_REDESIGN","interpretation":"cash-poverty trigger, not necessarily elderly income-gap trigger"},
        {"candidate":"B_INCOME_BELOW_NEED","trigger":"parent income < minimum consumption need","uses_cash":"NO_DIRECT","uses_income":"YES","uses_minimum_consumption":"YES","uses_actual_consumption":"NO","uses_old_age":"NO","uses_retirement":"NO","status":"SHADOW_ONLY","interpretation":"would target flow insufficiency more directly"},
        {"candidate":"C_CASH_FLOW_DEFICIT","trigger":"cash opening + income - necessary consumption < 0","uses_cash":"YES","uses_income":"YES","uses_minimum_consumption":"YES","uses_actual_consumption":"YES/SETTLEMENT","uses_old_age":"NO","uses_retirement":"NO","status":"SHADOW_ONLY","interpretation":"closer to current liquidity problem but requires exact ordering"},
        {"candidate":"D_OLD_AGE_INCOME_GAP","trigger":"old-age status and insufficient income/cash","uses_cash":"YES","uses_income":"YES","uses_minimum_consumption":"YES","uses_actual_consumption":"OPTIONAL","uses_old_age":"YES","uses_retirement":"OPTIONAL","status":"SHADOW_ONLY","interpretation":"institutionally distinct from generic family support"},
    ])

    write_csv("child_support_capacity_audit.csv", [
        {"capacity_signal":"child Household available cash","formula":"max(child_cash - liquidity_reserve - own_necessary_consumption_buffer, 0)","legacy_rule_uses":"NO","safety":"strongest cash-conservation guard","status":"NOMINATED_FOR_FUTURE_DESIGN_ONLY"},
        {"capacity_signal":"child income after own minimum consumption","formula":"max(child_income - own_minimum_need, 0)","legacy_rule_uses":"NO","safety":"flow-based but may not be located cash","status":"SHADOW_ONLY"},
        {"capacity_signal":"positive saving","formula":"max(child_income - own_consumption, 0)","legacy_rule_uses":"NO","safety":"observed ex post; less safe for same-week payment","status":"SHADOW_ONLY"},
        {"capacity_signal":"cash reserve above buffer","formula":"cash - reserve - own need","legacy_rule_uses":"NO","safety":"recommended combined constraint","status":"NOMINATED_FOR_FUTURE_DESIGN_ONLY"},
    ])

    write_csv("support_amount_formula_audit.csv", [
        {"formula":"age_productivity(child.age) * PRODUCTIVITY_TO_INCOME * support_ratio","meaning":"modeled earning-capacity / legacy income proxy","parent_need_sensitive":"NO except binary parent filter","child_cash_sensitive":"NO","child_actual_income_sensitive":"NO","capacity_capped":"NO","economic_coherence":"INCOMPLETE","risk":"can promise support beyond child Household cash"},
        {"formula":"min(legacy_formula, child_available_cash)","meaning":"legacy capacity bounded by located cash","parent_need_sensitive":"NO except binary filter","child_cash_sensitive":"YES","child_actual_income_sensitive":"INDIRECT","capacity_capped":"YES","economic_coherence":"BETTER BUT STILL ENGINEERING"},
        {"formula":"min(parent_need_gap, child_available_cash)","meaning":"need/capacity bounded transfer","parent_need_sensitive":"YES","child_cash_sensitive":"YES","child_actual_income_sensitive":"INDIRECT","capacity_capped":"YES","economic_coherence":"REQUIRES FUTURE CONTRACT DESIGN"},
        {"formula":"fixed family-support share","meaning":"stable private obligation","parent_need_sensitive":"OPTIONAL","child_cash_sensitive":"MUST BE CAPPED","child_actual_income_sensitive":"OPTIONAL","capacity_capped":"MUST BE CAPPED","economic_coherence":"NOT_SELECTED"},
    ])

    write_csv("multiple_child_parent_edge_cases.csv", [
        {"case":"one parent / multiple adult children","legacy_behavior":"each child can independently allocate support; no cash settlement","risk":"duplicate/order-dependent income allocation","required_future_contract":"aggregate child capacity and allocate deterministically"},
        {"case":"two parents share children","legacy_behavior":"support ratio differs for one vs two needy parents; support split by parent_number","risk":"parent need evaluated per child; no global cap","required_future_contract":"parent gap and total family cap"},
        {"case":"children in different Social Households","legacy_behavior":"parent_ids can identify them, but no payer Household cash lookup in legacy function","risk":"unfunded transfer","required_future_contract":"resolve each child Household and settle cash separately"},
        {"case":"parent/child same Household","legacy_behavior":"no exclusion; parent can be counted if parent_ids link is live","risk":"intra-Household fake transfer or double-counted income","required_future_contract":"exclude same economic account or classify internal allocation"},
        {"case":"death during relationship","legacy_behavior":"World removes links to dead Person","risk":"stale support target avoided, but estate route is separate","required_future_contract":"evaluate alive relation and Estate independently"},
    ])

    write_csv("support_execution_placement.csv", [
        {"candidate":"A_AFTER_PAYROLL_BEFORE_CONSUMPTION","observed_state":"wages and Firm credit settled; Household cash/income available","current_need_visibility":"good for current cash; before food consumption","can_fund_current_consumption":"YES","artifact_risk":"must protect child reserve and avoid double-counting subsidy","recommendation":"preferred future placement, not activated"},
        {"candidate":"B_AFTER_CONSUMPTION","observed_state":"current consumption already paid","current_need_visibility":"good for realized shortfall, too late for current consumption","can_fund_current_consumption":"NO, next week only","artifact_risk":"support becomes delayed and may create oscillation","recommendation":"not preferred"},
        {"candidate":"C_BEFORE_PAYROLL","observed_state":"prior-week closing cash only","current_need_visibility":"poor for current wage income","can_fund_current_consumption":"indirectly","artifact_risk":"support can substitute for payroll/credit timing","recommendation":"not preferred"},
        {"candidate":"D_AFTER_PUBLIC_FOOD_SUBSIDY","observed_state":"food need partly met in-kind","current_need_visibility":"avoids double support only if subsidy is explicitly netted","can_fund_current_consumption":"possible but late in current implementation","artifact_risk":"mixed private/public targeting","recommendation":"requires explicit interaction contract"},
    ])

    write_csv("family_support_food_subsidy_interaction.csv", [
        {"mechanism":"central bank poverty food subsidy","baseline_status":"ACTIVE_IN_KIND","positive_weeks":subsidy_weeks,"total_units":subsidy_units,"total_value":subsidy_value,"timing":"after Household food demand calculation, before final food sales settlement","need_basis":"Household food need and public subsidy balance","cash_to_household":"NO","overlap_risk":"family support evaluated after subsidy could double-cover the same minimum need","future_boundary":"choose whether private support targets pre-subsidy cash gap or residual post-subsidy gap"},
        {"mechanism":"private child-to-parent support","baseline_status":"NOT_ACTIVE","positive_weeks":0,"total_units":0,"total_value":0,"timing":"no current runtime position","need_basis":"legacy parent cash < basic need","cash_to_household":"DESIGN REQUIREMENT","overlap_risk":"must not silently duplicate in-kind public support","future_boundary":"separate private cash from public in-kind assistance"},
    ])

    write_csv("potential_private_support_coverage.csv", [
        {"scope":"accepted Household data","elderly_households":"AUTHORITATIVE COUNT AVAILABLE ONLY IN AGGREGATE","elderly_with_living_adult_child":"UNAVAILABLE","elderly_with_identifiable_child_household":"UNAVAILABLE","elderly_with_positive_child_transfer_capacity":"UNAVAILABLE","theoretical_full_coverage":"NOT_IDENTIFIABLE","reason":"accepted social_household_snapshots persist counts/cash but not Person IDs or parent-child links"},
        {"scope":"runtime relationship layer","elderly_with_living_adult_child":"COMPUTABLE AT RUNTIME","elderly_with_identifiable_child_household":"COMPUTABLE AT RUNTIME","elderly_with_positive_child_transfer_capacity":"REQUIRES FUTURE CASH SCREEN","theoretical_full_coverage":"NOT_RUN","reason":"no behavior change or coverage rerun in this audit"},
    ])

    write_csv("private_support_distributional_screen.csv", [
        {"parent_cash_quantile":"BOTTOM_50","parent_support_coverage":"UNAVAILABLE","child_capacity":"UNAVAILABLE","support_concentration":"NOT_IDENTIFIABLE","reason":"no persisted linked parent-child Household panel"},
        {"parent_cash_quantile":"MIDDLE_40","parent_support_coverage":"UNAVAILABLE","child_capacity":"UNAVAILABLE","support_concentration":"NOT_IDENTIFIABLE","reason":"no persisted linked parent-child Household panel"},
        {"parent_cash_quantile":"TOP_10","parent_support_coverage":"UNAVAILABLE","child_capacity":"UNAVAILABLE","support_concentration":"NOT_IDENTIFIABLE","reason":"no persisted linked parent-child Household panel"},
        {"overall":"shadow conclusion","parent_cash_quantile":"ALL","parent_support_coverage":"NOT_IDENTIFIABLE","child_capacity":"NOT_IDENTIFIABLE","support_concentration":"HIGH_INEQUALITY_RISK_CONCEPTUALLY","reason":"relationship-dependent support is structurally exposed to unequal child resources; no realized distribution claim"},
    ])

    write_csv("legacy_public_support_cleanup_screen.csv", [
        {"mechanism":"in-kind food subsidy","status":"active engineering/public safety-net path","classification":"RETAIN_BUT_MIGRATE_TO_FUTURE_GOVERNMENT_BOUNDARY","household_cash_effect":"no direct cash","money_creation":"conditional public inventory/central-bank operation","note":"do not merge with private family cash support"},
        {"mechanism":"public wealth from no-heir/deceased/empty Household","status":"active cleanup/accounting path","classification":"RETAIN_AS_ACCOUNTING_CLEANUP_UNTIL_PUBLIC_INSTITUTION_DESIGN","household_cash_effect":"Household -> public","money_creation":"none","note":"not a Household benefit"},
        {"mechanism":"Firm working-capital credit","status":"active","classification":"FIRM_ONLY_NOT_HOUSEHOLD_SUPPORT","household_cash_effect":"indirect through wages only","money_creation":"loan money creation","note":"must not fund private family support"},
    ])

    flags = {
        "verdict":"B. REDESIGN_PRIVATE_SUPPORT_BEFORE_ACTIVATION",
        "canonical_activation":False,
        "isolated_prototype_executed":False,
        "legacy_income_function_runtime_calls":0,
        "legacy_rule_is_partially_migrated":True,
        "income_without_cash_would_create_unfunded_purchasing_power":True,
        "authoritative_person_parent_links_exist_at_runtime":True,
        "persisted_linked_coverage_panel_available":False,
        "current_formula_capacity_capped":False,
        "current_formula_need_sensitive":False,
        "current_formula_economically_complete":False,
        "direct_public_cash_to_households":False,
        "public_in_kind_food_subsidy_active":subsidy_weeks > 0,
        "new_rng_draws":0,
        "transfer_rules_changed":False,
        "event_order_changed":False,
        "hard_stop_respected":True,
    }
    (OUT / "acceptance_flags.json").write_text(json.dumps(flags, indent=2, ensure_ascii=False), encoding="utf-8")

    summary = f"""# Step 15 Private Family Support Contract Restoration / Legacy-Income Audit

Verdict: **{flags['verdict']}**

## Conclusion

`IncomeSystem.distribute_income()` is best classified as a **partially migrated legacy income mechanism**: its parent-support branch is defined, but it is disconnected from the active Firm/Ledger cash architecture. It must not be activated unchanged.

The current formula is:

`age_productivity(child.age) * PRODUCTIVITY_TO_INCOME * support_ratio`

It is a modeled earning-capacity proxy. It does not use actual child Household cash, does not cap support by child liquidity, and does not make the parent amount proportional to a measured need gap beyond the binary legacy cash threshold.

If the old function were wired in as-is, it would increase `Household.income_this_step`, which directly affects the consumption budget, while leaving `Household.wealth` unchanged. That would create unfunded purchasing power and a cash/accounting inconsistency.

## Required redesign boundary

The future payer must be the child Social Household account because Person has no independent authoritative cash account. A valid transfer must debit child Household cash, credit parent Household cash, record transfer income/outflow, and close with zero money creation or destruction. A child liquidity cap and an explicit interaction with the existing in-kind food subsidy are required.

Runtime `parent_ids` and `children_ids` are authoritative while Persons are alive and remain valid across Household changes, but accepted historical Household snapshots do not persist linked Person/Household IDs. Therefore theoretical elderly coverage and distributional concentration cannot be safely quantified in this audit.

The active public food subsidy is in-kind, not direct Household cash: {subsidy_units:.6f} units across {subsidy_weeks} positive weeks in the accepted diagnostic baseline. It should remain separate from private cash support and later migrate to a Government boundary.

No canonical support activation, isolated transfer prototype, event-order change, pension, Government, wage, consumption, or retirement change was performed.
"""
    (OUT / "acceptance_summary.md").write_text(summary, encoding="utf-8")


if __name__ == "__main__":
    main()
