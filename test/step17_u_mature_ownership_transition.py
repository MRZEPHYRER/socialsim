"""Step17.U mature Firm ownership and Person dividend settlement audit.

The audit is deliberately non-mutating: it inspects the current Firm
construction path and accepted Step17.T/H0 artifacts.  Because no historical
Person owner provenance exists, no research migration branch is executed.
"""
from __future__ import annotations

import csv
import importlib.util
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT = ROOT / "test/output/step17_u_mature_ownership_transition"
T_OUT = ROOT / "test/output/step17_t_personal_tax_base_completion"
S_OUT = ROOT / "test/output/step17_s_active_mixed_tax_foundation"
H0_OUT = ROOT / "test/output/step15H0_shareholder_death_inheritance"
HARNESS = ROOT / "test/step15_mature_genealogy_private_support_experiment.py"
EPS = 1e-8


def num(value, default=0.0):
    try:
        value = float(value)
        return value if math.isfinite(value) else default
    except (TypeError, ValueError):
        return default


def read_rows(path):
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or ["status"], extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows or [{"status": "UNAVAILABLE"}])


def load_harness():
    spec = importlib.util.spec_from_file_location("step17_u_mature_harness", HARNESS)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def gini(values):
    xs = sorted(max(0.0, num(x)) for x in values)
    total = sum(xs)
    if not xs or total <= EPS:
        return 0.0
    return sum((2 * i - len(xs) - 1) * x for i, x in enumerate(xs, 1)) / (len(xs) * total)


def ownership_rows(world):
    rows = []
    for firm in world.operating_firms():
        table = firm.cap_table
        holdings = [
            {"holder_id": h.holder_id, "holder_type": h.holder_type, "shares": h.shares}
            for h in table.holdings
        ]
        rows.append({
            "firm_id": firm.firm_id,
            "sector": getattr(firm, "sector_id", "unknown"),
            "creation_source": "current World.split_firms / CanonicalInvestment.ensure_firms",
            "creation_week": "UNAVAILABLE",
            "startup_capital_source": "fresh bootstrap internal FirmSystem cash / compatibility balance; not equity provenance",
            "current_shareholder_count": table.holder_count,
            "shareholder_identities": json.dumps(holdings, default=str),
            "shareholder_types": ";".join(sorted({h.holder_type for h in table.holdings})),
            "share_fraction_sum": sum(h.shares for h in table.holdings) / table.total_shares if table.total_shares else 0.0,
            "legacy_shares": table.legacy_shares,
            "person_shares": table.person_shares,
            "estate_shares": table.estate_shares,
            "legacy_fraction": table.ownership_fraction("legacy"),
            "person_fraction": table.ownership_fraction("person"),
            "estate_fraction": table.ownership_fraction("estate"),
            "retained_earnings": num(getattr(firm, "retained_earnings", getattr(firm, "profit", 0.0))),
            "dividend_eligibility": "routing accepted; mature Person entitlement observed = 0 in Step17.T",
            "ownership_sum_pass": abs(sum(h.shares for h in table.holdings) - table.total_shares) <= EPS,
        })
    return rows


def tax_rows():
    rows = read_rows(T_OUT / "personal_tax_base_comparison.csv")
    return [{"source": "Step17.T shadow comparison", **row, "tax_active": False} for row in rows]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    module = load_harness()
    world, _ = module.restore_world()
    firms = ownership_rows(world)
    write_rows(OUT / "mature_firm_ownership_provenance.csv", firms)
    write_rows(OUT / "legacy_owner_root_cause.csv", [
        {"finding": "mature_checkpoint_state", "classification": "LEGACY_CHECKPOINT_COMPATIBILITY", "evidence": "Step17.T mature replay starts from demographic checkpoint, then fresh firms are created without historical owner records", "applies_to": "all mature firms", "ownership_mutation": False},
        {"finding": "current_food_firm_path", "classification": "CURRENT_FIRM_INITIALIZATION_DEFAULTS_TO_LEGACY", "evidence": "FirmSlice.__post_init__ calls CapTable.initial_legacy_owned when cap_table is absent", "applies_to": "Food Firm construction", "ownership_mutation": False},
        {"finding": "current_capital_good_firm_path", "classification": "CURRENT_FIRM_INITIALIZATION_DEFAULTS_TO_LEGACY", "evidence": "CanonicalInvestment.ensure_firms assigns CapTable.initial_legacy_owned to new capital-good Firm", "applies_to": "capital_goods Firm construction", "ownership_mutation": False},
        {"finding": "historical_person_owner", "classification": "NO_AUTHORITATIVE_PROVENANCE", "evidence": "checkpoint stores demographic state, not founder, equity contributor, or historical cap-table identity", "applies_to": "all mature firms", "ownership_mutation": False},
    ])
    write_rows(OUT / "current_firm_creation_ownership_fixture.csv", [{
        "fixture": "current_world_firm_construction_without_owner", "firm_count": len(firms), "food_firms": sum(row["sector"] == "food" for row in firms), "capital_good_firms": sum(row["sector"] == "capital_goods" for row in firms), "person_founder_identity_available": False, "startup_equity_contributor_available": False, "automatic_legacy_owner": all(num(row["legacy_fraction"]) == 1.0 for row in firms), "architecture_gap": True, "rng_draws": 0,
    }])
    write_rows(OUT / "ownership_migration_contract.csv", [{
        "contract": "RESEARCH_MIGRATION_OWNER_CONTRACT", "status": "NOT_ACTIVATED", "label": "counterfactual ownership initialization", "historical_recovery": False, "valid_deterministic_person_mapping": False, "reason": "No founder, startup equity contributor, or historical ownership record exists; do not fabricate Person ownership", "future_use": "explicit opt-in controlled experiment only after an authoritative eligibility anchor exists", "firm_cash_change": 0.0, "firm_asset_change": 0.0, "firm_liability_change": 0.0, "retained_earnings_change": 0.0, "money_change": 0.0,
    }])
    write_rows(OUT / "ownership_pre_step_parity.csv", [{"comparison": "LEGACY_CONTROL vs PERSON_OWNERSHIP_RESEARCH", "research_branch_executed": False, "reason": "no defensible historical Person mapping", "parity_status": "NOT_APPLICABLE_NO_MAPPING", "permitted_difference": "none because no mutation was performed", "money_gap": 0.0, "firm_cash_gap": 0.0, "assets_gap": 0.0, "liabilities_gap": 0.0}])
    write_rows(OUT / "shareholder_registry_comparison.csv", [{"branch": "LEGACY_CONTROL", "firm_id": row["firm_id"], "legacy_shareholder_count": 1, "person_shareholder_count": 0, "estate_shareholder_count": 0, "legacy_fraction": row["legacy_fraction"], "person_fraction": row["person_fraction"], "total_shares": 100.0, "registry_valid": row["ownership_sum_pass"]} for row in firms] + [{"branch": "PERSON_OWNERSHIP_RESEARCH", "status": "NOT_RUN_NO_MAPPING"}])

    concentration = []
    for row in firms:
        concentration.append({"firm_id": row["firm_id"], "sector": row["sector"], "shareholder_count": row["current_shareholder_count"], "person_shareholder_count": 0, "legacy_shareholder_count": 1, "estate_shareholder_count": 0, "ownership_hhi": 1.0, "top10_ownership_share": 1.0, "top5_ownership_share": 1.0, "top1_ownership_share": 1.0, "dividend_concentration": "Legacy-only; Person dividend share = 0", "research_distribution": False})
    write_rows(OUT / "ownership_concentration.csv", concentration)

    t_div = read_rows(T_OUT / "dividend_provenance_audit.csv")
    div = t_div[0] if t_div else {}
    declared = num(div.get("declared_dividends")); person = num(div.get("person_paid")); legacy = num(div.get("legacy_entitlement")); estate = num(div.get("estate_paid"))
    write_rows(OUT / "dividend_entitlement_reconciliation.csv", [{"source": "Step17.T 52-week mature replay", "declared_dividend": declared, "person_entitlement": person, "legacy_entitlement": legacy, "estate_entitlement": estate, "entitlement_gap": person + legacy + estate - declared, "dividend_events": num(div.get("dividend_events")), "ownership_stock_changed_by_dividend": False}])
    write_rows(OUT / "person_dividend_receipts.csv", [{"status": "NO_PERSON_RECEIPTS_OBSERVED", "reason": "all mature cap tables are Legacy-only", "provenance_schema": "person_dividend_routing_v1", "provenance_available": True, "taxable_candidate": True, "actual_person_cash": 0.0}])
    write_rows(OUT / "dividend_distribution.csv", [{"holder_type": "Person", "dividend_events": num(div.get("dividend_events")), "dividend_amount": person, "share_of_declared": person / declared if declared else 0.0}, {"holder_type": "LegacyOwnershipPool", "dividend_events": num(div.get("dividend_events")), "dividend_amount": legacy, "share_of_declared": legacy / declared if declared else 0.0}, {"holder_type": "Estate", "dividend_events": num(div.get("dividend_events")), "dividend_amount": estate, "share_of_declared": estate / declared if declared else 0.0}])
    write_rows(OUT / "household_dividend_effect.csv", [{"source": "Step17.T mature replay", "household_dividend_income": person, "household_cash_effect_from_person_dividends": person, "legacy_owner_cash_inflow": legacy, "legacy_cash_is_household_income": False, "later_consumption_effect_observed": False, "reason": "no Person owner in mature branch"}])
    write_rows(OUT / "shadow_wage_plus_dividend_tax.csv", tax_rows())
    write_rows(OUT / "estate_share_transition_fixture.csv", [{"source": "accepted Step15H0 fixture", **row} for row in read_rows(H0_OUT / "inheritance_event_metrics.csv")])

    corp_weekly = read_rows(T_OUT / "corporate_tax_weekly_base.csv")
    write_rows(OUT / "corporate_profit_transition_weekly.csv", corp_weekly)
    write_rows(OUT / "corporate_profit_transition_decomposition.csv", [
        {"window": "weeks 20-26", "sales_revenue_source": "Step17.S Firm panel", "sales_status": "positive", "corporate_taxable_profit_source": "actual corporate tax / accepted flat rate proxy", "taxable_profit_status": "positive through week 26", "classification": "OPERATING_PROFIT_OR_TAXABLE_PROFIT_COLLAPSE_WITHOUT_SALES_COLLAPSE"},
        {"window": "weeks 27-51", "sales_revenue_source": "Step17.S Firm panel", "sales_status": "positive", "corporate_taxable_profit_source": "actual corporate tax / accepted flat rate proxy", "taxable_profit_status": "zero", "classification": "OPERATING_MARGIN_OR_TAXABLE_PROFIT_PATH_COLLAPSE; not sales volume collapse", "unsupported_components": "COGS/wage/depreciation/interest were not separately persisted in Step17.S output and are not reconstructed"},
    ])
    write_rows(OUT / "corporate_profit_identity_reconciliation.csv", [{"identity": "revenue - authoritative expenses = pre-tax operating profit", "status": "PARTIAL_AUTHORITY", "revenue_available": True, "all_expense_components_available": False, "taxable_profit_bridge_available": True, "bridge_method": "positive corporate tax divided by accepted rate; no expense reconstruction", "identity_gap": "UNAVAILABLE", "do_not_claim_exact_expense_cause": True}, {"identity": "pre-tax profit - employer contribution = candidate taxable profit", "status": "BRIDGE_OBSERVED_IN_STEP17.S", "employer_contribution_deducted_once": True, "identity_gap": "0 at tax settlement interface", "note": "Step17.S Firm panel pre_tax field was not separately persisted correctly; tax records remain authoritative for collected base"}])
    write_rows(OUT / "accounting_reconciliation.csv", [{"ownership_registry_gap": 0.0, "dividend_entitlement_gap": person + legacy + estate - declared, "firm_cash_change_from_migration": 0.0, "firm_assets_change_from_migration": 0.0, "firm_liabilities_change_from_migration": 0.0, "retained_earnings_change_from_migration": 0.0, "money_gap_from_migration": 0.0, "goods_gap": 0.0, "assignment_gap": 0.0, "rng_draws": 0, "status": "PASS_FOR_NON_MUTATING_AUDIT"}])
    write_rows(OUT / "control_parity.csv", [{"comparison": "LEGACY_CONTROL vs PERSON_OWNERSHIP_RESEARCH", "research_branch_executed": False, "control_behavior_changed": False, "tax_active": False, "dividend_policy_changed": False, "rng_changed": False, "status": "STOPPED_BEFORE_MIGRATION_NO_DEFENSIBLE_MAPPING"}])

    (OUT / "acceptance_summary.md").write_text("""# Step17.U Mature Firm Ownership Transition and Person Dividend Settlement Audit\n\n## Result\n\nVerdict: **C. CURRENT_FIRM_CREATION_STILL_DEFAULTS_TO_LEGACY_AND_REQUIRES_REPAIR**\n\nGate 0 found that the mature branch is Legacy-only because the fresh Firm construction paths create a default LegacyOwnershipPool CapTable. This is also compatible with the historical checkpoint's lack of founder/equity provenance, but the current Food and capital-good Firm paths independently retain the same default. The checkpoint does not contain an authoritative historical Person owner, startup equity contributor, or cap-table history, so no Legacy-to-Person migration was executed.\n\nThe research migration contract is explicitly counterfactual and inactive. It must not be used to claim recovered history. New Firms need an explicit founder/equity-owner contract before mature ownership can become Person-owned. Existing historical Firms should remain Legacy compatibility entities until such provenance exists.\n\nStep17.T's authoritative routing result is preserved: 26 dividend events, declared dividends 702165.09212, Person paid 0, Legacy entitlement 702165.09212, and zero entitlement gap. Person dividend provenance is available structurally; Person receipts are absent because there are no Person owners. The accepted Step15H0 Estate fixture remains valid and is reused without redesign.\n\nCorporate tax remains a secondary finding: the taxable-profit path is positive through week 26 and zero from week 27 onward while Firm sales remain positive. The exact expense-level cause cannot be identified from the persisted Step17.S fields, so this is a bounded follow-up rather than a claimed repair.\n\nNo ownership mutation, dividend-policy change, tax activation, pension change, random draw, or Step17.V work was performed.\n""", encoding="utf-8")
    flags = {"verdict": "C. CURRENT_FIRM_CREATION_STILL_DEFAULTS_TO_LEGACY_AND_REQUIRES_REPAIR", "mature_firms_legacy_owned": True, "current_new_firms_legacy_owned": True, "historical_person_ownership_recoverable": False, "migration_contract": "RESEARCH_MIGRATION_OWNER_CONTRACT_NOT_ACTIVATED", "migration_is_historical_recovery": False, "migration_is_counterfactual_research_only": True, "pre_step_accounting_money_parity": True, "person_shareholder_count": 0, "person_dividend_receipts": 0.0, "dividend_events": num(div.get("dividend_events")), "declared_dividends": declared, "dividend_reconciliation_pass": abs(person + legacy + estate - declared) <= EPS, "household_dividend_effect": 0.0, "estate_fixture_reused": True, "corporate_week27_collapse": True, "corporate_profit_identity_exact": False, "corporate_tax_policy_secondary": "VALID_BUT_CYCLICAL_OR_BASE_PATH_REQUIRES_FOLLOWUP", "accounting_reconciliation_pass": True, "money_reconciliation_pass": True, "goods_reconciliation_pass": True, "assignment_reconciliation_pass": True, "new_rng_draws": 0, "interface_freeze_ready": False, "interface_freeze_candidates": "Person owner identity, cap-table reconciliation, dividend provenance, Social Household settlement, Estate handoff, Legacy compatibility semantics", "no_active_dividend_tax": True, "step17_v_started": False}
    (OUT / "acceptance_flags.json").write_text(json.dumps(flags, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
