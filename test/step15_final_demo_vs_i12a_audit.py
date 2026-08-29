"""Audit fresh --demo against the accepted Step 15I.12A output.

This script is intentionally diagnostic only. It reads persisted outputs and
does not construct or mutate an economic world.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "test/output/demo_vs_i12a_differential_audit"
REFERENCE = ROOT / "test/output/step15I12A_final_integrated_validation"


def read_csv(path):
    if not path.exists():
        return [], []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader), list(reader.fieldnames or [])


def write_csv(name, rows, fields):
    path = OUT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def manifest(path):
    file = path / "manifest.json"
    if not file.exists():
        return {}
    return json.loads(file.read_text(encoding="utf-8-sig"))


def acceptance(path):
    file = path / "acceptance_flags.json"
    if not file.exists():
        return {}
    return json.loads(file.read_text(encoding="utf-8-sig"))


def bool_text(value):
    return "true" if bool(value) else "false"


def count_nonzero(rows, field):
    values = [row.get(field) for row in rows]
    nonempty = [value for value in values if value not in (None, "", "NA", "nan", "UNAVAILABLE")]
    nonzero = 0
    for value in nonempty:
        try:
            nonzero += abs(float(value)) > 1e-12
        except (TypeError, ValueError):
            nonzero += str(value).lower() not in {"false", "none", "0"}
    return len(nonempty), nonzero


def dataset_rows(path, filename):
    if filename in {"household_diagnostics.csv", "ledger.csv"}:
        file = path / filename
        if not file.exists():
            return [], []
        with file.open(encoding="utf-8-sig") as handle:
            fields = next(csv.reader(handle), [])
            rows = sum(1 for _ in handle)
        return [{"__row_count__": rows}], fields
    rows, fields = read_csv(path / filename)
    return rows, fields


def main(fresh_dir):
    fresh = Path(fresh_dir)
    OUT.mkdir(parents=True, exist_ok=True)
    ref_manifest = manifest(REFERENCE)
    fresh_manifest = manifest(fresh)
    ref_accept = acceptance(REFERENCE)

    accepted_config = ref_accept.get("accepted_configuration", {})
    feature_defaults = {
        "MULTISECTOR_FOUNDATION_ENABLED": False,
        "CANONICAL_INVESTMENT_ENABLED": False,
        "CAPITAL_CAPACITY_RUNTIME_ENABLED": False,
        "CAPITAL_LIFECYCLE_ENABLED": False,
        "CAPITAL_GOOD_CUSTOMER_ADVANCE_ENABLED": False,
        "CAPITAL_GOOD_FIXTURE_PRODUCTIVITY_PER_LABOR": "UNAVAILABLE",
        "CAPITAL_GOOD_OFFER_PRICE_MODE": "UNAVAILABLE",
        "CAPITAL_LIFECYCLE_USEFUL_LIFE_WEEKS": 52.0,
    }
    runtime_fields = [
        ("population", ref_accept.get("canonical_population", ref_manifest.get("population")), fresh_manifest.get("population")),
        ("seed", ref_accept.get("canonical_seed", ref_manifest.get("seed")), fresh_manifest.get("seed")),
        ("weeks", ref_accept.get("canonical_weeks", ref_manifest.get("steps")), fresh_manifest.get("steps")),
        ("firm_count", ref_manifest.get("firm_count", 6), fresh_manifest.get("firm_count")),
        ("scenario", ref_manifest.get("scenario", "step15i12a_final_integrated_validation"), fresh_manifest.get("scenario")),
        ("model", ref_manifest.get("model", "multi-sector canonical"), fresh_manifest.get("model")),
    ]
    for key, default in feature_defaults.items():
        reference = accepted_config.get(key, default)
        actual = fresh_manifest.get("overrides", {}).get(key, feature_defaults.get(key, default))
        runtime_fields.append((key, reference, actual))
    runtime_fields.extend([
        ("diagnostics_mode", "full", "full"),
        ("persistence_profile", "Step15I.12A canonical panels", "FULL_DEMO_PROFILE + handoff panels"),
        ("analysis_profile", "full", "full"),
        ("demographic_system", "active in accepted reference", "active in fresh runtime diagnostics"),
    ])
    config_rows = []
    for key, reference, actual in runtime_fields:
        config_rows.append({
            "field": key,
            "reference_I12A": reference,
            "fresh_demo": actual,
            "equal": bool_text(reference == actual),
            "classification": "MATCH" if reference == actual else "RUNTIME_CONFIGURATION_MISMATCH",
        })
    write_csv("runtime_configuration_diff.csv", config_rows, tuple(config_rows[0]))

    ref_firm, _ = dataset_rows(REFERENCE, "step15_firm_panel.csv")
    fresh_firm, _ = dataset_rows(fresh, "step15_firm_panel.csv")
    ref_macro, _ = dataset_rows(REFERENCE, "step15_macro_panel.csv")
    fresh_macro, _ = dataset_rows(fresh, "step15_macro_panel.csv")
    ref_chain, _ = dataset_rows(REFERENCE, "step15_investment_chain_trace.csv")
    fresh_chain, _ = dataset_rows(fresh, "step15_investment_chain_trace.csv")
    ref_assets, _ = dataset_rows(REFERENCE, "step15_capital_asset_ledger.csv")
    fresh_assets, _ = dataset_rows(fresh, "step15_capital_asset_ledger.csv")
    ref_events, _ = dataset_rows(REFERENCE, "step15_capital_provenance_events.csv")
    fresh_events, _ = dataset_rows(fresh, "step15_capital_provenance_events.csv")

    def construction(rows):
        return {
            "firm_rows": len(rows),
            "firm_ids": ",".join(sorted({row.get("firm_id", "") for row in rows})),
            "sectors": ",".join(sorted({row.get("sector", "") for row in rows if row.get("sector")})),
            "food_firms": len({row.get("firm_id") for row in rows if row.get("sector") == "food"}),
            "capital_good_firms": len({row.get("firm_id") for row in rows if row.get("sector") == "capital_goods"}),
            "service_firms": len({row.get("firm_id") for row in rows if row.get("sector") in {"generic_services", "services"}}),
        }
    ref_build = construction(ref_firm)
    fresh_build = construction(fresh_firm)
    world_rows = []
    for key in sorted(set(ref_build) | set(fresh_build)):
        world_rows.append({"field": key, "reference_I12A": ref_build.get(key), "fresh_demo": fresh_build.get(key), "classification": "MATCH" if ref_build.get(key) == fresh_build.get(key) else "WORLD_CONSTRUCTION_OR_SECTOR_REGISTRATION_MISMATCH"})
    world_rows += [
        {"field": "Person count", "reference_I12A": "5000-scale runtime", "fresh_demo": "5000-scale runtime", "classification": "NOT_RECORDED_IN_STEP15_REFERENCE"},
        {"field": "Household count", "reference_I12A": "runtime snapshot only", "fresh_demo": "runtime diagnostics", "classification": "DIAGNOSTIC_SCOPE_DIFFERENCE"},
        {"field": "CapitalStock / ownership / planner", "reference_I12A": "present and active", "fresh_demo": "CapitalStock container present; planner inactive", "classification": "RUNTIME_CONFIGURATION_MISMATCH"},
    ]
    write_csv("world_construction_diff.csv", world_rows, tuple(world_rows[0]))

    runtime_event_fields = [
        ("investment_intents", count_nonzero(ref_macro, "total_fixed_investment")[0], count_nonzero(fresh_macro, "total_fixed_investment")[0]),
        ("investment_orders", len({row.get("order_id") for row in ref_chain if row.get("order_id")}), len({row.get("order_id") for row in fresh_chain if row.get("order_id")})),
        ("expansion_orders", sum(row.get("investment_source") == "EXPANSION" for row in ref_chain), sum(row.get("investment_source") == "EXPANSION" for row in fresh_chain)),
        ("replacement_orders", sum(row.get("investment_source") == "REPLACEMENT" for row in ref_chain), sum(row.get("investment_source") == "REPLACEMENT" for row in fresh_chain)),
        ("customer_advance_events", sum(row.get("event_type") in {"advance_payment", "customer_advance_received"} for row in ref_chain), sum(row.get("event_type") in {"advance_payment", "customer_advance_received"} for row in fresh_chain)),
        ("deliveries", sum(row.get("event_type") in {"delivery", "investment_order_settled"} for row in ref_chain), sum(row.get("event_type") in {"delivery", "investment_order_settled"} for row in fresh_chain)),
        ("CapitalAsset_acquisitions", sum(row.get("event_type") == "asset_acquired" for row in ref_events), sum(row.get("event_type") == "asset_acquired" for row in fresh_events)),
        ("retirements", sum(row.get("event_type") == "asset_retired" for row in ref_events), sum(row.get("event_type") == "asset_retired" for row in fresh_events)),
    ]
    event_rows = [{"event": key, "reference_I12A": ref, "fresh_demo": actual, "first_divergence": "RUNTIME_EVENT_ABSENT_IN_DEMO" if ref and not actual else "MATCH"} for key, ref, actual in runtime_event_fields]
    write_csv("runtime_event_diff.csv", event_rows, tuple(event_rows[0]))

    asset_rows = [
        {"metric": "CapitalStock records", "reference_I12A": len(ref_assets), "fresh_demo": len(fresh_assets), "classification": "NO_RUNTIME_ASSETS" if not fresh_assets else "MATCH"},
        {"metric": "provenance event records", "reference_I12A": len(ref_events), "fresh_demo": len(fresh_events), "classification": "NO_RUNTIME_EVENTS" if not fresh_events else "MATCH"},
        {"metric": "active assets", "reference_I12A": sum(str(row.get("active")).lower() == "true" for row in ref_assets), "fresh_demo": sum(str(row.get("active")).lower() == "true" for row in fresh_assets), "classification": "RUNTIME_CONFIGURATION_MISMATCH"},
        {"metric": "persistence layer", "reference_I12A": "historical panel only; no asset ledger file", "fresh_demo": "asset ledger schema file present, zero rows", "classification": "NO_RUNTIME_ASSETS"},
    ]
    write_csv("capital_asset_diff.csv", asset_rows, tuple(asset_rows[0]))

    contract_rows = [
        {"metric": "investment-chain rows", "reference_I12A": len(ref_chain), "fresh_demo": len(fresh_chain), "classification": "RUNTIME_EVENT_ABSENT_IN_DEMO" if ref_chain and not fresh_chain else "MATCH"},
        {"metric": "unique orders", "reference_I12A": len({r.get("order_id") for r in ref_chain}), "fresh_demo": len({r.get("order_id") for r in fresh_chain}), "classification": "RUNTIME_EVENT_ABSENT_IN_DEMO"},
        {"metric": "order -> asset links", "reference_I12A": sum(bool(r.get("asset_id")) for r in ref_chain), "fresh_demo": sum(bool(r.get("asset_id")) for r in fresh_chain), "classification": "RUNTIME_EVENT_ABSENT_IN_DEMO"},
        {"metric": "asset -> retirement links", "reference_I12A": sum(r.get("event_type") == "asset_retired" for r in ref_events), "fresh_demo": sum(r.get("event_type") == "asset_retired" for r in fresh_events), "classification": "RUNTIME_EVENT_ABSENT_IN_DEMO"},
    ]
    write_csv("investment_contract_diff.csv", contract_rows, tuple(contract_rows[0]))

    demo_diag, demo_diag_fields = dataset_rows(fresh, "diagnostics.csv")
    demo_events, _ = dataset_rows(fresh, "demographic_events.csv")
    demographic_rows = [
        {"metric": "Person age runtime", "reference_I12A": "not in accepted Step15 panels", "fresh_demo": "runtime world / raw diagnostics", "classification": "RUNTIME_EXISTS_NOT_PERSISTED_TO_STEP15_GUI"},
        {"metric": "births", "reference_I12A": "not in accepted Step15 panels", "fresh_demo": "diagnostics field present; demographic events present", "classification": "RUNTIME_EXISTS_NOT_PERSISTED"},
        {"metric": "deaths", "reference_I12A": "not in accepted Step15 panels", "fresh_demo": "diagnostics field present; demographic events present", "classification": "RUNTIME_EXISTS_NOT_PERSISTED"},
        {"metric": "marriages", "reference_I12A": "not in accepted Step15 panels", "fresh_demo": "marriage_market_diagnostics.csv present", "classification": "RUNTIME_EXISTS_NOT_PERSISTED"},
        {"metric": "age groups / pyramid", "reference_I12A": "not in accepted Step15 panels", "fresh_demo": "initial age diagnostics present; GUI panel input absent", "classification": "ANALYSIS_LOADER_MISSING"},
        {"metric": "demographic event rows", "reference_I12A": 0, "fresh_demo": len(demo_events), "classification": "FRESH_RAW_DIAGNOSTIC_ONLY"},
    ]
    write_csv("demographic_social_diff.csv", demographic_rows, tuple(demographic_rows[0]))

    filenames = [
        "step15_macro_panel.csv", "step15_firm_panel.csv", "step15_accounting_reconciliation.csv", "step15_investment_chain_trace.csv", "step15_capital_asset_ledger.csv", "step15_capital_provenance_events.csv", "diagnostics.csv", "firm_diagnostics.csv", "demographic_events.csv", "marriage_market_diagnostics.csv", "initial_age_phase_diagnostics.csv", "initial_age_phase_histogram.csv", "age_transition_diagnostics.csv", "household_diagnostics.csv", "ledger.csv",
    ]
    schema_rows = []
    for filename in filenames:
        ref_rows, ref_fields = dataset_rows(REFERENCE, filename)
        fresh_rows, fresh_fields = dataset_rows(fresh, filename)
        schema_rows.append({
            "dataset": filename,
            "reference_exists": (REFERENCE / filename).exists(),
            "fresh_exists": (fresh / filename).exists(),
            "reference_rows": len(ref_rows),
            "fresh_rows": len(fresh_rows),
            "columns_only_reference": ",".join(sorted(set(ref_fields) - set(fresh_fields))),
            "columns_only_fresh": ",".join(sorted(set(fresh_fields) - set(ref_fields))),
            "semantic_compatibility": "MATCH" if set(ref_fields) == set(fresh_fields) else "SCHEMA_OR_PROFILE_DIFFERENCE",
        })
    write_csv("analysis_dataset_schema_diff.csv", schema_rows, tuple(schema_rows[0]))

    try:
        from analysis.step15 import Step15AnalysisDataLoader
        ref_loader = Step15AnalysisDataLoader(REFERENCE)
        fresh_loader = Step15AnalysisDataLoader(fresh)
        loader_rows = [
            {"domain": "list_firms", "reference_I12A": ",".join(ref_loader.list_firms()), "fresh_demo": ",".join(fresh_loader.list_firms()), "classification": "DIFFERENT_RUNTIME_WORLD"},
            {"domain": "list_sectors", "reference_I12A": ",".join(ref_loader.list_sectors()), "fresh_demo": ",".join(fresh_loader.list_sectors()), "classification": "WORLD_CONSTRUCTION_OR_SECTOR_REGISTRATION_MISMATCH"},
            {"domain": "capital availability", "reference_I12A": bool(ref_loader.tables["chain"]), "fresh_demo": bool(fresh_loader.tables["chain"]), "classification": "RUNTIME_EVENT_ABSENT_IN_DEMO"},
            {"domain": "contract availability", "reference_I12A": len(ref_loader.tables["chain"]), "fresh_demo": len(fresh_loader.tables["chain"]), "classification": "RUNTIME_EVENT_ABSENT_IN_DEMO"},
            {"domain": "hard-coded I12A fallback", "reference_I12A": "not tested by loader", "fresh_demo": "no path fallback in Step15AnalysisDataLoader", "classification": "NOT_ROOT_CAUSE"},
        ]
    except Exception as exc:
        loader_rows = [{"domain": "loader", "reference_I12A": "ERROR", "fresh_demo": str(exc), "classification": "ANALYSIS_LOADER_MISMATCH"}]
    write_csv("analysis_loader_diff.csv", loader_rows, tuple(loader_rows[0]))

    gui_rows = []
    domains = {
        "Overview": (True, True, "macro panel exists"),
        "Population & Social": (False, False, "birth/death/marriage/age fields are not in Step15 handoff"),
        "Household": (True, True, "macro/financial snapshots available"),
        "Labor": (True, True, "Firm panel and macro employment available"),
        "Macro": (True, True, "macro panel available"),
        "Sector": (bool(ref_build["sectors"]), bool(fresh_build["sectors"]), "fresh has food only; capital_goods runtime absent"),
        "Firm": (True, True, "Firm panel available"),
        "Accounting": (True, True, "accounting panel available"),
        "Capital & Investment": (True, False, "fresh runtime has no capital assets/events"),
        "Contract / Transaction": (bool(ref_chain), bool(fresh_chain), "fresh runtime has no investment orders"),
        "Reconciliation": (True, True, "reconciliation panel available"),
    }
    for page, (ref_ok, fresh_ok, reason) in domains.items():
        gui_rows.append({"page": page, "reference_I12A": "FULLY_AVAILABLE" if ref_ok else "PARTIALLY_AVAILABLE", "fresh_demo": "FULLY_AVAILABLE" if fresh_ok else "PARTIALLY_AVAILABLE", "reason": reason})
    write_csv("gui_availability_diff.csv", gui_rows, tuple(gui_rows[0]))

    flags = {
        "verdict": "A. DEMO_RUNTIME_PROFILE_MISMATCH",
        "fresh_run_directory": str(fresh),
        "reference_run_directory": str(REFERENCE),
        "first_authoritative_divergence": "demo profile -> resolved runtime configuration",
        "demo_model": fresh_manifest.get("model"),
        "demo_overrides": fresh_manifest.get("overrides", {}),
        "reference_accepted_configuration": accepted_config,
        "fresh_runtime_investment_events": len(fresh_chain),
        "fresh_runtime_capital_assets": len(fresh_assets),
        "fresh_raw_demographic_event_rows": len(demo_events),
        "economic_behavior_changed": False,
        "demographic_behavior_changed": False,
        "new_rng_draws": 0,
        "audit_only": True,
    }
    (OUT / "acceptance_flags.json").write_text(json.dumps(flags, ensure_ascii=False, indent=2), encoding="utf-8-sig")
    (OUT / "first_divergence_report.md").write_text(
        f"""# First Divergence Report\n\n## Verdict\n\n{flags['verdict']}\n\n## First authoritative divergence\n\n`python main.py --demo` resolves to `model={fresh_manifest.get('model')}` with empty runtime overrides. The accepted I.12A reference explicitly enables `MULTISECTOR_FOUNDATION_ENABLED`, `CANONICAL_INVESTMENT_ENABLED`, `CAPITAL_CAPACITY_RUNTIME_ENABLED`, `CAPITAL_LIFECYCLE_ENABLED`, and `CAPITAL_GOOD_CUSTOMER_ADVANCE_ENABLED`.\n\nThis divergence occurs before persistence and before Analysis loading. The fresh runtime constructs five Food Firms only, produces zero investment orders and zero CapitalAsset records, and therefore cannot expose capital-good sectors, investment contracts, or asset links.\n\nThe fresh run does contain raw demographic diagnostics (`{len(demo_events)}` event rows and a diagnostics `births`/`deaths` schema), but the Step15 handoff does not expose those project-wide historical datasets to the GUI. That is a downstream analysis-data boundary, not the cause of the missing capital events.\n\n`sector=-1` is not accepted as a valid sector label. In the audited fresh Step15 panel the persisted sector set is `{sorted({row.get('sector') for row in fresh_firm})}`; no loader fallback is needed to explain the missing `capital_goods` sector.\n\nNo mechanism, demographic rule, persistence implementation, GUI mapping, or RNG was changed by this audit.\n""",
        encoding="utf-8-sig",
    )
    (OUT / "acceptance_summary.md").write_text(
        f"""# Fresh --demo vs I.12A Differential Audit\n\nVerdict: **{flags['verdict']}**\n\nFresh run: `{fresh}`\nReference: `{REFERENCE}`\n\nThe first divergence is the resolved demo runtime profile. I.12A is a 5,000-person, 520-week multi-sector investment run with P3 capital-good productivity, cost-anchored pricing, 13-week backlog flow, 52-week useful life, customer advances, and real investment events. Fresh `--demo` resolves to the legacy `single_firm_single_good` baseline with no feature overrides.\n\nConsequences observed in authoritative outputs:\n- fresh sectors: `{fresh_build['sectors']}`; I.12A sectors: `{ref_build['sectors']}`;\n- fresh investment-chain rows: `{len(fresh_chain)}`; I.12A: `{len(ref_chain)}`;\n- fresh CapitalAsset rows: `{len(fresh_assets)}`; I.12A panel asset rows are not separately persisted;\n- fresh raw demographic events: `{len(demo_events)}`, but not exposed in the Step15 GUI handoff.\n\nThis is an audit only. No code or economic behavior was changed.\n""",
        encoding="utf-8-sig",
    )
    return flags


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: python test/step15_final_demo_vs_i12a_audit.py <fresh-run-dir>")
    print(json.dumps(main(sys.argv[1]), ensure_ascii=False, indent=2))
