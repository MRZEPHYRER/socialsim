from pathlib import Path
import json

import pytest

import main


def test_demo_profile_is_step15_analysis_ready():
    profile = main.FULL_DEMO_PROFILE
    assert profile["profile_name"] == "STEP15_ANALYSIS_READY_DEMO"
    overrides = profile["runtime_overrides"]
    assert overrides["MULTISECTOR_FOUNDATION_ENABLED"] is True
    assert overrides["CANONICAL_INVESTMENT_ENABLED"] is True
    assert overrides["CAPITAL_CAPACITY_RUNTIME_ENABLED"] is True
    assert overrides["CAPITAL_LIFECYCLE_ENABLED"] is True
    assert overrides["CAPITAL_GOOD_CUSTOMER_ADVANCE_ENABLED"] is True
    assert overrides["CAPITAL_GOOD_FIXTURE_PRODUCTIVITY_PER_LABOR"] == 3.0
    assert overrides["CAPITAL_GOOD_INVESTMENT_REVIEW_INTERVAL_WEEKS"] == 13
    assert overrides["ADULT_SETTLEMENT_LABOR_ELIGIBILITY_ENABLED"] is True
    assert overrides["DETERMINISTIC_FIRM_REVIEW_PHASE_STAGGERING_ENABLED"] is False


def test_demo_resolves_exact_accepted_reference_directory():
    resolved = main.resolve_demo_analysis_dir()
    assert resolved.name == "step15_final_canonical_integrated_v2"
    assert resolved == Path(main.__file__).resolve().parent / main.REFERENCE_DEMO_RELATIVE_DIR


def test_fresh_demo_handoff_fixture_contains_step15_domains():
    run_dir = Path("test/output/step15_final_demo_runtime_alignment_fixture2")
    if not run_dir.exists():
        pytest.skip("fresh demo fixture is not present")
    flags = json.loads(
        Path("test/output/step15_final_demo_runtime_alignment/acceptance_flags.json")
        .read_text(encoding="utf-8-sig")
    )
    assert flags["profile"] == "STEP15_ANALYSIS_READY_DEMO"
    assert flags["food_firms"] == 5
    assert flags["capital_good_firms"] == 1
    assert set(flags["sectors"]) == {"food", "capital_goods"}
    assert flags["investment_orders"] > 0
    assert flags["capital_assets"] > 0
    assert flags["customer_advance_events"] > 0
    assert flags["demographic_rows"] > 1
