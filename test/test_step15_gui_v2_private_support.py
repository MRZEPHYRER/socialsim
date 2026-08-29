from pathlib import Path

import pytest

from analysis.gui_v2.data import CanonicalDataStore
from analysis.gui_v2.query import AnalysisV2Query

ROOT = Path(__file__).resolve().parents[1]
SUPPORT = ROOT / "test/output/step15_mature_genealogy_private_support_experiment"


def test_private_support_requires_explicit_dataset_selection():
    default = CanonicalDataStore()
    support = CanonicalDataStore(run_dir=SUPPORT)
    assert default.dataset_kind == "canonical"
    assert support.dataset_kind == "mature_private_support"
    assert support.base_run_dir == default.run_dir


def test_private_support_query_matches_persisted_event_summary():
    query = AnalysisV2Query(CanonicalDataStore(run_dir=SUPPORT))
    series = query.get_private_support_series()
    events = query.get_private_support_summary()["events"]
    treatment = events[events.branch.eq("treatment")].iloc[0]
    assert len(series) == 1040
    assert set(series.branch) == {"control", "treatment"}
    assert treatment.transfer_events == pytest.approx(829384)
    assert treatment.total_value == pytest.approx(1376298.3614281542)


def test_default_store_hashes_remain_unchanged_after_read_only_queries():
    store = CanonicalDataStore()
    before = store.hashes()
    query = AnalysisV2Query(store)
    query.get_macro_series(["household_income", "household_consumption"])
    assert before == store.hashes()