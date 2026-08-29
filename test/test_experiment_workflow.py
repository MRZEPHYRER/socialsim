import json
from pathlib import Path

import pytest

from experiment_runner import BranchTask
from experiment_workflow import ExperimentBranch, ExperimentSpec, task_fingerprint


def test_spec_parses_and_generates_stable_tasks(tmp_path):
    spec = ExperimentSpec.from_mapping({
        "experiment_name": "example",
        "output_root": str(tmp_path / "out"),
        "steps": 13,
        "seed": 42,
        "branches": [
            {"branch_name": "CONTROL"},
            {"branch_name": "TREATMENT_A", "config_overrides": {"world": {"diagnostics_mode": "compact"}}},
        ],
    })
    tasks = spec.tasks()
    assert [task.branch_name for task in tasks] == ["CONTROL", "TREATMENT_A"]
    assert tasks[0].output_dir != tasks[1].output_dir
    assert tasks[1].config_overrides["world"]["diagnostics_mode"] == "compact"
    assert task_fingerprint(tasks[0], None) == task_fingerprint(tasks[0], None)


def test_unknown_override_rejected_before_task_generation(tmp_path):
    spec = ExperimentSpec.from_mapping({
        "experiment_name": "typo",
        "output_root": str(tmp_path / "out"),
        "steps": 1,
        "branches": [{"branch_name": "CONTROL", "config_overrides": {"economy.config": {"PENSION_REPLACEMENT_RTAIO": 1}}}],
    })
    with pytest.raises(ValueError, match="unknown economy.config"):
        spec.tasks()


def test_unsafe_branch_and_duplicate_names_rejected(tmp_path):
    for branches, message in [
        ([{"branch_name": "CONTROL"}, {"branch_name": "CONTROL"}], "unique"),
        ([{"branch_name": "../escape"}], "single safe path"),
    ]:
        spec = ExperimentSpec.from_mapping({
            "experiment_name": "invalid",
            "output_root": str(tmp_path / "out"),
            "steps": 1,
            "branches": branches,
        })
        with pytest.raises(ValueError, match=message):
            spec.tasks()


def test_direct_branch_task_and_spec_task_have_same_contract(tmp_path):
    direct = BranchTask(
        branch_name="CONTROL",
        output_dir=str(tmp_path / "out" / "CONTROL"),
        steps=13,
        seed=42,
        observability_mode="RESEARCH_FAST",
    )
    spec = ExperimentSpec(
        experiment_name="parity",
        output_root=str(tmp_path / "out"),
        steps=13,
        seed=42,
        branches=(ExperimentBranch("CONTROL"),),
    )
    generated = spec.tasks()[0]
    assert direct.to_dict() == generated.to_dict()


def test_example_spec_is_valid_json():
    path = Path(__file__).parent / "examples" / "step16_p2_2_example.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    spec = ExperimentSpec.from_mapping(payload, path.parent)
    assert [branch.branch_name for branch in spec.branches] == ["CONTROL", "TREATMENT_A", "TREATMENT_B", "TREATMENT_C"]
