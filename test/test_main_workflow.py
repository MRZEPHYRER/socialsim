from pathlib import Path
from types import SimpleNamespace

import main


def _args(**overrides):
    values = dict(main.FULL_DEMO_PROFILE)
    values.update({
        "demo": True,
        "run_only": False,
        "output_dir": None,
        "save_checkpoint": None,
    })
    values.update(overrides)
    return SimpleNamespace(**values)


def test_full_demo_profile_is_centralized(monkeypatch):
    monkeypatch.setattr(main.sys, "argv", ["main.py", "--demo"])
    args = main.apply_full_demo_profile(_args())
    assert args.population == 5000
    assert args.steps == 2000
    assert args.persist_diagnostics is True
    assert args.analysis_profile == "full"
    assert args.plot_mode == "save-only"
    assert args._auto_launch_gui is True
    assert args._demo_checkpoint_default is True


def test_run_only_disables_gui_and_output_is_versioned(tmp_path):
    args = _args(output_dir=str(tmp_path / "demo"), run_only=True)
    args._demo_output_dir = True
    Path(args.output_dir).mkdir()
    first = main.experiment_output_dir(args, 2000)
    assert first.endswith("_run1")
    Path(first).mkdir()
    second = main.experiment_output_dir(args, 2000)
    assert second.endswith("_run2")


def test_gui_handoff_writes_actual_run_panels(tmp_path):
    from analysis.gui.handoff import write_gui_compatibility_panels

    context = SimpleNamespace(
        world=SimpleNamespace(seed=42, scenario_name="baseline"),
        firm_diagnostics=[{"step": 0, "firm_id": 0, "sector": "food"}],
        tables={
            "weekly_macro": [{"step": 0, "population": 10, "employment": 4, "consumption": 2, "executed_payroll": 3}],
            "weekly_firm": [{"step": 0, "firm_id": 0, "employee_count": 4, "revenue": 2}],
            "accounting_weekly": [],
        },
    )
    output = write_gui_compatibility_panels(context, tmp_path)
    assert output == tmp_path
    assert (tmp_path / "step15_macro_panel.csv").exists()
    assert (tmp_path / "step15_firm_panel.csv").exists()
