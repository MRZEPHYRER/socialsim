"""Generate the Step 12.R3 analysis/reporting surface inventory."""

import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "test" / "output" / "step12_R3_analysis_audit"


def functions(path):
    tree = ast.parse(path.read_text(encoding="utf-8-sig"))
    return [node.name for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]


def main():
    current = sorted((ROOT / "analysis").glob("*.py")) + [ROOT / "main.py", ROOT / "world.py", ROOT / "scenarios.py", ROOT / "checkpoint.py"]
    legacy_tests = sorted(
        path for path in (ROOT / "test").glob("*.py")
        if any(token in path.name.lower() for token in ("analy", "report", "regression", "diagnostic", "summary", "scan", "sweep"))
    )
    payload = {
        "time_contract": "1 step = 1 week; 52 steps = 1 model year",
        "current_analysis_surface": [
            {"file": str(path.relative_to(ROOT)), "functions": functions(path)}
            for path in current
        ],
        "legacy_test_analysis_surface": [
            {"file": str(path.relative_to(ROOT)), "classification": "LEGACY MIXED-TIME REFERENCE; preserved, not rewritten"}
            for path in legacy_tests
        ],
        "rolling_windows": {
            "5": "five-week firm production review interval",
            "12": "twelve-week price evaluation window",
            "15": "fifteen-week target inventory coverage",
            "52": "one-model-year analysis/demographic window",
            "80": "legacy food-price cash window preserved as eighty weekly observations",
            "200": "diagnostics/reporting window of 200 weeks",
            "500": "steady-state diagnostics window of 500 weeks",
        },
        "resolved_issues": [
            "Replaced raw Step/Year plot labels with Simulation week.",
            "Reclassified legacy GDP fields as weekly nominal output value.",
            "Labeled births/population and deaths/population as weekly realized event rates.",
            "Separated planning, realized transaction, and legacy representative prices.",
            "Made active_households the primary macro household count and retained raw object counts for diagnostics.",
            "Documented six months as a 26-week target-wealth reserve.",
            "Documented 150000 as an industry-wide fixed credit buffer allocated by productive-capacity share.",
            "Documented principal repayment as a 35% weekly cash-constrained target/cap.",
            "Added absolute global step versus elapsed-week metadata for continuation analysis.",
        ],
        "critical_unresolved_mixed_time_semantics": [],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "analysis_surface_inventory.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(OUT / "analysis_surface_inventory.json")


if __name__ == "__main__":
    main()
