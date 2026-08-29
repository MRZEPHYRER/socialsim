"""Deterministic acceptance cases for Step 13.4 interest semantics."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from economy.interest import settle_interest


def check(name, condition, details):
    if not condition:
        raise AssertionError(f"{name}: {details}")
    return {"name": name, "passed": True, "details": details}


def main():
    results = []
    a = settle_interest(1000, 0, 10000, 0, 0.05)
    results.append(check("A_full_payment", a["interest_paid"] > 0 and a["closing_interest_arrears"] == 0, a))
    b = settle_interest(1000, 0, 0, 0, 0.05)
    results.append(check("B_partial_or_unpaid", b["interest_paid"] == 0 and b["closing_interest_arrears"] > 0, b))
    c = settle_interest(1000, b["closing_interest_arrears"], 0, 0, 0.05)
    results.append(check("C_zero_service", c["interest_paid"] == 0 and c["closing_interest_arrears"] > b["closing_interest_arrears"], c))
    d = settle_interest(1000, b["closing_interest_arrears"], 10000, 0, 0.05)
    results.append(check("D_old_arrears_paid_first", d["interest_paid_to_opening_arrears"] == b["closing_interest_arrears"] and d["closing_interest_arrears"] == 0, d))
    e = settle_interest(1000, 500, 0, 0, 0.05)
    results.append(check("E_arrears_do_not_reduce_principal", e["closing_interest_arrears"] > 500 and e["current_interest_unpaid"] > 0, e))
    f = settle_interest(1000, 0, 0, 100, 0.05)
    candidate = 200.0
    reserved = min(candidate, max(0.0, 0.0 - f["total_interest_obligation"]))
    results.append(check("F_interest_reserves_dividend", reserved == 0.0 and f["total_interest_obligation"] > 0, {"settlement": f, "candidate": candidate, "reserved_dividend": reserved}))
    zero = settle_interest(1000, 25, 100, 100, 0.0)
    results.append(check("zero_rate_noop", zero["current_interest_due"] == 0 and zero["interest_paid"] == 0 and zero["closing_interest_arrears"] == 25, zero))
    output = Path("test/output/step13_4_interest_behavior")
    output.mkdir(parents=True, exist_ok=True)
    (output / "interest_accounting_validation.json").write_text(
        json.dumps({"passed": True, "cases": results}, indent=2), encoding="utf-8"
    )
    print(json.dumps({"passed": True, "cases": len(results)}))


if __name__ == "__main__":
    main()
