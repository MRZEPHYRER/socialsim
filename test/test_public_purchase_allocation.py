import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from economy.multi_firm import allocate_inventory_constrained_purchase


class PublicPurchaseAllocationTests(unittest.TestCase):
    def test_zero_inventory_donor_reallocates_remainder(self):
        actual = allocate_inventory_constrained_purchase(10.0, [0.5, 0.5], [0.0, 20.0])
        self.assertAlmostEqual(sum(actual), 10.0)
        self.assertEqual(actual[0], 0.0)
        self.assertAlmostEqual(actual[1], 10.0)

    def test_multiple_capped_donors_redistribute_iteratively(self):
        actual = allocate_inventory_constrained_purchase(
            10.0,
            [0.5, 0.3, 0.2],
            [1.0, 1.0, 20.0],
        )
        self.assertAlmostEqual(sum(actual), 10.0)
        self.assertLessEqual(actual[0], 1.0)
        self.assertLessEqual(actual[1], 1.0)
        self.assertAlmostEqual(actual[2], 8.0)

    def test_industry_shortage_is_reported_by_executed_total(self):
        actual = allocate_inventory_constrained_purchase(10.0, [0.5, 0.5], [2.0, 3.0])
        self.assertAlmostEqual(sum(actual), 5.0)
        self.assertAlmostEqual(sum(actual), min(10.0, 5.0))

    def test_single_firm_is_capped_by_its_inventory(self):
        actual = allocate_inventory_constrained_purchase(10.0, [1.0], [4.0])
        self.assertEqual(actual, [4.0])

    def test_all_firms_have_stock_preserves_target_allocation(self):
        actual = allocate_inventory_constrained_purchase(
            10.0,
            [0.2, 0.3, 0.5],
            [20.0, 20.0, 20.0],
        )
        self.assertAlmostEqual(actual[0], 2.0)
        self.assertAlmostEqual(actual[1], 3.0)
        self.assertAlmostEqual(actual[2], 5.0)


if __name__ == "__main__":
    unittest.main()
