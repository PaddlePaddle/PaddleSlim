import sys
sys.path.append("../")
import unittest
import numpy as np
from paddleslim.dygraph.prune.pruning_plan import PruningPlan, PruningMask


class TestPruningPlan(unittest.TestCase):
    def testAdd(self):
        plan = PruningPlan()
        mask = PruningMask([0], [0, 1, 1], 0.33)
        plan.add("a", mask)
        mask = PruningMask([0], [0, 1, 0], 0.33)
        plan.add("a", mask)
        a_mask = plan.masks["a"]
        self.assertTrue(len(a_mask) == 1)
        self.assertTrue(a_mask[0].mask == [0, 1, 0])
        self.assertTrue(a_mask[0].dims == [0])


if __name__ == '__main__':
    unittest.main()
