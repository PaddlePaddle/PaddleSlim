import sys
sys.path.append("../../")
import unittest
import paddle
import numpy as np
from paddleslim import UnstructuredPrunerGMP
from paddle.vision.models import mobilenet_v1


class TestUnstructuredPruner(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestUnstructuredPruner, self).__init__(*args, **kwargs)
        paddle.disable_static()
        self._gen_model()

    def _gen_model(self):
        self.net = mobilenet_v1(num_classes=10, pretrained=False)
        configs = {
            'stable_iterations': 0,
            'pruning_iterations': 1000,
            'tunning_iterations': 1000,
            'resume_iteration': 0,
            'pruning_steps': 20,
            'initial_ratio': 0.05,
        }
        self.pruner = UnstructuredPrunerGMP(
            self.net, mode='ratio', ratio=0.98, configs=configs)

    def test_unstructured_prune_gmp(self):
        last_ratio = 0.0
        ratio = 0.0
        while len(self.pruner.ratios_stack) > 0:
            self.pruner.step()
            last_ratio = ratio
            ratio = self.pruner.ratio
            print(ratio)
            self.assertGreaterEqual(ratio, last_ratio)
        self.assertEqual(ratio, 0.98)


if __name__ == "__main__":
    unittest.main()
