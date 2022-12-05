import sys
sys.path.append("../../")
import unittest
import paddle
import numpy as np
from paddleslim import UnstructuredPruner, GMPUnstructuredPruner


class TestUnstructuredPruner(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestUnstructuredPruner, self).__init__(*args, **kwargs)
        paddle.disable_static()
        self._gen_model()

    def _gen_model(self):
        self.net = paddle.vision.models.mobilenet_v1(
            num_classes=10, pretrained=False)
        configs = {
            'stable_iterations': 0,
            'pruning_iterations': 1000,
            'tunning_iterations': 1000,
            'resume_iteration': 500,
            'pruning_steps': 20,
            'initial_ratio': 0.05,
        }
        self.pruner = GMPUnstructuredPruner(
            self.net, ratio=0.55, configs=configs)

        self.assertGreater(self.pruner.ratio, 0.3)

    def test_unstructured_prune_gmp(self):
        last_ratio = 0.0
        ratio = 0.0
        while len(self.pruner.ratios_stack) > 0:
            self.pruner.step()
            last_ratio = ratio
            ratio = self.pruner.ratio
            self.assertGreaterEqual(ratio, last_ratio)
        self.assertEqual(ratio, 0.55)


if __name__ == "__main__":
    unittest.main()
