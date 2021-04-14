import sys
sys.path.append("../../")
import unittest
import paddle
import numpy as np
from paddleslim.dygraph.prune.unstructured_pruner import UnstructuredPruner
from paddle.vision.models import mobilenet_v1


class TestUnstructuredPruner(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestUnstructuredPruner, self).__init__(*args, **kwargs)
        paddle.disable_static()
        self._gen_model()

    def _gen_model(self):
        self.net = mobilenet_v1(num_classes=10, pretrained=False)
        self.pruner = UnstructuredPruner(
            self.net, mode='ratio', ratio=0.98, threshold=0.0)

    def test_prune(self):
        ori_density = UnstructuredPruner.total_sparse(self.net)
        ori_threshold = self.pruner.threshold
        self.pruner.step()
        self.net(
            paddle.to_tensor(
                np.random.uniform(0, 1, [16, 3, 32, 32]), dtype='float32'))
        cur_density = UnstructuredPruner.total_sparse(self.net)
        cur_threshold = self.pruner.threshold
        print("Original threshold: {}".format(ori_threshold))
        print("Current threshold: {}".format(cur_threshold))
        print("Original density: {}".format(ori_density))
        print("Current density: {}".format(cur_density))
        self.assertLessEqual(ori_threshold, cur_threshold)
        self.assertLessEqual(cur_density, ori_density)

        self.pruner.update_params()
        self.assertEqual(cur_density, UnstructuredPruner.total_sparse(self.net))


if __name__ == "__main__":
    unittest.main()
