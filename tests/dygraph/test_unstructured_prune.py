import sys
sys.path.append("../../")
import unittest
import paddle
import numpy as np
from paddleslim import UnstructuredPruner
from paddle.vision.models import mobilenet_v1


class TestUnstructuredPruner(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestUnstructuredPruner, self).__init__(*args, **kwargs)
        paddle.disable_static()
        self._gen_model()

    def _gen_model(self):
        self.net = mobilenet_v1(num_classes=10, pretrained=False)
        self.net_conv1x1 = mobilenet_v1(num_classes=10, pretrained=False)
        self.net_mxn = mobilenet_v1(num_classes=10, pretrained=False)
        self.pruner = UnstructuredPruner(
            self.net, mode='ratio', ratio=0.55, local_sparsity=True)
        self.pruner_conv1x1 = UnstructuredPruner(
            self.net_conv1x1,
            mode='ratio',
            ratio=0.55,
            prune_params_type='conv1x1_only',
            local_sparsity=False)
        self.pruner_mxn = UnstructuredPruner(
            self.net_mxn,
            mode='ratio',
            ratio=0.55,
            local_sparsity=True,
            sparse_block=[2, 1])

    def test_prune(self):
        ori_sparsity = UnstructuredPruner.total_sparse(self.net)
        ori_threshold = self.pruner.threshold
        self.pruner.step()
        self.net(
            paddle.to_tensor(
                np.random.uniform(0, 1, [16, 3, 32, 32]), dtype='float32'))
        cur_sparsity = UnstructuredPruner.total_sparse(self.net)
        cur_threshold = self.pruner.threshold
        print("Original threshold: {}".format(ori_threshold))
        print("Current threshold: {}".format(cur_threshold))
        print("Original sparsity: {}".format(ori_sparsity))
        print("Current sparsity: {}".format(cur_sparsity))
        self.assertLessEqual(ori_threshold, cur_threshold)
        self.assertGreaterEqual(cur_sparsity, ori_sparsity)

        self.pruner.update_params()
        self.assertEqual(cur_sparsity,
                         UnstructuredPruner.total_sparse(self.net))

    def test_summarize_weights(self):
        max_value = -float("inf")
        threshold = self.pruner.summarize_weights(self.net, 1.0)
        for name, sub_layer in self.net.named_sublayers():
            if not self.pruner._should_prune_layer(sub_layer):
                continue
            for param in sub_layer.parameters(include_sublayers=False):
                max_value = max(
                    max_value,
                    np.max(np.abs(np.array(param.value().get_tensor()))))
        print("The returned threshold is {}.".format(threshold))
        print("The max_value is {}.".format(max_value))
        self.assertEqual(max_value, threshold)

    def test_unstructured_prune_conv1x1(self):
        print(self.pruner.skip_params)
        print(self.pruner_conv1x1.skip_params)
        self.assertTrue(
            len(self.pruner.skip_params) < len(self.pruner_conv1x1.skip_params))
        self.pruner_conv1x1.step()
        self.pruner_conv1x1.update_params()
        cur_sparsity = UnstructuredPruner.total_sparse_conv1x1(self.net_conv1x1)
        self.assertTrue(abs(cur_sparsity - 0.55) < 0.01)

    def test_block_prune_mxn(self):
        self.pruner_mxn.step()
        self.pruner_mxn.update_params()
        cur_sparsity = UnstructuredPruner.total_sparse(self.net_mxn)
        self.assertTrue(abs(cur_sparsity - 0.55) < 0.01)


if __name__ == "__main__":
    unittest.main()
