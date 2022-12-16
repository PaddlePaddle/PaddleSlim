import sys
sys.path.append("../../")
import unittest
import paddle
import numpy as np
from paddle.static import InputSpec as Input
from paddleslim import UnstructuredPruner


class TestStaticMasks(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestStaticMasks, self).__init__(*args, **kwargs)
        paddle.disable_static()
        transform = paddle.vision.transforms.Compose([
            paddle.vision.transforms.Transpose(),
            paddle.vision.transforms.Normalize([127.5], [127.5])
        ])
        self.train_dataset = paddle.vision.datasets.MNIST(
            mode="train", backend="cv2", transform=transform)
        self.train_loader = paddle.io.DataLoader(
            self.train_dataset,
            places=paddle.set_device('cpu'),
            return_list=True)

        def _reader():
            for data in self.val_dataset:
                yield data

        self.val_reader = _reader

    def _update_masks(self, pruner, t):
        for name, sub_layer in pruner.model.named_sublayers():
            for param in sub_layer.parameters(include_sublayers=False):
                mask = pruner.masks.get(param.name)
                bool_tmp = (paddle.abs(param) < t)
                paddle.assign(bool_tmp, output=mask)

    def runTest(self):
        paddle.disable_static()
        net = paddle.vision.models.LeNet()
        optimizer = paddle.optimizer.Adam(
            learning_rate=0.001, parameters=net.parameters())
        inputs = [Input([None, 1, 28, 28], 'float32', name='image')]
        labels = [Input([None, 1], 'int64', name='label')]
        pruner = UnstructuredPruner(net, mode='ratio', ratio=0.55)
        net.train()
        self._update_masks(pruner, 0.0)
        pruner.update_params()
        self._update_masks(pruner, 1.0)
        pruner.set_static_masks()
        sparsity_0 = UnstructuredPruner.total_sparse(net)
        for i, data in enumerate(self.train_loader):
            x_data = data[0]
            y_data = paddle.to_tensor(data[1])
            logits = net(x_data)
            loss = paddle.nn.functional.cross_entropy(logits, y_data)
            loss.backward()
            optimizer.step()
            optimizer.clear_grad()
            if i == 10: break
        sparsity_1 = UnstructuredPruner.total_sparse(net)
        pruner.update_params()
        sparsity_2 = UnstructuredPruner.total_sparse(net)
        print(sparsity_0, sparsity_1, sparsity_2)
        self.assertEqual(sparsity_0, 1.0)
        self.assertLess(abs(sparsity_2 - 1), 0.001)
        self.assertLess(sparsity_1, 1.0)


if __name__ == "__main__":
    unittest.main()
