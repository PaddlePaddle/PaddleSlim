import numbers
import numpy as np
from collections import OrderedDict
import paddle
import paddle.nn as nn


def analysis_model(models, input_size=None, dtypes=None, input=None):
    def _all_is_numper(items):
        for item in items:
            if not isinstance(item, numbers.Number):
                return False
        return True

    def _build_dtypes(input_size, dtype):
        if dtype is None:
            dtype = 'float32'

        if isinstance(input_size, (list, tuple)) and _all_is_numper(input_size):
            return [dtype]
        else:
            return [_build_dtypes(i, dtype) for i in input_size]

    if not isinstance(dtypes, (list, tuple)):
        dtypes = _build_dtypes(input_size, dtypes)

    def _get_shape_from_tensor(x):
        if isinstance(x, (paddle.fluid.Variable, paddle.fluid.core.VarBase)):
            return list(x.shape)
        elif isinstance(x, (list, tuple)):
            return [_get_shape_from_tensor(xx) for xx in x]

    def _get_output_shape(output):
        if isinstance(output, (list, tuple)):
            output_shape = [_get_output_shape(o) for o in output]
        elif hasattr(output, 'shape'):
            output_shape = list(output.shape)
        else:
            output_shape = []
        return output_shape

    def register_hook(model):
        def add_hook(layer_name, layer):
            def hook(layer, input, output):
                m_key = layer_name
                summary[m_key] = OrderedDict()

                try:
                    summary[m_key]["input_shape"] = _get_shape_from_tensor(
                        input)
                except:
                    warnings.warn('Get layer {} input shape failed!')
                    summary[m_key]["input_shape"] = []

                try:
                    summary[m_key]["output_shape"] = _get_output_shape(output)
                except:
                    warnings.warn('Get layer {} output shape failed!')
                    summary[m_key]["output_shape"]

            return hook

        hooks.append(
            model.register_forward_post_hook(add_hook('model_info', model)))
        for name, sublayer in model.named_sublayers():
            hooks.append(
                sublayer.register_forward_post_hook(add_hook(name, sublayer)))

    def build_input(input_size, dtypes):
        if isinstance(input_size, (list, tuple)) and _all_is_numper(input_size):
            if isinstance(dtypes, (list, tuple)):
                dtype = dtypes[0]
            else:
                dtype = dtypes
            return paddle.cast(paddle.rand(list(input_size)), dtype)
        else:
            return [
                build_input(i, dtype) for i, dtype in zip(input_size, dtypes)
            ]

    if input_size is None and input is None:
        raise ValueError("input_size and input cannot be None at the same time")

    if input_size is None and input is not None:
        if paddle.is_tensor(input):
            input_size = tuple(input.shape)
        elif isinstance(input, (list, tuple)):
            input_size = []
            for x in input:
                input_size.append(tuple(x.shape))
        elif isinstance(input, dict):
            input_size = []
            for key in input.keys():
                input_size.append(tuple(input[key].shape))
        else:
            raise ValueError(
                "Input is not tensor, list, tuple and dict, unable to determine input_size, please input input_size."
            )

    if isinstance(input_size, tuple):
        input_size = [input_size]

    if isinstance(models, nn.Layer):
        models = [models]

    for idx, model in enumerate(models):
        summary = OrderedDict()
        model.eval()
        hooks = []
        register_hook(model)

        if input is not None:
            x = input
            model(x)
        else:
            x = build_input(input_size, dtypes)
            # make a forward pass
            model(*x)

        # remove these hooks
        for h in hooks:
            h.remove()

        print("========================= Model {} ==========================".
              format(idx))
        for layer in summary:
            line = "layer name: \033[1;35m {:<20} \033[0m input_shapes: {:<20} output_shape: \033[1;35m {:<20} \033[0m".format(
                layer,
                str(summary[layer]['input_shape']),
                str(summary[layer]['output_shape']))
            print(line)
        print("=============================================================")


if __name__ == '__main__':

    class Model(nn.Layer):
        def __init__(self):
            super(Model, self).__init__()
            self.conv1 = nn.Conv2D(3, 3, 3, padding=1)
            self.conv2 = nn.Conv2D(3, 3, 3, padding=1)
            self.conv3 = nn.Conv2D(3, 3, 3, padding=1)
            self.fc = nn.Linear(3072, 10)

        def forward(self, x):
            self.conv1_out = self.conv1(x)
            conv2_out = self.conv2(self.conv1_out)
            self.conv3_out = self.conv3(conv2_out)
            out = paddle.reshape(self.conv3_out, shape=[x.shape[0], -1])
            out = self.fc(out)
            return out

    model = Model()
    model2 = Model()
    analysis_model([model, model2], input_size=(1, 3, 32, 32))
    #a = paddle.ones(shape=(1,3,32,32))
    #model(a)

    #model = paddle.vision.models.mobilenet_v2()
    #analysis_model(model, input_size=(1,3,224,224))
