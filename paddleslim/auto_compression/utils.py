import paddle
from paddle.fluid.framework import IrGraph
from paddle.fluid import core
from paddle.fluid.contrib.slim.quantization import QuantizationTransformPass, AddQuantDequantPass, QuantizationFreezePass


class PTQDataFree(object):
    """
    Utilizing post training quantization methon to quantize the FP32 model,
    and it not uses calibrate data.
    Usage:
        paddle.enable_static()
        place = paddle.CPUPlace()
        exe = paddle.static.Executor(place)
        ptq_df = PTQDataFree(executor=exe,
            model_dir='./inference_model/MobileNet/',
            model_filename='model',
            params_filename='params',
            save_model_path='data_free')
        ptq_df()
    """

    def __init__(self,
                 executor=None,
                 scope=None,
                 model_dir=None,
                 model_filename=None,
                 params_filename=None,
                 save_model_path=None,
                 quantizable_op_type=["conv2d", "depthwise_conv2d", "mul"],
                 is_full_quantize=False,
                 activation_bits=8,
                 weight_bits=8,
                 activation_quantize_type='range_abs_max',
                 weight_quantize_type='channel_wise_abs_max'):
        self._support_activation_quantize_type = [
            'range_abs_max', 'moving_average_abs_max', 'abs_max'
        ]
        self._support_weight_quantize_type = ['abs_max', 'channel_wise_abs_max']
        self._dynamic_quantize_op_type = ['lstm']
        self._weight_supported_quantizable_op_type = QuantizationTransformPass._supported_quantizable_op_type
        self._act_supported_quantizable_op_type = AddQuantDequantPass._supported_quantizable_op_type
        self._support_quantize_op_type = \
            list(set(self._weight_supported_quantizable_op_type +
                self._act_supported_quantizable_op_type +
                self._dynamic_quantize_op_type))

        # Check inputs
        assert executor is not None, "The executor cannot be None."
        assert model_dir is not None, "The model_dir cannot be None."
        assert activation_quantize_type in self._support_activation_quantize_type, \
            "The activation_quantize_type ({}) should in ({}).".format(
            activation_quantize_type, self._support_activation_quantize_type)
        assert weight_quantize_type in self._support_weight_quantize_type, \
            "The weight_quantize_type ({}) shoud in ({}).".format(
            weight_quantize_type, self._support_weight_quantize_type)

        # Save input params
        self._executor = executor
        self._place = self._executor.place
        self._scope = paddle.static.global_scope() if scope == None else scope
        self._model_dir = model_dir
        self._model_filename = model_filename
        self._params_filename = params_filename
        self._save_model_path = save_model_path
        self._activation_bits = activation_bits
        self._weight_bits = weight_bits
        self._activation_quantize_type = activation_quantize_type
        self._weight_quantize_type = weight_quantize_type
        self._is_full_quantize = is_full_quantize
        if is_full_quantize:
            self._quantizable_op_type = self._support_quantize_op_type
        else:
            self._quantizable_op_type = quantizable_op_type
            for op_type in self._quantizable_op_type:
                assert op_type in self._support_quantize_op_type, \
                    op_type + " is not supported for quantization."
        self._program = None

    def __call__(self):
        self._program, _feed_list, _fetch_list = paddle.fluid.io.load_inference_model(
            self._model_dir,
            self._executor,
            model_filename=self._model_filename,
            params_filename=self._params_filename)

        graph = IrGraph(core.Graph(self._program.desc), for_test=True)

        # use QuantizationTransformPass to insert fake_quant/fake_dequantize op
        major_quantizable_op_types = []
        for op_type in self._weight_supported_quantizable_op_type:
            if op_type in self._quantizable_op_type:
                major_quantizable_op_types.append(op_type)
        transform_pass = QuantizationTransformPass(
            scope=self._scope,
            place=self._place,
            weight_bits=self._weight_bits,
            activation_bits=self._activation_bits,
            activation_quantize_type=self._activation_quantize_type,
            weight_quantize_type=self._weight_quantize_type,
            quantizable_op_type=major_quantizable_op_types)

        for sub_graph in graph.all_sub_graphs():
            # Insert fake_quant/fake_dequantize op must in test graph, so
            # set per graph's _for_test is True.
            sub_graph._for_test = True
            transform_pass.apply(sub_graph)

        # use AddQuantDequantPass to insert fake_quant_dequant op
        minor_quantizable_op_types = []
        for op_type in self._act_supported_quantizable_op_type:
            if op_type in self._quantizable_op_type:
                minor_quantizable_op_types.append(op_type)
        add_quant_dequant_pass = AddQuantDequantPass(
            scope=self._scope,
            place=self._place,
            quantizable_op_type=minor_quantizable_op_types)

        for sub_graph in graph.all_sub_graphs():
            sub_graph._for_test = True
            add_quant_dequant_pass.apply(sub_graph)

        # apply QuantizationFreezePass, and obtain the final quant model
        freeze_pass = QuantizationFreezePass(
            scope=self._scope,
            place=self._place,
            weight_bits=self._weight_bits,
            activation_bits=self._activation_bits,
            weight_quantize_type=self._weight_quantize_type,
            quantizable_op_type=major_quantizable_op_types)

        for sub_graph in graph.all_sub_graphs():
            sub_graph._for_test = True
            freeze_pass.apply(sub_graph)

        self._program = graph.to_program()

        paddle.fluid.io.save_inference_model(
            dirname=self._save_model_path,
            model_filename=self._model_filename,
            params_filename=self._params_filename,
            feeded_var_names=_feed_list,
            target_vars=_fetch_list,
            executor=self._executor,
            main_program=self._program)
        print("The quantized model is saved in " + self._save_model_path)
