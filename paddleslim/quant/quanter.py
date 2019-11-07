import paddle
import paddle.fluid as fluid
from paddle.fluid.framework import IrGraph
from paddle.fluid.contrib.slim.quantization import QuantizationTransformPass
from paddle.fluid.contrib.slim.quantization import QuantizationFreezePass
from paddle.fluid.contrib.slim.quantization import ConvertToInt8Pass
from paddle.fluid.contrib.slim.quantization import TransformForMobilePass
from paddle.fluid import core

def quant_aware(program, scope, place, config, for_test=False, loss_name=''):
    main_graph = IrGraph(core.Graph(program.desc), for_test=for_test)

    weight_quant_type = 'abs_max'
    activation_quant_type = 'abs_max'
    if 'weight_quantize_type' in config:
        weight_quant_type = config['weight_quantize_type']
    if 'activation_quantize_type' in config:
        activation_quant_type = config['activation_quantize_type']

    weight_bits = 8
    activation_bits = 8
    if 'weight_bits' in config:
        weight_bits = config['weight_bits']
    if 'activation_bits' in config:
        activation_bits = config['activation_bits']

    window_size=10000
    if 'window_size' in config:
        window_size = config['window_size']

    moving_rate = 10000
    if 'moving_rate' in config:
        moving_rate = config['moving_rate']

    not_quant_pattern=['skip_quant']
    assert not_quant_pattern is list, 'not_quant_pattern should config as list, for example, not_quant_pattern:["skip_quant"]'


    transform_pass = QuantizationTransformPass(
        scope=scope, place=place,
        weight_bits=weight_bits,
        activation_bits=activation_bits,
        activation_quantize_type=activation_quant_type,
        weight_quantize_type=weight_quant_type,
        window_size=window_size,
        moving_rate=moving_rate,
        skip_pattern=''#not_quant_pattern
    )


    transform_pass.apply(main_graph)

    if for_test:
        quant_program = main_graph.to_program()
    else:
        quant_program = fluid.CompiledProgram(main_graph.graph)
    return quant_program

def quant_post(program, scope, place, config):
    main_graph = IrGraph(core.Graph(program.desc), for_test=True)

    weight_quant_type = 'abs_max'
    activation_quant_type = 'abs_max'
    if 'weight_quantize_type' in config:
        weight_quant_type = config['weight_quantize_type']
    if 'activation_quantize_type' in config:
        activation_quant_type = config['activation_quantize_type']

    transform_pass = QuantizationTransformPass(
        scope=scope, place=place,
        activation_quantize_type=activation_quant_type,
        weight_quantize_type=weight_quant_type)
    transform_pass.apply(main_graph)


    quant_program = main_graph.to_program()
    return quant_program

def convert(program, scope, place, config, save_int8=False):
    test_graph = IrGraph(core.Graph(program.desc), for_test=True)

    # 2. Freeze the graph after training by adjusting the quantize
    # operators' order for the inference.

    weight_quant_type = 'abs_max'
    if 'weight_quantize_type' in config:
        weight_quant_type = config['weight_quantize_type']
    freeze_pass = QuantizationFreezePass(
        scope=scope,
        place=place,
        weight_quantize_type=weight_quant_type)
    freeze_pass.apply(test_graph)
    freezed_program = test_graph.to_program()
    freezed_program_int8 = None

    if save_int8:
        convert_int8_pass = ConvertToInt8Pass(scope=fluid.global_scope(), place=place)
        convert_int8_pass.apply(test_graph)
        freezed_program_int8 = test_graph.to_program()

    return freezed_program, freezed_program_int8


