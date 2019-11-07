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

    transform_pass = QuantizationTransformPass(
        scope=scope, place=place,
        activation_quantize_type=activation_quant_type,
        weight_quantize_type=weight_quant_type)
    transform_pass.apply(main_graph)

    if for_test:
        quant_program = main_graph.to_program()
    else:
        build_strategy = fluid.BuildStrategy()
        build_strategy.memory_optimize = False
        build_strategy.enable_inplace = False
        binary = fluid.CompiledProgram(main_graph.graph).with_data_parallel(
            loss_name=loss_name, build_strategy=build_strategy)

        quant_program = binary
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


