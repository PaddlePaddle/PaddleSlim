import paddle
import paddle.fluid as fluid
from paddle.fluid import core
from paddle.fluid.framework import IrGraph
from paddle.fluid.executor import global_scope
import sys
import argparse

paddle.enable_static()

def _remove_unused_var_nodes(graph):
        all_used_vars = set()
        ops = graph.all_op_nodes()
        for op_node in ops:
            for input_node in op_node.inputs:
                all_used_vars.add(input_node)
            for output_node in op_node.outputs:
                all_used_vars.add(output_node)

        all_used_vars = {n.node for n in all_used_vars}
        all_unused_vars = {
            n
            for n in filter(lambda node: node.node not in all_used_vars,
                            graph.all_var_nodes())
        }
        graph.safe_remove_nodes(all_unused_vars)
        return graph

def _apply_pass(scope, graph, pass_name, attrs=None,
                attr_values=None, debug=False):
    ir_pass = core.get_pass(pass_name)
    cpp_graph = graph.graph
    if not cpp_graph.has('__param_scope__'):
        cpp_graph.set_not_owned('__param_scope__', scope)
    if attrs:
        assert attr_values and len(attrs) == len(
            attr_values
        ), "Different number of pass attributes and their values."
        for attr, value in zip(attrs, attr_values):
            ir_pass.set(attr, value)
    ir_pass.apply(cpp_graph)
    if debug:
        graph.draw('.', 'qat_fp32_{}'.format(pass_name),
                    graph.all_op_nodes())
    _remove_unused_var_nodes(graph)
    return graph

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_model_dir', type=str, default='')
    parser.add_argument('--load_model_filename', type=str, default='')
    parser.add_argument('--load_params_filename', type=str, default='')
    parser.add_argument('--save_model_dir', type=str, default='')
    parser.add_argument('--save_model_filename', type=str, default='')
    parser.add_argument('--save_params_filename', type=str, default='')
    args = parser.parse_args()
    
    assert args.load_model_dir != "" and args.save_model_dir != ""
    if args.load_model_filename == "":
        args.load_model_filename = None
    if args.load_params_filename == "":
        args.load_params_filename = None
    if args.save_model_filename == "":
        args.save_model_filename = None
    if args.save_params_filename == "":
        args.save_params_filename = None

    return args
    

if __name__ == "__main__":
    args = parse_args()
    
    exe = fluid.Executor(fluid.CPUPlace())
    scope = global_scope()
    [program, feed_list, fetch_list] = \
            fluid.io.load_inference_model(dirname=args.load_model_dir,
                                          executor=exe,
                                          model_filename=args.load_model_filename,
                                          params_filename=args.load_params_filename)
    graph = IrGraph(core.Graph(program.desc), for_test=True)
    graph = _apply_pass(scope, graph, 'conv_bn_fuse_pass')
    graph = _apply_pass(scope, graph, 'depthwise_conv_bn_fuse_pass')
    graph = _apply_pass(scope, graph, 'conv_eltwiseadd_bn_fuse_pass')
    graph = _apply_pass(scope, graph, 'depthwise_conv_eltwiseadd_bn_fuse_pass')

    program = graph.to_program()

    fluid.io.save_inference_model(
                dirname=args.save_model_dir,
                feeded_var_names=feed_list,
                target_vars=fetch_list,
                executor=exe,
                main_program=program,
                model_filename=args.save_model_filename,
                params_filename=args.save_params_filename)
