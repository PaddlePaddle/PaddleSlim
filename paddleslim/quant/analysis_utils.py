import os
import sys
import csv
import logging
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from ..common import get_logger
import paddle
import paddle.nn.functional as F
from paddle.static.quantization.utils import load_variable_data

_logger = get_logger(__name__, level=logging.INFO)


def collect_vars(scope, var_names):
    all_vars = {}
    for var_name in var_names:
        var_tensor = load_variable_data(scope, var_name)
        all_vars[var_name] = var_tensor
    return all_vars


def plot_box_distribution(box_data, save_dir, save_name):
    all_values = sum(list(box_data.values()), [])
    max_value = np.max(all_values)
    min_value = np.min(all_values)
    pdf_path = os.path.join(save_dir, save_name)
    labels = sorted(box_data.keys())
    with PdfPages(pdf_path) as pdf:
        for i in range(0, len(labels), 20):
            r = i + 20 if i + 20 < len(labels) else len(labels)
            dist = [box_data[n] for n in labels[i:r]]
            plt.boxplot(
                dist, labels=labels[i:r], showbox=True, patch_artist=True)
            plt.xticks(rotation=90)
            plt.tick_params(axis='x')
            plt.ylim([min_value, max_value])
            if 'act' in save_name:
                plt.xlabel('Activation Name')
            else:
                plt.xlabel('Weight Name')
            plt.ylabel("Box Distribution")
            plt.tight_layout()
            plt.show()
            pdf.savefig()
            plt.close()
    _logger.info('Box plots is saved in {}'.format(pdf_path))


def plot_hist_distribution(hist_data, save_dir, save_name):
    pdf_path = os.path.join(save_dir, save_name)
    with PdfPages(pdf_path) as pdf:
        for name in hist_data:
            plt.hist(hist_data[name][0], bins=hist_data[name][1])
            plt.xlabel(name)
            plt.ylabel("Probability")
            locs, _ = plt.yticks()
            plt.yticks(locs, np.round(locs / len(hist_data[name][0]), 3))
            if 'act' in save_name:
                plt.title("Hist of Activation {}".format(name))
            else:
                plt.title("Hist of Weight {}".format(name))
            plt.show()
            pdf.savefig()
            plt.close()
    _logger.info('Histogram plot is saved in {}'.format(pdf_path))


def save_csv(data, save_dir, save_name, csv_columns):
    save_path = os.path.join(save_dir, save_name)
    with open(save_path, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
        writer.writeheader()
        for d in data:
            writer.writerow(d)
    _logger.info('Activation Statistic is saved in {}'.format(save_path))


def fp_quant_cosine_similarity(executor, data_loader, float_program,
                               quant_program, float_scope, quant_scope,
                               float_fetch_list, quant_fetch_list):
    cosine_similarity = []
    for step, data in enumerate(data_loader()):
        with paddle.static.scope_guard(float_scope):
            float_preds = executor.run(
                program=float_program,
                feed=data,
                fetch_list=float_fetch_list,
                return_numpy=False)
            float_preds = float_preds[0]
        with paddle.static.scope_guard(quant_scope):
            quant_preds = executor.run(
                program=quant_program,
                feed=data,
                fetch_list=quant_fetch_list,
                return_numpy=False)
            quant_preds = quant_preds[0]
        paddle.disable_static()
        float_preds = paddle.to_tensor(float_preds)
        quant_preds = paddle.to_tensor(quant_preds)
        cos_sim = F.cosine_similarity(float_preds, quant_preds).mean()
        cos_sim = cos_sim.numpy()
        cosine_similarity.append(cos_sim)
        if step != 0 and (step % 10 == 0):
            _logger.info("[step]: %d, cosine similarity: %.9f" %
                         (step, np.array(cosine_similarity).mean()))
        paddle.enable_static()

    return np.array(cosine_similarity).mean()


def get_new_in_out_map(input_name, graph, float_scope, quant_scope, place):

    input_rename_map = {}
    output_rename_map = {}
    removed_ops = []
    for op_node in graph.all_op_nodes():
        if op_node.id() in removed_ops:
            continue
        in_names = op_node.input_arg_names()
        out_names = op_node.output_arg_names()
        if out_names[0] == input_name:
            in_var = graph._find_node_by_name(op_node.inputs,
                                              op_node.input('X')[0])
            out_var = graph._find_node_by_name(op_node.outputs,
                                               op_node.output('Y')[0])
            if not in_var.persistable():
                # act
                for op in graph.all_op_nodes():
                    o_ns = op.output_arg_names()
                    if len(o_ns) == 1 and o_ns[0] == in_var.name():
                        in_var_1 = graph._find_node_by_name(
                            op.inputs, op.input('X')[0])
                        graph.safe_remove_nodes(op)
                        removed_ops.append(op.id())
                        input_rename_map[out_var.node] = in_var_1
            else:
                # weight
                with paddle.static.scope_guard(float_scope):
                    float_name = in_var.name().replace('.quantized', '')
                    float_weight = np.array(
                        float_scope.find_var(float_name).get_tensor())
                with paddle.static.scope_guard(quant_scope):
                    quant_scope.find_var(in_var.name()).get_tensor().set(
                        float_weight, place)
                input_rename_map[out_var.node] = in_var
            graph.safe_remove_nodes(op_node)
            removed_ops.append(op_node.id())
            output_rename_map[in_var.node] = out_var

    return input_rename_map, output_rename_map, removed_ops


def relink_graph(graph, input_rename_map, output_rename_map, removed_ops):
    for op_node in graph.all_op_nodes():
        if op_node.id() in removed_ops:
            continue
        for var in op_node.inputs:
            if var.node in input_rename_map:
                old_in = var
                new_in = input_rename_map[var.node]
                graph.update_input_link(old_in, new_in, op_node)
                _logger.info(
                    f'relink {op_node.name()} \'s input node from {old_in.name()} to {new_in.name()}.'
                )
        for var in op_node.outputs:
            if var.node in output_rename_map:
                old_out = var
                new_out = output_rename_map[var.node]
                graph.update_input_link(old_out, new_out, op_node)
                _logger.info(
                    f'relink {op_node.name()} \'s output node from {old_out.name()} to {new_out.name()}.'
                )

    return graph.to_program()
