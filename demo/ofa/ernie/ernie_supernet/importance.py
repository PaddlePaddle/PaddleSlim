import os
import numpy as np
import paddle
import paddle.fluid as F
import paddle.fluid.dygraph as FD
import paddle.fluid.layers as L


def compute_neuron_head_importance(args, model, tokenizer, dev_ds, place):
    n_layers, n_heads = model.cfg['num_hidden_layers'], model.cfg[
        'num_attention_heads']
    head_importance = L.zeros(shape=[n_layers, n_heads], dtype='float32')
    head_mask = L.ones(shape=[n_layers, n_heads], dtype='float32')
    head_mask.stop_gradient = False

    intermediate_weight = []
    intermediate_bias = []
    output_weight = []

    for name, w in model.named_parameters():
        ### BERT Encoder match ERNIE Block, not same, need to check it
        if 'ffn.i' in name:
            if len(w.shape) > 1:
                intermediate_weight.append(w)
            else:
                intermediate_bias.append(w)

        if 'ffn.o' in name:
            if len(w.shape) > 1:
                output_weight.append(w)

    neuron_importance = []
    for w in intermediate_weight:
        neuron_importance.append(np.zeros(shape=[w.shape[1]], dtype='float32'))

    eval_task_names = ('mnli', 'mnli-mm') if args.task == 'mnli' else (
        args.task, )

    for eval_task in eval_task_names:
        for batch in dev_ds.start(place):
            ids, sids, label = batch
            loss, _, _ = model(ids, sids, labels=label, head_mask=head_mask)
            loss.backward()
            head_importance += L.abs(FD.to_variable(head_mask.gradient()))

            for w1, b1, w2, current_importance in zip(
                    intermediate_weight, intermediate_bias, output_weight,
                    neuron_importance):
                current_importance += np.abs(
                    (np.sum(w1.numpy() * w1.gradient(), axis=0) + b1.numpy() *
                     b1.gradient()))
                current_importance += np.abs(
                    np.sum(w2.numpy() * w2.gradient(), axis=1))

    return head_importance, neuron_importance


def reorder_neuron_head(model, head_importance, neuron_importance):
    # reorder heads and ffn neurons
    for layer, current_importance in enumerate(neuron_importance):
        # reorder heads
        idx = L.argsort(head_importance[layer], descending=True)[-1]
        model.encoder_stack.block[layer].attn.reorder_heads(idx)
        # reorder neurons
        idx = L.argsort(FD.to_variable(current_importance), descending=True)[-1]
        model.encoder_stack.block[layer].ffn.reorder_neurons(idx)
        #model.encoder_stack.layer[layer].ffn.o.reorder_neurons(idx)
