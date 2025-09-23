import importlib
import os

from .. import registry
from quantization.managers.base_quantization_manager import BaseQuantizationManager

build_quantization_manager, register_quantization_manager, QUANTIZATION_MANAGER_REGISTRY = registry.setup_registry(
    '--quantization-manager',
    base_class=BaseQuantizationManager,
    default=None,
)

# automatically import Python files in the quantization_managers/ directory
for file in os.listdir(os.path.dirname(__file__)):
    if file.endswith('.py') and not file.startswith('_'):
        module = file[:file.find('.py')]
        importlib.import_module('quantization.managers.' + module)

# Namespace(data='/home/songmiao/workplace/datasets/imagenet/', arch='resnet18', lr=0.1, momentum=0.9, weight_decay=5e-05, batch_size=256, workers=24, epochs=100, lr_policy='cosine', step_size=40, print_freq=10, seed=None, evaluate=False, qconfig=Config (path: cfg/qt_W4_minmax_det__A4_mse_det__G4_minmax_sto.yaml): {'quantization_manager': 'qt_quantization_manager', 'weight_forward_config': {'qtype': 'int4', 'granularity': 'N', 'scaling': 'dynamic', 'observer': 'minmax', 'stochastic': False, 'filters': [{'id': 0, 'qtype': 'full'}, {'name': 'fc', 'qtype': 'full'}]}, 'weight_backward_config': {'qtype': 'int4', 'granularity': 'C', 'scaling': 'dynamic', 'observer': 'minmax', 'stochastic': False, 'filters': [{'id': 0, 'qtype': 'full'}, {'name': 'fc', 'qtype': 'full'}]}, 'input_config': {'qtype': 'int4', 'granularity': 'N', 'scaling': 'dynamic', 'observer': 'mse', 'stochastic': False, 'filters': [{'id': 0, 'qtype': 'full'}, {'name': 'fc', 'qtype': 'full'}]}, 'grad_input_config': {'qtype': 'int4', 'granularity': 'NHW', 'scaling': 'dynamic', 'observer': 'minmax', 'stochastic': True, 'filters': [{'id': 0, 'qtype': 'full'}, {'name': 'fc', 'qtype': 'full'}]}, 'grad_weight_config': {'qtype': 'int4', 'granularity': 'NC', 'scaling': 'dynamic', 'observer': 'minmax', 'stochastic': True, 'filters': [{'id': 0, 'qtype': 'full'}, {'name': 'fc', 'qtype': 'full'}]}}, eval_quant=False, upbn=False, results_dir='./checkpoint', save='2024-09-29_19-42-02', save_path='./checkpoint/2024-09-29_19-42-02')
# Config (path: cfg/qt_W4_minmax_det__A4_mse_det__G4_minmax_sto.yaml): {'quantization_manager': 'qt_quantization_manager', 'weight_forward_config': {'qtype': 'int4', 'granularity': 'N', 'scaling': 'dynamic', 'observer': 'minmax', 'stochastic': False, 'filters': [{'id': 0, 'qtype': 'full'}, {'name': 'fc','qtype': 'full'}]}, 'weight_backward_config': {'qtype': 'int4', 'granularity': 'C', 'scaling': 'dynamic', 'observer': 'minmax', 'stochastic': False, 'filters': [{'id': 0, 'qtype': 'full'}, {'name': 'fc', 'qtype': 'full'}]}, 'input_config': {'qtype': 'int4', 'granularity': 'N', 'scaling': 'dynamic', 'observer': 'mse', 'stochastic': False, 'filters': [{'id': 0, 'qtype': 'full'}, {'name': 'fc', 'qtype': 'full'}]}, 'grad_input_config': {'qtype': 'int4', 'granularity': 'NHW', 'scaling': 'dynamic', 'observer': 'minmax', 'stochastic': True, 'filters': [{'id': 0, 'qtype': 'full'}, {'name': 'fc', 'qtype': 'full'}]}, 'grad_weight_config': {'qtype': 'int4', 'granularity': 'NC', 'scaling': 'dynamic', 'observer': 'minmax', 'stochastic': True, 'filters': [{'id': 0, 'qtype': 'full'}, {'name': 'fc', 'qtype': 'full'}]}}