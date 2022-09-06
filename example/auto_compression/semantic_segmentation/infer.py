# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import cv2
import numpy as np
import argparse
import time
from tqdm import tqdm
import paddle
import paddleseg.transforms as T
from paddleseg.cvlibs import Config as PaddleSegDataConfig
from paddleseg.core.infer import reverse_transform
from paddleseg.utils.visualize import get_pseudo_color_map
from paddleseg.utils import metrics
from paddle.inference import create_predictor, PrecisionType
from paddle.inference import Config as PredictConfig


def _transforms(dataset):
    transforms = []
    if dataset == "human":
        transforms.append(T.PaddingByAspectRatio(aspect_ratio=1.77777778))
        transforms.append(T.Resize(target_size=[398, 224]))
        transforms.append(T.Normalize())
    elif dataset == "cityscape":
        transforms.append(T.Normalize())
    return transforms


def auto_tune_trt(args, data):
    auto_tuned_shape_file = "./auto_tuning_shape"
    pred_cfg = PredictConfig(args.model_path, args.params_path)
    pred_cfg.enable_use_gpu(100, 0)
    pred_cfg.collect_shape_range_info("./auto_tuning_shape")
    predictor = create_predictor(pred_cfg)
    input_names = predictor.get_input_names()
    input_handle = predictor.get_input_handle(input_names[0])
    input_handle.reshape(data.shape)
    input_handle.copy_from_cpu(data)
    predictor.run()
    return auto_tuned_shape_file


def load_predictor(args, data):
    pred_cfg = PredictConfig(args.model_path, args.params_path)
    pred_cfg.disable_glog_info()
    pred_cfg.enable_memory_optim()
    pred_cfg.switch_ir_optim(True)
    if args.device == "GPU":
        pred_cfg.enable_use_gpu(100, 0)

    if args.use_trt:
        # To collect the dynamic shapes of inputs for TensorRT engine
        auto_tuned_shape_file = auto_tune_trt(args, data)
        precision_map = {
            "fp16": PrecisionType.Half,
            "fp32": PrecisionType.Float32,
            "int8": PrecisionType.Int8
        }
        pred_cfg.enable_tensorrt_engine(
            workspace_size=1 << 30,
            max_batch_size=1,
            min_subgraph_size=4,
            precision_mode=precision_map[args.precision],
            use_static=False,
            use_calib_mode=False)
        allow_build_at_runtime = True
        pred_cfg.enable_tuned_tensorrt_dynamic_shape(auto_tuned_shape_file,
                                                     allow_build_at_runtime)
    predictor = create_predictor(pred_cfg)
    return predictor


def predict_image(args):

    transforms = _transforms(args.dataset)
    transform = T.Compose(transforms)

    # Step1: Load image and preprocess
    im = cv2.imread(args.image_file).astype('float32')
    data, _ = transform(im)
    data = np.array(data)[np.newaxis, :]

    # Step2: Prepare prdictor
    predictor = load_predictor(args, data)

    # Step3: Inference
    input_names = predictor.get_input_names()
    input_handle = predictor.get_input_handle(input_names[0])
    output_names = predictor.get_output_names()
    output_handle = predictor.get_output_handle(output_names[0])
    input_handle.reshape(data.shape)
    input_handle.copy_from_cpu(data)

    warmup, repeats = 0, 1
    if args.benchmark:
        warmup, repeats = 20, 100

    for i in range(warmup):
        predictor.run()

    start_time = time.time()
    for i in range(repeats):
        predictor.run()
        results = output_handle.copy_to_cpu()
    total_time = time.time() - start_time
    avg_time = float(total_time) / repeats
    print(f"Average inference time: \033[91m{round(avg_time*1000, 2)}ms\033[0m")

    # Step4: Post process
    if args.dataset == "human":
        results = reverse_transform(
            paddle.to_tensor(results), im.shape, transforms, mode='bilinear')
        results = np.argmax(results, axis=1)
    result = get_pseudo_color_map(results[0])

    # Step5: Save result to file
    if args.save_file is not None:
        result.save(args.save_file)
        print(f"Saved result to \033[91m{args.save_file}\033[0m")


def eval(args):
    # DataLoader need run on cpu
    paddle.set_device('cpu')
    data_cfg = PaddleSegDataConfig(args.dataset_config)
    eval_dataset = data_cfg.val_dataset

    batch_sampler = paddle.io.BatchSampler(
        eval_dataset, batch_size=1, shuffle=False, drop_last=False)
    loader = paddle.io.DataLoader(
        eval_dataset,
        batch_sampler=batch_sampler,
        num_workers=1,
        return_list=True)

    total_iters = len(loader)
    intersect_area_all = 0
    pred_area_all = 0
    label_area_all = 0

    print("Start evaluating (total_samples: {}, total_iters: {})...".format(
        len(eval_dataset), total_iters))

    init_predictor = False
    for (image, label) in tqdm(loader):
        label = np.array(label).astype('int64')
        ori_shape = np.array(label).shape[-2:]
        data = np.array(image)

        if not init_predictor:
            predictor = load_predictor(args, data)
            init_predictor = True

        input_names = predictor.get_input_names()
        input_handle = predictor.get_input_handle(input_names[0])
        input_handle.reshape(data.shape)
        input_handle.copy_from_cpu(data)

        predictor.run()

        output_names = predictor.get_output_names()
        output_handle = predictor.get_output_handle(output_names[0])
        results = output_handle.copy_to_cpu()

        logit = reverse_transform(
            paddle.to_tensor(results),
            ori_shape,
            eval_dataset.transforms.transforms,
            mode='bilinear')
        pred = paddle.to_tensor(logit)
        if len(
                pred.shape
        ) == 4:  # for humanseg model whose prediction is distribution but not class id
            pred = paddle.argmax(pred, axis=1, keepdim=True, dtype='int32')

        intersect_area, pred_area, label_area = metrics.calculate_area(
            pred,
            paddle.to_tensor(label),
            eval_dataset.num_classes,
            ignore_index=eval_dataset.ignore_index)
        intersect_area_all = intersect_area_all + intersect_area
        pred_area_all = pred_area_all + pred_area
        label_area_all = label_area_all + label_area

    class_iou, miou = metrics.mean_iou(intersect_area_all, pred_area_all,
                                       label_area_all)
    class_acc, acc = metrics.accuracy(intersect_area_all, pred_area_all)
    kappa = metrics.kappa(intersect_area_all, pred_area_all, label_area_all)
    class_dice, mdice = metrics.dice(intersect_area_all, pred_area_all,
                                     label_area_all)

    infor = "[EVAL] #Images: {} mIoU: {:.4f} Acc: {:.4f} Kappa: {:.4f} Dice: {:.4f}".format(
        len(eval_dataset), miou, acc, kappa, mdice)
    print(infor)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--image_file',
        type=str,
        default=None,
        help="Image path to be processed.")
    parser.add_argument(
        '--save_file',
        type=str,
        default=None,
        help="The path to save the processed image.")
    parser.add_argument(
        '--model_path', type=str, help="Inference model filepath.")
    parser.add_argument(
        '--params_path', type=str, help="Inference parameters filepath.")
    parser.add_argument(
        '--dataset',
        type=str,
        default="human",
        choices=["human", "cityscape"],
        help="The type of given image which can be 'human' or 'cityscape'.")
    parser.add_argument(
        '--dataset_config',
        type=str,
        default=None,
        help="path of dataset config.")
    parser.add_argument(
        '--benchmark',
        type=bool,
        default=False,
        help="Whether to run benchmark or not.")
    parser.add_argument(
        '--use_trt',
        type=bool,
        default=False,
        help="Whether to use tensorrt engine or not.")
    parser.add_argument(
        '--device',
        type=str,
        default='GPU',
        choices=["CPU", "GPU"],
        help="Choose the device you want to run, it can be: CPU/GPU, default is GPU"
    )
    parser.add_argument(
        '--precision',
        type=str,
        default='fp32',
        choices=["fp32", "fp16", "int8"],
        help="The precision of inference. It can be 'fp32', 'fp16' or 'int8'. Default is 'fp16'."
    )
    args = parser.parse_args()
    if args.image_file:
        predict_image(args)
    else:
        eval(args)
