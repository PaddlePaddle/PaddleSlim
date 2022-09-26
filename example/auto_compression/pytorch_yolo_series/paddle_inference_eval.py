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

import time
import os
import sys
import argparse
import cv2
import numpy as np
from tqdm import tqdm
import pkg_resources as pkg

import paddle
from paddle.inference import Config
from paddle.inference import create_predictor
from dataset import COCOValDataset
from post_process import YOLOPostProcess, coco_metric


def argsparser():
    """
    argsparser func
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model_path", type=str, help="inference model filepath")
    parser.add_argument(
        "--image_file",
        type=str,
        default=None,
        help="image path, if set image_file, it will not eval coco.")
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="dataset/coco",
        help="COCO dataset dir.")
    parser.add_argument(
        "--val_image_dir",
        type=str,
        default="val2017",
        help="COCO dataset val image dir.")
    parser.add_argument(
        "--val_anno_path",
        type=str,
        default="annotations/instances_val2017.json",
        help="COCO dataset anno path.")
    parser.add_argument(
        "--benchmark",
        type=bool,
        default=False,
        help="Whether run benchmark or not.")
    parser.add_argument(
        "--use_dynamic_shape",
        type=bool,
        default=True,
        help="Whether use dynamic shape or not.")
    parser.add_argument(
        "--use_trt",
        type=bool,
        default=False,
        help="Whether use TensorRT or not.")
    parser.add_argument(
        "--precision",
        type=str,
        default="paddle",
        help="mode of running(fp32/fp16/int8)")
    parser.add_argument(
        "--device",
        type=str,
        default="GPU",
        help="Choose the device you want to run, it can be: CPU/GPU/XPU, default is GPU",
    )
    parser.add_argument(
        "--arch", type=str, default="YOLOv5", help="architectures name.")
    parser.add_argument("--img_shape", type=int, default=640, help="input_size")
    parser.add_argument(
        "--batch_size", type=int, default=1, help="Batch size of model input.")
    parser.add_argument(
        "--use_mkldnn",
        type=bool,
        default=False,
        help="Whether use mkldnn or not.")
    parser.add_argument(
        "--cpu_threads", type=int, default=1, help="Num of cpu threads.")
    return parser


CLASS_LABEL = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
    'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
    'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
    'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
    'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
    'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant',
    'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
    'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush'
]


def preprocess(image, input_size, mean=None, std=None, swap=(2, 0, 1)):
    """
    image preprocess func
    """
    if len(image.shape) == 3:
        padded_img = np.ones((input_size[0], input_size[1], 3)) * 114.0
    else:
        padded_img = np.ones(input_size) * 114.0
    img = np.array(image)
    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    resized_img = cv2.resize(
        img,
        (int(img.shape[1] * r), int(img.shape[0] * r)),
        interpolation=cv2.INTER_LINEAR, ).astype(np.float32)
    padded_img[:int(img.shape[0] * r), :int(img.shape[1] * r)] = resized_img

    padded_img = padded_img[:, :, ::-1]
    padded_img /= 255.0
    if mean is not None:
        padded_img -= mean
    if std is not None:
        padded_img /= std
    padded_img = padded_img.transpose(swap)
    padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
    return padded_img, r


def get_color_map_list(num_classes):
    """
    get_color_map_list func
    """
    color_map = num_classes * [0, 0, 0]
    for i in range(0, num_classes):
        j = 0
        lab = i
        while lab:
            color_map[i * 3] |= ((lab >> 0) & 1) << (7 - j)
            color_map[i * 3 + 1] |= ((lab >> 1) & 1) << (7 - j)
            color_map[i * 3 + 2] |= ((lab >> 2) & 1) << (7 - j)
            j += 1
            lab >>= 3
    color_map = [color_map[i:i + 3] for i in range(0, len(color_map), 3)]
    return color_map


def draw_box(img, boxes, scores, cls_ids, conf=0.5, class_names=None):
    """
    draw_box func
    """
    color_list = get_color_map_list(len(class_names))
    for i, _ in enumerate(boxes):
        box = boxes[i]
        cls_id = int(cls_ids[i])
        color = tuple(color_list[cls_id])
        score = scores[i]
        if score < conf:
            continue
        x0 = int(box[0])
        y0 = int(box[1])
        x1 = int(box[2])
        y1 = int(box[3])

        text = "{}:{:.1f}%".format(class_names[cls_id], score * 100)
        font = cv2.FONT_HERSHEY_SIMPLEX

        txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]
        cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)
        cv2.rectangle(img, (x0, y0 + 1), (
            x0 + txt_size[0] + 1, y0 + int(1.5 * txt_size[1])), color, -1)
        cv2.putText(
            img,
            text, (x0, y0 + txt_size[1]),
            font,
            0.8, (0, 255, 0),
            thickness=2)

    return img


def get_current_memory_mb():
    """
    It is used to Obtain the memory usage of the CPU and GPU during the running of the program.
    And this function Current program is time-consuming.
    """
    import pynvml
    import psutil
    import GPUtil

    gpu_id = int(os.environ.get("CUDA_VISIBLE_DEVICES", 0))

    pid = os.getpid()
    p = psutil.Process(pid)
    info = p.memory_full_info()
    cpu_mem = info.uss / 1024.0 / 1024.0
    gpu_mem = 0
    gpu_percent = 0
    gpus = GPUtil.getGPUs()
    if gpu_id is not None and len(gpus) > 0:
        gpu_percent = gpus[gpu_id].load
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        gpu_mem = meminfo.used / 1024.0 / 1024.0
    return round(cpu_mem, 4), round(gpu_mem, 4)


def load_predictor(
        model_dir,
        precision="fp32",
        use_trt=False,
        use_mkldnn=False,
        batch_size=1,
        device="CPU",
        min_subgraph_size=3,
        use_dynamic_shape=False,
        trt_min_shape=1,
        trt_max_shape=1280,
        trt_opt_shape=640,
        cpu_threads=1, ):
    """set AnalysisConfig, generate AnalysisPredictor
    Args:
        model_dir (str): root path of __model__ and __params__
        precision (str): mode of running(fp32/fp16/int8)
        use_trt (bool): whether use TensorRT or not.
        use_mkldnn (bool): whether use MKLDNN or not in CPU.
        device (str): Choose the device you want to run, it can be: CPU/GPU, default is CPU
        use_dynamic_shape (bool): use dynamic shape or not
        trt_min_shape (int): min shape for dynamic shape in trt
        trt_max_shape (int): max shape for dynamic shape in trt
        trt_opt_shape (int): opt shape for dynamic shape in trt
    Returns:
        predictor (PaddlePredictor): AnalysisPredictor
    Raises:
        ValueError: predict by TensorRT need device == 'GPU'.
    """
    rerun_flag = False
    if device != "GPU" and use_trt:
        raise ValueError(
            "Predict by TensorRT mode: {}, expect device=='GPU', but device == {}".
            format(precision, device))
    config = Config(
        os.path.join(model_dir, "model.pdmodel"),
        os.path.join(model_dir, "model.pdiparams"))
    if device == "GPU":
        # initial GPU memory(M), device ID
        config.enable_use_gpu(200, 0)
        # optimize graph and fuse op
        config.switch_ir_optim(True)
    else:
        config.disable_gpu()
        config.set_cpu_math_library_num_threads(cpu_threads)
        config.switch_ir_optim()
        if use_mkldnn:
            config.enable_mkldnn()
            if precision == "int8":
                config.enable_mkldnn_int8({"conv2d", "transpose2", "pool2d"})

    precision_map = {
        "int8": Config.Precision.Int8,
        "fp32": Config.Precision.Float32,
        "fp16": Config.Precision.Half,
    }
    if precision in precision_map.keys() and use_trt:
        config.enable_tensorrt_engine(
            workspace_size=(1 << 25) * batch_size,
            max_batch_size=batch_size,
            min_subgraph_size=min_subgraph_size,
            precision_mode=precision_map[precision],
            use_static=True,
            use_calib_mode=False, )

        if use_dynamic_shape:
            dynamic_shape_file = os.path.join(FLAGS.model_path,
                                              "dynamic_shape.txt")
            if os.path.exists(dynamic_shape_file):
                config.enable_tuned_tensorrt_dynamic_shape(dynamic_shape_file,
                                                           True)
                print("trt set dynamic shape done!")
            else:
                config.collect_shape_range_info(dynamic_shape_file)
                print("Start collect dynamic shape...")
                rerun_flag = True

    # enable shared memory
    config.enable_memory_optim()
    predictor = create_predictor(config)
    return predictor, rerun_flag


def eval(predictor, val_loader, anno_file, rerun_flag=False):
    """
    eval main func
    """
    bboxes_list, bbox_nums_list, image_id_list = [], [], []
    cpu_mems, gpu_mems = 0, 0
    sample_nums = len(val_loader)
    predict_time = 0.0
    time_min = float("inf")
    time_max = float("-inf")
    input_names = predictor.get_input_names()
    output_names = predictor.get_output_names()
    boxes_tensor = predictor.get_output_handle(output_names[0])
    for batch_id, data in enumerate(val_loader):
        data_all = {k: np.array(v) for k, v in data.items()}
        inputs = {}
        if FLAGS.arch == "YOLOv6":
            inputs["x2paddle_image_arrays"] = data_all["image"]
        else:
            inputs["x2paddle_images"] = data_all["image"]
        for i, _ in enumerate(input_names):
            input_tensor = predictor.get_input_handle(input_names[i])
            input_tensor.copy_from_cpu(inputs[input_names[i]])
        start_time = time.time()
        predictor.run()
        outs = boxes_tensor.copy_to_cpu()
        end_time = time.time()
        timed = end_time - start_time
        time_min = min(time_min, timed)
        time_max = max(time_max, timed)
        predict_time += timed
        if rerun_flag:
            return
        postprocess = YOLOPostProcess(
            score_threshold=0.001, nms_threshold=0.65, multi_label=True)
        res = postprocess(np.array(outs), data_all["scale_factor"])
        bboxes_list.append(res["bbox"])
        bbox_nums_list.append(res["bbox_num"])
        image_id_list.append(np.array(data_all["im_id"]))
        cpu_mem, gpu_mem = get_current_memory_mb()
        cpu_mems += cpu_mem
        gpu_mems += gpu_mem
        if batch_id % 100 == 0:
            print("Eval iter:", batch_id)
            sys.stdout.flush()
    print("[Benchmark]Avg cpu_mem:{} MB, avg gpu_mem: {} MB".format(
        cpu_mems / sample_nums, gpu_mems / sample_nums))
    time_avg = predict_time / sample_nums
    print("[Benchmark]Inference time(ms): min={}, max={}, avg={}".format(
        round(time_min * 1000, 2),
        round(time_max * 1000, 1), round(time_avg * 1000, 1)))

    map_res = coco_metric(anno_file, bboxes_list, bbox_nums_list, image_id_list)
    print("[Benchmark] COCO mAP: {}".format(map_res[0]))
    sys.stdout.flush()


def infer(predictor):
    """
    infer image main func
    """
    warmup, repeats = 1, 1
    if FLAGS.benchmark:
        warmup, repeats = 50, 100
    origin_img = cv2.imread(FLAGS.image_file)
    input_image, scale_factor = preprocess(origin_img,
                                           [FLAGS.img_shape, FLAGS.img_shape])
    input_image = np.expand_dims(input_image, axis=0)
    scale_factor = np.array([[scale_factor, scale_factor]])
    inputs = {}
    if FLAGS.arch == "YOLOv6":
        inputs["x2paddle_image_arrays"] = input_image
    else:
        inputs["x2paddle_images"] = input_image
    input_names = predictor.get_input_names()
    for i, _ in enumerate(input_names):
        input_tensor = predictor.get_input_handle(input_names[i])
        input_tensor.copy_from_cpu(inputs[input_names[i]])

    for i in range(warmup):
        predictor.run()

    np_boxes = None
    predict_time = 0.0
    time_min = float("inf")
    time_max = float("-inf")
    cpu_mems, gpu_mems = 0, 0
    for i in range(repeats):
        start_time = time.time()
        predictor.run()
        output_names = predictor.get_output_names()
        boxes_tensor = predictor.get_output_handle(output_names[0])
        np_boxes = boxes_tensor.copy_to_cpu()
        end_time = time.time()
        timed = end_time - start_time
        time_min = min(time_min, timed)
        time_max = max(time_max, timed)
        predict_time += timed
        cpu_mem, gpu_mem = get_current_memory_mb()
        cpu_mems += cpu_mem
        gpu_mems += gpu_mem
    print("[Benchmark]Avg cpu_mem:{} MB, avg gpu_mem: {} MB".format(
        cpu_mems / repeats, gpu_mems / repeats))

    time_avg = predict_time / repeats
    print("[Benchmark]Inference time(ms): min={}, max={}, avg={}".format(
        round(time_min * 1000, 2),
        round(time_max * 1000, 1), round(time_avg * 1000, 1)))
    postprocess = YOLOPostProcess(
        score_threshold=0.001, nms_threshold=0.65, multi_label=True)
    res = postprocess(np_boxes, scale_factor)
    # Draw rectangles and labels on the original image
    dets = res["bbox"]
    if dets is not None:
        final_boxes, final_scores, final_class = dets[:, 2:], dets[:,
                                                                   1], dets[:,
                                                                            0]
        res_img = draw_box(
            origin_img,
            final_boxes,
            final_scores,
            final_class,
            conf=0.5,
            class_names=CLASS_LABEL)
        cv2.imwrite("output.jpg", res_img)
        print("The prediction results are saved in output.jpg.")


def main():
    """
    main func
    """
    predictor, rerun_flag = load_predictor(
        FLAGS.model_path,
        device=FLAGS.device,
        use_trt=FLAGS.use_trt,
        use_mkldnn=FLAGS.use_mkldnn,
        precision=FLAGS.precision,
        use_dynamic_shape=FLAGS.use_dynamic_shape,
        cpu_threads=FLAGS.cpu_threads, )

    if FLAGS.image_file:
        infer(predictor)
    else:
        dataset = COCOValDataset(
            dataset_dir=FLAGS.dataset_dir,
            image_dir=FLAGS.val_image_dir,
            anno_path=FLAGS.val_anno_path)
        anno_file = dataset.ann_file
        val_loader = paddle.io.DataLoader(
            dataset, batch_size=FLAGS.batch_size, drop_last=True)
        eval(predictor, val_loader, anno_file, rerun_flag=rerun_flag)

    if rerun_flag:
        print(
            "***** Collect dynamic shape done, Please rerun the program to get correct results. *****"
        )


if __name__ == "__main__":
    paddle.enable_static()
    parser = argsparser()
    FLAGS = parser.parse_args()

    # DataLoader need run on cpu
    paddle.set_device("cpu")

    main()
