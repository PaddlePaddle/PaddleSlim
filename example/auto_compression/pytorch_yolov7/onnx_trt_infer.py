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

import numpy as np
import cv2
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import os
import time
import random
import argparse

EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
EXPLICIT_PRECISION = 1 << (
    int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_PRECISION)

# load coco labels
CLASS_LABEL = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
]


def preprocess(image, input_size, mean=None, std=None, swap=(2, 0, 1)):
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


def postprocess(predictions, ratio):
    boxes = predictions[:, :4]
    scores = predictions[:, 4:5] * predictions[:, 5:]
    boxes_xyxy = np.ones_like(boxes)
    boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.
    boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.
    boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.
    boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.
    boxes_xyxy /= ratio
    dets = multiclass_nms(boxes_xyxy, scores, nms_thr=0.45, score_thr=0.1)
    return dets


def nms(boxes, scores, nms_thr):
    """Single class NMS implemented in Numpy."""
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= nms_thr)[0]
        order = order[inds + 1]

    return keep


def multiclass_nms(boxes, scores, nms_thr, score_thr):
    """Multiclass NMS implemented in Numpy"""
    final_dets = []
    num_classes = scores.shape[1]
    for cls_ind in range(num_classes):
        cls_scores = scores[:, cls_ind]
        valid_score_mask = cls_scores > score_thr
        if valid_score_mask.sum() == 0:
            continue
        else:
            valid_scores = cls_scores[valid_score_mask]
            valid_boxes = boxes[valid_score_mask]
            keep = nms(valid_boxes, valid_scores, nms_thr)
            if len(keep) > 0:
                cls_inds = np.ones((len(keep), 1)) * cls_ind
                dets = np.concatenate(
                    [valid_boxes[keep], valid_scores[keep, None], cls_inds], 1)
                final_dets.append(dets)
    if len(final_dets) == 0:
        return None
    return np.concatenate(final_dets, 0)


def get_color_map_list(num_classes):
    color_map = num_classes * [0, 0, 0]
    for i in range(0, num_classes):
        j = 0
        lab = i
        while lab:
            color_map[i * 3] |= (((lab >> 0) & 1) << (7 - j))
            color_map[i * 3 + 1] |= (((lab >> 1) & 1) << (7 - j))
            color_map[i * 3 + 2] |= (((lab >> 2) & 1) << (7 - j))
            j += 1
            lab >>= 3
    color_map = [color_map[i:i + 3] for i in range(0, len(color_map), 3)]
    return color_map


def draw_box(img, boxes, scores, cls_ids, conf=0.5, class_names=None):
    color_list = get_color_map_list(len(class_names))
    for i in range(len(boxes)):
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

        text = '{}:{:.1f}%'.format(class_names[cls_id], score * 100)
        font = cv2.FONT_HERSHEY_SIMPLEX

        txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]
        cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)
        cv2.rectangle(img, (x0, y0 + 1),
                      (x0 + txt_size[0] + 1, y0 + int(1.5 * txt_size[1])),
                      color, -1)
        cv2.putText(
            img,
            text, (x0, y0 + txt_size[1]),
            font,
            0.8, (0, 255, 0),
            thickness=2)

    return img


def get_engine(precision, model_file_path):
    # TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)
    TRT_LOGGER = trt.Logger()
    builder = trt.Builder(TRT_LOGGER)
    config = builder.create_builder_config()
    if precision == 'int8':
        network = builder.create_network(EXPLICIT_BATCH | EXPLICIT_PRECISION)
    else:
        network = builder.create_network(EXPLICIT_BATCH)
    parser = trt.OnnxParser(network, TRT_LOGGER)

    runtime = trt.Runtime(TRT_LOGGER)
    if model_file_path.endswith('.trt'):
        # If a serialized engine exists, use it instead of building an engine.
        print("Reading engine from file {}".format(model_file_path))
        with open(model_file_path,
                  "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())
            for i in range(network.num_layers):
                layer = network.get_layer(i)
                print(i, layer.name)
            return engine
    else:
        config.max_workspace_size = 1 << 30

        if precision == "fp16":
            if not builder.platform_has_fast_fp16:
                print("FP16 is not supported natively on this platform/device")
            else:
                config.set_flag(trt.BuilderFlag.FP16)
        elif precision == "int8":
            if not builder.platform_has_fast_int8:
                print("INT8 is not supported natively on this platform/device")
            else:
                if builder.platform_has_fast_fp16:
                    # Also enable fp16, as some layers may be even more efficient in fp16 than int8
                    config.set_flag(trt.BuilderFlag.FP16)
                config.set_flag(trt.BuilderFlag.INT8)

        builder.max_batch_size = 1
        print('Loading ONNX file from path {}...'.format(model_file_path))
        with open(model_file_path, 'rb') as model:
            print('Beginning ONNX file parsing')
            if not parser.parse(model.read()):
                print('ERROR: Failed to parse the ONNX file.')
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                return None

        print('Completed parsing of ONNX file')
        print('Building an engine from file {}; this may take a while...'.
              format(model_file_path))
        plan = builder.build_serialized_network(network, config)
        engine = runtime.deserialize_cuda_engine(plan)
        print("Completed creating Engine")
        with open(model_file_path, "wb") as f:
            f.write(engine.serialize())
        for i in range(network.num_layers):
            layer = network.get_layer(i)
            print(i, layer.name)
        return engine


# Simple helper data class that's a little nicer to use than a 2-tuple.
class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


def allocate_buffers(engine):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(
            binding)) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings, stream


def run_inference(context, bindings, inputs, outputs, stream):
    # Transfer input data to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference.
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    return [out.host for out in outputs]


def main(args):
    onnx_model = args.model_path
    img_path = args.image_file
    num_class = len(CLASS_LABEL)
    repeat = 1000
    engine = get_engine(args.precision, onnx_model)

    model_all_names = []
    for idx in range(engine.num_bindings):
        is_input = engine.binding_is_input(idx)
        name = engine.get_binding_name(idx)
        op_type = engine.get_binding_dtype(idx)
        model_all_names.append(name)
        shape = engine.get_binding_shape(idx)
        print('input id:', idx, '   is input: ', is_input, '  binding name:',
              name, '  shape:', shape, 'type: ', op_type)

    context = engine.create_execution_context()
    print('Allocate buffers ...')
    inputs, outputs, bindings, stream = allocate_buffers(engine)
    print("TRT set input ...")

    origin_img = cv2.imread(img_path)
    input_shape = [args.img_shape, args.img_shape]
    input_image, ratio = preprocess(origin_img, input_shape)

    inputs[0].host = np.expand_dims(input_image, axis=0)

    for _ in range(0, 50):
        trt_outputs = run_inference(
            context,
            bindings=bindings,
            inputs=inputs,
            outputs=outputs,
            stream=stream)

    time1 = time.time()
    for _ in range(0, repeat):
        trt_outputs = run_inference(
            context,
            bindings=bindings,
            inputs=inputs,
            outputs=outputs,
            stream=stream)
    time2 = time.time()
    # total time cost(ms)
    total_inference_cost = (time2 - time1) * 1000
    print("model path: ", onnx_model, " precision: ", args.precision)
    print("In TensorRT, ",
          "average latency is : {} ms".format(total_inference_cost / repeat))
    # Do postprocess
    output = trt_outputs[0]
    predictions = np.reshape(output, (1, -1, int(5 + num_class)))[0]
    dets = postprocess(predictions, ratio)
    # Draw rectangles and labels on the original image
    if dets is not None:
        final_boxes, final_scores, final_cls_inds = dets[:, :
                                                         4], dets[:, 4], dets[:,
                                                                              5]
        origin_img = draw_box(
            origin_img,
            final_boxes,
            final_scores,
            final_cls_inds,
            conf=0.5,
            class_names=CLASS_LABEL)
    cv2.imwrite('output.jpg', origin_img)
    print('The prediction results are saved in output.jpg.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_path',
        type=str,
        default="quant_model.onnx",
        help="inference model filepath")
    parser.add_argument(
        '--image_file', type=str, default="bus.jpg", help="image path")
    parser.add_argument(
        '--precision', type=str, default='fp32', help="support fp32/fp16/int8.")
    parser.add_argument('--img_shape', type=int, default=640, help="input_size")
    args = parser.parse_args()
    main(args)
