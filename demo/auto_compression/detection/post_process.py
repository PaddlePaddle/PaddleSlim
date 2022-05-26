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


def box_area(boxes):
    """
    Args:
        boxes(np.ndarray): [N, 4]
    return: [N]
    """
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def box_iou(box1, box2):
    """
    Args:
        box1(np.ndarray): [N, 4]
        box2(np.ndarray): [M, 4]
    return: [N, M]
    """
    area1 = box_area(box1)
    area2 = box_area(box2)
    lt = np.maximum(box1[:, np.newaxis, :2], box2[:, :2])
    rb = np.minimum(box1[:, np.newaxis, 2:], box2[:, 2:])
    wh = rb - lt
    wh = np.maximum(0, wh)
    inter = wh[:, :, 0] * wh[:, :, 1]
    iou = inter / (area1[:, np.newaxis] + area2 - inter)
    return iou


def nms(boxes, scores, iou_threshold):
    """
    Non Max Suppression numpy implementation.
    args:
        boxes(np.ndarray): [N, 4]
        scores(np.ndarray): [N, 1]
        iou_threshold(float): Threshold of IoU.
    """
    idxs = scores.argsort()
    keep = []
    while idxs.size > 0:
        max_score_index = idxs[-1]
        max_score_box = boxes[max_score_index][None, :]
        keep.append(max_score_index)
        if idxs.size == 1:
            break
        idxs = idxs[:-1]
        other_boxes = boxes[idxs]
        ious = box_iou(max_score_box, other_boxes)
        idxs = idxs[ious[0] <= iou_threshold]

    keep = np.array(keep)
    return keep


class YOLOv5PostProcess(object):
    """
    Post process of YOLOv5 network.
    args:
        score_threshold(float): Threshold to filter out bounding boxes with low 
                confidence score. If not provided, consider all boxes.
        nms_threshold(float): The threshold to be used in NMS.
        multi_label(bool): Whether keep multi label in boxes.
        keep_top_k(int): Number of total bboxes to be kept per image after NMS
                step. -1 means keeping all bboxes after NMS step.
    """

    def __init__(self,
                 score_threshold=0.25,
                 nms_threshold=0.5,
                 multi_label=False,
                 keep_top_k=300):
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.multi_label = multi_label
        self.keep_top_k = keep_top_k

    def _xywh2xyxy(self, x):
        # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
        y = np.copy(x)
        y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
        y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
        y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
        y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
        return y

    def _non_max_suppression(self, prediction):
        max_wh = 4096  # (pixels) minimum and maximum box width and height
        nms_top_k = 30000  # 

        cand_boxes = prediction[..., 4] > self.score_threshold  # candidates
        output = [np.zeros((0, 6))] * prediction.shape[0]

        for batch_id, boxes in enumerate(
                prediction):  # image index, image inference
            # Apply constraints
            boxes = boxes[cand_boxes[batch_id]]  # confidence
            if not boxes.shape[0]:
                continue
            # Compute conf (conf = obj_conf * cls_conf)
            boxes[:, 5:] *= boxes[:, 4:5]

            # Box (center x, center y, width, height) to (x1, y1, x2, y2)
            convert_box = self._xywh2xyxy(boxes[:, :4])

            # Detections matrix nx6 (xyxy, conf, cls)
            if self.multi_label:
                i, j = (boxes[:, 5:] > self.score_threshold).nonzero()
                boxes = np.concatenate(
                    (convert_box[i], boxes[i, j + 5, None],
                     j[:, None].astype(np.float32)),
                    axis=1)
            else:
                conf = np.max(boxes[:, 5:], axis=1)
                j = np.argmax(boxes[:, 5:], axis=1)
                re = np.array(conf.reshape(-1) > self.score_threshold)
                conf = conf.reshape(-1, 1)
                j = j.reshape(-1, 1)
                boxes = np.concatenate((convert_box, conf, j), axis=1)[re]

            num_box = boxes.shape[0]
            if not num_box:
                continue
            elif num_box > nms_top_k:
                boxes = boxes[boxes[:, 4].argsort()[::-1][:nms_top_k]]

            # Batched NMS
            c = boxes[:, 5:6] * max_wh
            clean_boxes, scores = boxes[:, :4] + c, boxes[:, 4]
            keep = nms(clean_boxes, scores, self.nms_threshold)
            # limit detection box num
            if keep.shape[0] > self.keep_top_k:
                keep = keep[:self.keep_top_k]
            output[batch_id] = boxes[keep]
        return output

    def __call__(self, outs, scale_factor):
        preds = self._non_max_suppression(outs)
        bboxs, box_nums = [], []
        for i, pred in enumerate(preds):
            if len(pred.shape) > 2:
                pred = np.squeeze(pred)
            if len(pred.shape) == 1:
                pred = pred[np.newaxis, :]
            pred_bboxes = pred[:, :4]
            scale_factor = np.tile(scale_factor[i][::-1], (1, 2))
            pred_bboxes /= scale_factor
            bbox = np.concatenate(
                [
                    pred[:, -1][:, np.newaxis], pred[:, -2][:, np.newaxis],
                    pred_bboxes
                ],
                axis=-1)
            bboxs.append(bbox)
            box_num = bbox.shape[0]
            box_nums.append(box_num)
        bboxs = np.concatenate(bboxs, axis=0)
        box_nums = np.array(box_nums)
        return {'bbox': bboxs, 'bbox_num': box_nums}
