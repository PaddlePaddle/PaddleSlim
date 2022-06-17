
# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved. 
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
import logging
import os
import json
from collections import defaultdict, OrderedDict
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from scipy.io import loadmat, savemat
import cv2
from paddleslim.common import get_logger
logger = get_logger(__name__, level=logging.INFO)

def get_affine_mat_kernel(h, w, s, inv=False):
    if w < h:
        w_ = s
        h_ = int(np.ceil((s / w * h) / 64.) * 64)
        scale_w = w
        scale_h = h_ / w_ * w

    else:
        h_ = s
        w_ = int(np.ceil((s / h * w) / 64.) * 64)
        scale_h = h
        scale_w = w_ / h_ * h

    center = np.array([np.round(w / 2.), np.round(h / 2.)])

    size_resized = (w_, h_)
    trans = get_affine_transform(
        center, np.array([scale_w, scale_h]), 0, size_resized, inv=inv)

    return trans, size_resized


def get_affine_transform(center,
                         input_size,
                         rot,
                         output_size,
                         shift=(0., 0.),
                         inv=False):
    """Get the affine transform matrix, given the center/scale/rot/output_size.
    Args:
        center (np.ndarray[2, ]): Center of the bounding box (x, y).
        input_size (np.ndarray[2, ]): Size of input feature (width, height).
        rot (float): Rotation angle (degree).
        output_size (np.ndarray[2, ]): Size of the destination heatmaps.
        shift (0-100%): Shift translation ratio wrt the width/height.
            Default (0., 0.).
        inv (bool): Option to inverse the affine transform direction.
            (inv=False: src->dst or inv=True: dst->src)
    Returns:
        np.ndarray: The transform matrix.
    """
    assert len(center) == 2
    assert len(output_size) == 2
    assert len(shift) == 2

    if not isinstance(input_size, (np.ndarray, list)):
        input_size = np.array([input_size, input_size], dtype=np.float32)
    scale_tmp = input_size

    shift = np.array(shift)
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = rotate_point([0., src_w * -0.5], rot_rad)
    dst_dir = np.array([0., dst_w * -0.5])

    src = np.zeros((3, 2), dtype=np.float32)

    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    src[2, :] = _get_3rd_point(src[0, :], src[1, :])

    dst = np.zeros((3, 2), dtype=np.float32)
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir
    dst[2, :] = _get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


def get_warp_matrix(theta, size_input, size_dst, size_target):
    """This code is based on
        https://github.com/open-mmlab/mmpose/blob/master/mmpose/core/post_processing/post_transforms.py
        Calculate the transformation matrix under the constraint of unbiased.
    Paper ref: Huang et al. The Devil is in the Details: Delving into Unbiased
    Data Processing for Human Pose Estimation (CVPR 2020).
    Args:
        theta (float): Rotation angle in degrees.
        size_input (np.ndarray): Size of input image [w, h].
        size_dst (np.ndarray): Size of output image [w, h].
        size_target (np.ndarray): Size of ROI in input plane [w, h].
    Returns:
        matrix (np.ndarray): A matrix for transformation.
    """
    theta = np.deg2rad(theta)
    matrix = np.zeros((2, 3), dtype=np.float32)
    scale_x = size_dst[0] / size_target[0]
    scale_y = size_dst[1] / size_target[1]
    matrix[0, 0] = np.cos(theta) * scale_x
    matrix[0, 1] = -np.sin(theta) * scale_x
    matrix[0, 2] = scale_x * (
        -0.5 * size_input[0] * np.cos(theta) + 0.5 * size_input[1] *
        np.sin(theta) + 0.5 * size_target[0])
    matrix[1, 0] = np.sin(theta) * scale_y
    matrix[1, 1] = np.cos(theta) * scale_y
    matrix[1, 2] = scale_y * (
        -0.5 * size_input[0] * np.sin(theta) - 0.5 * size_input[1] *
        np.cos(theta) + 0.5 * size_target[1])
    return matrix


def _get_3rd_point(a, b):
    """To calculate the affine matrix, three pairs of points are required. This
    function is used to get the 3rd point, given 2D points a & b.
    The 3rd point is defined by rotating vector `a - b` by 90 degrees
    anticlockwise, using b as the rotation center.
    Args:
        a (np.ndarray): point(x,y)
        b (np.ndarray): point(x,y)
    Returns:
        np.ndarray: The 3rd point.
    """
    assert len(
        a) == 2, 'input of _get_3rd_point should be point with length of 2'
    assert len(
        b) == 2, 'input of _get_3rd_point should be point with length of 2'
    direction = a - b
    third_pt = b + np.array([-direction[1], direction[0]], dtype=np.float32)

    return third_pt


def rotate_point(pt, angle_rad):
    """Rotate a point by an angle.
    Args:
        pt (list[float]): 2 dimensional point to be rotated
        angle_rad (float): rotation angle by radian
    Returns:
        list[float]: Rotated point.
    """
    assert len(pt) == 2
    sn, cs = np.sin(angle_rad), np.cos(angle_rad)
    new_x = pt[0] * cs - pt[1] * sn
    new_y = pt[0] * sn + pt[1] * cs
    rotated_pt = [new_x, new_y]

    return rotated_pt


def transpred(kpts, h, w, s):
    trans, _ = get_affine_mat_kernel(h, w, s, inv=True)

    return warp_affine_joints(kpts[..., :2].copy(), trans)


def warp_affine_joints(joints, mat):
    """Apply affine transformation defined by the transform matrix on the
    joints.
    Args:
        joints (np.ndarray[..., 2]): Origin coordinate of joints.
        mat (np.ndarray[3, 2]): The affine matrix.
    Returns:
        matrix (np.ndarray[..., 2]): Result coordinate of joints.
    """
    joints = np.array(joints)
    shape = joints.shape
    joints = joints.reshape(-1, 2)
    return np.dot(np.concatenate(
        (joints, joints[:, 0:1] * 0 + 1), axis=1),
                  mat.T).reshape(shape)


def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.]).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]


def transform_preds(coords, center, scale, output_size):
    target_coords = np.zeros(coords.shape)
    trans = get_affine_transform(center, scale * 200, 0, output_size, inv=1)
    for p in range(coords.shape[0]):
        target_coords[p, 0:2] = affine_transform(coords[p, 0:2], trans)
    return target_coords


def oks_iou(g, d, a_g, a_d, sigmas=None, in_vis_thre=None):
    if not isinstance(sigmas, np.ndarray):
        sigmas = np.array([
            .26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62, 1.07, 1.07,
            .87, .87, .89, .89
        ]) / 10.0
    vars = (sigmas * 2)**2
    xg = g[0::3]
    yg = g[1::3]
    vg = g[2::3]
    ious = np.zeros((d.shape[0]))
    for n_d in range(0, d.shape[0]):
        xd = d[n_d, 0::3]
        yd = d[n_d, 1::3]
        vd = d[n_d, 2::3]
        dx = xd - xg
        dy = yd - yg
        e = (dx**2 + dy**2) / vars / ((a_g + a_d[n_d]) / 2 + np.spacing(1)) / 2
        if in_vis_thre is not None:
            ind = list(vg > in_vis_thre) and list(vd > in_vis_thre)
            e = e[ind]
        ious[n_d] = np.sum(np.exp(-e)) / e.shape[0] if e.shape[0] != 0 else 0.0
    return ious

def oks_nms(kpts_db, thresh, sigmas=None, in_vis_thre=None):
    """greedily select boxes with high confidence and overlap with current maximum <= thresh
    rule out overlap >= thresh
    Args:
        kpts_db (list): The predicted keypoints within the image
        thresh (float): The threshold to select the boxes
        sigmas (np.array): The variance to calculate the oks iou
            Default: None
        in_vis_thre (float): The threshold to select the high confidence boxes
            Default: None
    Return:
        keep (list): indexes to keep
    """

    if len(kpts_db) == 0:
        return []

    scores = np.array([kpts_db[i]['score'] for i in range(len(kpts_db))])
    kpts = np.array(
        [kpts_db[i]['keypoints'].flatten() for i in range(len(kpts_db))])
    areas = np.array([kpts_db[i]['area'] for i in range(len(kpts_db))])

    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        oks_ovr = oks_iou(kpts[i], kpts[order[1:]], areas[i], areas[order[1:]],
                          sigmas, in_vis_thre)

        inds = np.where(oks_ovr <= thresh)[0]
        order = order[inds + 1]

    return keep

def rescore(overlap, scores, thresh, type='gaussian'):
    assert overlap.shape[0] == scores.shape[0]
    if type == 'linear':
        inds = np.where(overlap >= thresh)[0]
        scores[inds] = scores[inds] * (1 - overlap[inds])
    else:
        scores = scores * np.exp(-overlap**2 / thresh)

    return scores


def soft_oks_nms(kpts_db, thresh, sigmas=None, in_vis_thre=None):
    """greedily select boxes with high confidence and overlap with current maximum <= thresh
    rule out overlap >= thresh
    Args:
        kpts_db (list): The predicted keypoints within the image
        thresh (float): The threshold to select the boxes
        sigmas (np.array): The variance to calculate the oks iou
            Default: None
        in_vis_thre (float): The threshold to select the high confidence boxes
            Default: None
    Return:
        keep (list): indexes to keep
    """

    if len(kpts_db) == 0:
        return []

    scores = np.array([kpts_db[i]['score'] for i in range(len(kpts_db))])
    kpts = np.array(
        [kpts_db[i]['keypoints'].flatten() for i in range(len(kpts_db))])
    areas = np.array([kpts_db[i]['area'] for i in range(len(kpts_db))])

    order = scores.argsort()[::-1]
    scores = scores[order]

    # max_dets = order.size
    max_dets = 20
    keep = np.zeros(max_dets, dtype=np.intp)
    keep_cnt = 0
    while order.size > 0 and keep_cnt < max_dets:
        i = order[0]

        oks_ovr = oks_iou(kpts[i], kpts[order[1:]], areas[i], areas[order[1:]],
                          sigmas, in_vis_thre)

        order = order[1:]
        scores = rescore(oks_ovr, scores[1:], thresh)

        tmp = scores.argsort()[::-1]
        order = order[tmp]
        scores = scores[tmp]

        keep[keep_cnt] = i
        keep_cnt += 1

    keep = keep[:keep_cnt]

    return keep


class KeyPointTopDownCOCOEval(object):
    """refer to
        https://github.com/leoxiaobin/deep-high-resolution-net.pytorch
        Copyright (c) Microsoft, under the MIT License.
    """

    def __init__(self,
                 anno_file,
                 num_samples,
                 num_joints,
                 output_eval,
                 iou_type='keypoints',
                 in_vis_thre=0.2,
                 oks_thre=0.9,
                 save_prediction_only=False):
        super(KeyPointTopDownCOCOEval, self).__init__()
        self.coco = COCO(anno_file)
        self.num_samples = num_samples
        self.num_joints = num_joints
        self.iou_type = iou_type
        self.in_vis_thre = in_vis_thre
        self.oks_thre = oks_thre
        self.output_eval = output_eval
        self.res_file = os.path.join(output_eval, "keypoints_results.json")
        self.save_prediction_only = save_prediction_only
        self.reset()

    def reset(self):
        self.results = {
            'all_preds': np.zeros(
                (self.num_samples, self.num_joints, 3), dtype=np.float32),
            'all_boxes': np.zeros((self.num_samples, 6)),
            'image_path': []
        }
        self.eval_results = {}
        self.idx = 0

    def update(self, inputs, outputs):
        kpts, _ = outputs['keypoint'][0]

        num_images = inputs['image'].shape[0]
        self.results['all_preds'][self.idx:self.idx + num_images, :, 0:
                                  3] = kpts[:, :, 0:3]
        self.results['all_boxes'][self.idx:self.idx + num_images, 0:2] = inputs[
            'center'][:, 0:2]
        self.results['all_boxes'][self.idx:self.idx + num_images, 2:4] = inputs[
            'scale'][:, 0:2]
        self.results['all_boxes'][self.idx:self.idx + num_images, 4] = np.prod(
            inputs['scale'] * 200, 1)
        self.results['all_boxes'][self.idx:self.idx + num_images, 5] = np.squeeze(inputs['score'])
        self.results['image_path'].extend(inputs['im_id'])

        self.idx += num_images

    def _write_coco_keypoint_results(self, keypoints):
        data_pack = [{
            'cat_id': 1,
            'cls': 'person',
            'ann_type': 'keypoints',
            'keypoints': keypoints
        }]
        results = self._coco_keypoint_results_one_category_kernel(data_pack[0])
        if not os.path.exists(self.output_eval):
            os.makedirs(self.output_eval)
        with open(self.res_file, 'w') as f:
            json.dump(results, f, sort_keys=True, indent=4)
            logger.info(f'The keypoint result is saved to {self.res_file}.')
        try:
            json.load(open(self.res_file))
        except Exception:
            content = []
            with open(self.res_file, 'r') as f:
                for line in f:
                    content.append(line)
            content[-1] = ']'
            with open(self.res_file, 'w') as f:
                for c in content:
                    f.write(c)

    def _coco_keypoint_results_one_category_kernel(self, data_pack):
        cat_id = data_pack['cat_id']
        keypoints = data_pack['keypoints']
        cat_results = []

        for img_kpts in keypoints:
            if len(img_kpts) == 0:
                continue

            _key_points = np.array(
                [img_kpts[k]['keypoints'] for k in range(len(img_kpts))])
            _key_points = _key_points.reshape(_key_points.shape[0], -1)

            result = [{
                'image_id': img_kpts[k]['image'],
                'category_id': cat_id,
                'keypoints': _key_points[k].tolist(),
                'score': img_kpts[k]['score'],
                'center': list(img_kpts[k]['center']),
                'scale': list(img_kpts[k]['scale'])
            } for k in range(len(img_kpts))]
            cat_results.extend(result)

        return cat_results

    def get_final_results(self, preds, all_boxes, img_path):
        _kpts = []
        for idx, kpt in enumerate(preds):
            _kpts.append({
                'keypoints': kpt,
                'center': all_boxes[idx][0:2],
                'scale': all_boxes[idx][2:4],
                'area': all_boxes[idx][4],
                'score': all_boxes[idx][5],
                'image': int(img_path[idx])
            })
        # image x person x (keypoints)
        kpts = defaultdict(list)
        for kpt in _kpts:
            kpts[kpt['image']].append(kpt)

        # rescoring and oks nms
        num_joints = preds.shape[1]
        in_vis_thre = self.in_vis_thre
        oks_thre = self.oks_thre
        oks_nmsed_kpts = []
        for img in kpts.keys():
            img_kpts = kpts[img]
            for n_p in img_kpts:
                box_score = n_p['score']
                kpt_score = 0
                valid_num = 0
                for n_jt in range(0, num_joints):
                    t_s = n_p['keypoints'][n_jt][2]
                    if t_s > in_vis_thre:
                        kpt_score = kpt_score + t_s
                        valid_num = valid_num + 1
                if valid_num != 0:
                    kpt_score = kpt_score / valid_num
                # rescoring
                n_p['score'] = kpt_score * box_score

            keep = oks_nms([img_kpts[i] for i in range(len(img_kpts))],
                           oks_thre)

            if len(keep) == 0:
                oks_nmsed_kpts.append(img_kpts)
            else:
                oks_nmsed_kpts.append([img_kpts[_keep] for _keep in keep])

        self._write_coco_keypoint_results(oks_nmsed_kpts)

    def accumulate(self):
        self.get_final_results(self.results['all_preds'],
                               self.results['all_boxes'],
                               self.results['image_path'])
        if self.save_prediction_only:
            logger.info(f'The keypoint result is saved to {self.res_file} '
                        'and do not evaluate the mAP.')
            return
        coco_dt = self.coco.loadRes(self.res_file)
        coco_eval = COCOeval(self.coco, coco_dt, 'keypoints')
        coco_eval.params.useSegm = None
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        keypoint_stats = []
        for ind in range(len(coco_eval.stats)):
            keypoint_stats.append((coco_eval.stats[ind]))
        self.eval_results['keypoint'] = keypoint_stats

    def log(self):
        if self.save_prediction_only:
            return
        stats_names = [
            'AP', 'Ap .5', 'AP .75', 'AP (M)', 'AP (L)', 'AR', 'AR .5',
            'AR .75', 'AR (M)', 'AR (L)'
        ]
        num_values = len(stats_names)
        print(' '.join(['| {}'.format(name) for name in stats_names]) + ' |')
        print('|---' * (num_values + 1) + '|')

        print(' '.join([
            '| {:.3f}'.format(value) for value in self.eval_results['keypoint']
        ]) + ' |')

    def get_results(self):
        return self.eval_results
    

class HRNetPostProcess(object):
    def __init__(self, use_dark=True):
        self.use_dark = use_dark

    def get_max_preds(self, heatmaps):
        '''get predictions from score maps
        Args:
            heatmaps: numpy.ndarray([batch_size, num_joints, height, width])
        Returns:
            preds: numpy.ndarray([batch_size, num_joints, 2]), keypoints coords
            maxvals: numpy.ndarray([batch_size, num_joints, 2]), the maximum confidence of the keypoints
        '''
        assert isinstance(heatmaps,
                          np.ndarray), 'heatmaps should be numpy.ndarray'
        assert heatmaps.ndim == 4, 'batch_images should be 4-ndim'

        batch_size = heatmaps.shape[0]
        num_joints = heatmaps.shape[1]
        width = heatmaps.shape[3]
        heatmaps_reshaped = heatmaps.reshape((batch_size, num_joints, -1))
        idx = np.argmax(heatmaps_reshaped, 2)
        maxvals = np.amax(heatmaps_reshaped, 2)

        maxvals = maxvals.reshape((batch_size, num_joints, 1))
        idx = idx.reshape((batch_size, num_joints, 1))

        preds = np.tile(idx, (1, 1, 2)).astype(np.float32)

        preds[:, :, 0] = (preds[:, :, 0]) % width
        preds[:, :, 1] = np.floor((preds[:, :, 1]) / width)

        pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2))
        pred_mask = pred_mask.astype(np.float32)

        preds *= pred_mask

        return preds, maxvals

    def gaussian_blur(self, heatmap, kernel):
        border = (kernel - 1) // 2
        batch_size = heatmap.shape[0]
        num_joints = heatmap.shape[1]
        height = heatmap.shape[2]
        width = heatmap.shape[3]
        for i in range(batch_size):
            for j in range(num_joints):
                origin_max = np.max(heatmap[i, j])
                dr = np.zeros((height + 2 * border, width + 2 * border))
                dr[border:-border, border:-border] = heatmap[i, j].copy()
                dr = cv2.GaussianBlur(dr, (kernel, kernel), 0)
                heatmap[i, j] = dr[border:-border, border:-border].copy()
                heatmap[i, j] *= origin_max / np.max(heatmap[i, j])
        return heatmap

    def dark_parse(self, hm, coord):
        heatmap_height = hm.shape[0]
        heatmap_width = hm.shape[1]
        px = int(coord[0])
        py = int(coord[1])
        if 1 < px < heatmap_width - 2 and 1 < py < heatmap_height - 2:
            dx = 0.5 * (hm[py][px + 1] - hm[py][px - 1])
            dy = 0.5 * (hm[py + 1][px] - hm[py - 1][px])
            dxx = 0.25 * (hm[py][px + 2] - 2 * hm[py][px] + hm[py][px - 2])
            dxy = 0.25 * (hm[py+1][px+1] - hm[py-1][px+1] - hm[py+1][px-1] \
                + hm[py-1][px-1])
            dyy = 0.25 * (
                hm[py + 2 * 1][px] - 2 * hm[py][px] + hm[py - 2 * 1][px])
            derivative = np.matrix([[dx], [dy]])
            hessian = np.matrix([[dxx, dxy], [dxy, dyy]])
            if dxx * dyy - dxy**2 != 0:
                hessianinv = hessian.I
                offset = -hessianinv * derivative
                offset = np.squeeze(np.array(offset.T), axis=0)
                coord += offset
        return coord

    def dark_postprocess(self, hm, coords, kernelsize):
        '''DARK postpocessing, Zhang et al. Distribution-Aware Coordinate
        Representation for Human Pose Estimation (CVPR 2020).
        '''

        hm = self.gaussian_blur(hm, kernelsize)
        hm = np.maximum(hm, 1e-10)
        hm = np.log(hm)
        for n in range(coords.shape[0]):
            for p in range(coords.shape[1]):
                coords[n, p] = self.dark_parse(hm[n][p], coords[n][p])
        return coords

    def get_final_preds(self, heatmaps, center, scale, kernelsize=3):
        """the highest heatvalue location with a quarter offset in the
        direction from the highest response to the second highest response.
        Args:
            heatmaps (numpy.ndarray): The predicted heatmaps
            center (numpy.ndarray): The boxes center
            scale (numpy.ndarray): The scale factor
        Returns:
            preds: numpy.ndarray([batch_size, num_joints, 2]), keypoints coords
            maxvals: numpy.ndarray([batch_size, num_joints, 1]), the maximum confidence of the keypoints
        """
        coords, maxvals = self.get_max_preds(heatmaps)

        heatmap_height = heatmaps.shape[2]
        heatmap_width = heatmaps.shape[3]

        if self.use_dark:
            coords = self.dark_postprocess(heatmaps, coords, kernelsize)
        else:
            for n in range(coords.shape[0]):
                for p in range(coords.shape[1]):
                    hm = heatmaps[n][p]
                    px = int(math.floor(coords[n][p][0] + 0.5))
                    py = int(math.floor(coords[n][p][1] + 0.5))
                    if 1 < px < heatmap_width - 1 and 1 < py < heatmap_height - 1:
                        diff = np.array([
                            hm[py][px + 1] - hm[py][px - 1],
                            hm[py + 1][px] - hm[py - 1][px]
                        ])
                        coords[n][p] += np.sign(diff) * .25
        preds = coords.copy()

        # Transform back
        for i in range(coords.shape[0]):
            preds[i] = transform_preds(coords[i], center[i], scale[i],
                                       [heatmap_width, heatmap_height])

        return preds, maxvals

    def __call__(self, output, center, scale):
        preds, maxvals = self.get_final_preds(np.array(output), center, scale)
        outputs = [[
            np.concatenate(
                (preds, maxvals), axis=-1), np.mean(
                    maxvals, axis=1)
        ]]
        return outputs
    