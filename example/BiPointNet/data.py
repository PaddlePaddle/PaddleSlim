# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import os
import pickle
import warnings
import numpy as np
import paddle
from paddle.io import Dataset
from tqdm import tqdm

warnings.filterwarnings("ignore")


def normalize_point_cloud(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc


def random_point_dropout(pc, max_dropout_ratio=0.875):
    dropout_ratio = np.random.random() * max_dropout_ratio  # 0~0.875
    drop_idx = np.where(np.random.random((pc.shape[0])) <= dropout_ratio)[0]
    if len(drop_idx) > 0:
        pc[drop_idx, :] = pc[0, :]  # set to the first point
    return pc


def random_scale_point_cloud(data, scale_low=0.8, scale_high=1.25):
    scales = np.random.uniform(scale_low, scale_high)
    data *= scales
    return data


def shift_point_cloud(data, shift_range=0.1):
    shifts = np.random.uniform(-shift_range, shift_range, (3))
    data += shifts
    return data


def jitter_point_cloud(data: np.ndarray, sigma: float=0.02, clip: float=0.05):
    assert clip > 0
    jittered_data = np.clip(sigma * np.random.randn(*data.shape), -clip, clip)
    data = data + jittered_data
    return data


def random_rotate_point_cloud(data: np.ndarray):
    angle = np.random.uniform() * 2 * np.pi
    cosval = np.cos(angle)
    sinval = np.sin(angle)
    rotation_matrix = np.array(
        [[cosval, sinval, 0], [-sinval, cosval, 0], [0, 0, 1]],
        dtype=data.dtype)
    data = data @ rotation_matrix
    return data


def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:, :3]
    centroids = np.zeros((npoint, ))
    distance = np.ones((N, )) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid)**2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point


class ModelNetDataset(Dataset):
    def __init__(
            self,
            root,
            num_point,
            use_uniform_sample=False,
            use_normals=False,
            num_category=40,
            split="train",
            process_data=False, ):
        self.root = root
        self.npoints = num_point
        self.split = split
        self.process_data = process_data
        self.uniform = use_uniform_sample
        self.use_normals = use_normals
        self.num_category = num_category

        if self.num_category == 10:
            self.catfile = os.path.join(self.root, "modelnet10_shape_names.txt")
        else:
            self.catfile = os.path.join(self.root, "modelnet40_shape_names.txt")

        self.cat = [line.rstrip() for line in open(self.catfile)]
        self.classes = dict(zip(self.cat, range(len(self.cat))))

        shape_ids = {}
        if self.num_category == 10:
            shape_ids["train"] = [
                line.rstrip() for line in
                open(os.path.join(self.root, "modelnet10_train.txt"))
            ]
            shape_ids["test"] = [
                line.rstrip() for line in
                open(os.path.join(self.root, "modelnet10_test.txt"))
            ]
        else:
            shape_ids["train"] = [
                line.rstrip() for line in
                open(os.path.join(self.root, "modelnet40_train.txt"))
            ]
            shape_ids["test"] = [
                line.rstrip() for line in
                open(os.path.join(self.root, "modelnet40_test.txt"))
            ]

        assert split == "train" or split == "test"
        shape_names = ["_".join(x.split("_")[0:-1]) for x in shape_ids[split]]
        self.datapath = [(shape_names[i], os.path.join(
            self.root, shape_names[i], shape_ids[split][i]) + ".txt", )
                         for i in range(len(shape_ids[split]))]
        print("The size of %s data is %d" % (split, len(self.datapath)))

        if self.uniform:
            self.save_path = os.path.join(
                root,
                "modelnet%d_%s_%dpts_fps.dat" % (self.num_category, split,
                                                 self.npoints), )
        else:
            self.save_path = os.path.join(
                root,
                "modelnet%d_%s_%dpts.dat" % (self.num_category, split,
                                             self.npoints), )

        if self.process_data:
            if not os.path.exists(self.save_path):
                print("Processing data %s (only running in the first time)..." %
                      self.save_path)
                self.list_of_points = [None] * len(self.datapath)
                self.list_of_labels = [None] * len(self.datapath)

                for index in tqdm(
                        range(len(self.datapath)), total=len(self.datapath)):
                    fn = self.datapath[index]
                    cls = self.classes[self.datapath[index][0]]
                    cls = np.array([cls]).astype(np.int32)
                    point_set = np.loadtxt(
                        fn[1], delimiter=",").astype(np.float32)

                    if self.uniform:
                        point_set = farthest_point_sample(
                            point_set, self.npoints)
                    else:
                        point_set = point_set[0:self.npoints, :]

                    self.list_of_points[index] = point_set
                    self.list_of_labels[index] = cls

                with open(self.save_path, "wb") as f:
                    pickle.dump([self.list_of_points, self.list_of_labels], f)
            else:
                print("Load processed data from %s..." % self.save_path)
                with open(self.save_path, "rb") as f:
                    self.list_of_points, self.list_of_labels = pickle.load(f)

    def __len__(self):
        return len(self.datapath)

    def _get_item(self, index):
        if self.process_data:
            point_set, label = self.list_of_points[index], self.list_of_labels[
                index]
        else:
            fn = self.datapath[index]
            cls = self.classes[self.datapath[index][0]]
            label = np.array([cls]).astype(np.int32)
            point_set = np.loadtxt(fn[1], delimiter=",").astype(np.float32)

            if self.uniform:
                point_set = farthest_point_sample(point_set, self.npoints)
            else:
                point_set = point_set[0:self.npoints, :]

        point_set[:, 0:3] = normalize_point_cloud(point_set[:, 0:3])
        if not self.use_normals:
            point_set = point_set[:, 0:3]

        if self.split == "train":
            # point_set[:, 0:3] = random_rotate_point_cloud(point_set[:, 0:3])
            point_set[:, 0:3] = jitter_point_cloud(point_set[:, 0:3])
            # point_set[:, 0:3] = random_point_dropout(point_set[:, 0:3])
            point_set[:, 0:3] = random_scale_point_cloud(point_set[:, 0:3])
            point_set[:, 0:3] = shift_point_cloud(point_set[:, 0:3])

        return paddle.to_tensor(point_set, dtype=paddle.float32), int(label[0])

    def __getitem__(self, index):
        return self._get_item(index)
