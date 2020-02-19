#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import argparse
from paddleslim.pantheon import Student

from utils import str2bool


def parse_args():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument(
        "--in_address0",
        type=str,
        default=None,
        help="Input address for teacher 0. (default: %(default)s)")
    parser.add_argument(
        "--in_path0",
        type=str,
        default=None,
        help="Input file path for teacher 0. (default: %(default)s)")
    parser.add_argument(
        "--in_address1",
        type=str,
        default=None,
        help="Input address for teacher 1. (default: %(default)s)")
    parser.add_argument(
        "--in_path1",
        type=str,
        default=None,
        help="Input file path for teacher 1. (default: %(default)s)")
    parser.add_argument(
        "--test_send_recv",
        type=str2bool,
        default=False,
        help="Whether to test send/recv interfaces. (default: %(default)s)")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="The batch size of student model. (default: %(default)s)")
    args = parser.parse_args()
    return args


def run(args):
    if args.in_address0 and args.in_path0:
        raise ValueError(
            "args.in_address0 and args.in_path0 should not be valid "
            "at the same time!")
    if not args.in_address0 and not args.in_path0:
        raise ValueError(
            "One of args.in_address0 and args.in_path0 must be valid!")

    if args.in_address1 and args.in_path1:
        raise ValueError(
            "args.in_address1 and args.in_path1 should not be valid "
            "at the same time!")
    if not args.in_address1 and not args.in_path1:
        raise ValueError(
            "One of args.in_address1 and args.in_path1 must be valid")

    student = Student(merge_strategy={"result": "sum"})

    student.register_teacher(
        in_address=args.in_address0, in_path=args.in_path0)
    student.register_teacher(
        in_address=args.in_address1, in_path=args.in_path1)
    student.start()

    if args.test_send_recv:
        for t in range(2):
            for i in range(3):
                print(student.recv(t))
        student.send("message from student!")

    knowledge_desc = student.get_knowledge_desc()
    data_generator = student.get_knowledge_generator(
        batch_size=args.batch_size, drop_last=False)
    for batch_data in data_generator():
        batch_size = list(batch_data.values())[0].shape[0]
        keys = batch_data.keys()
        for i in range(batch_size):
            data = {}
            for key in keys:
                data[key] = batch_data[key][i]
            print(data)


if __name__ == '__main__':
    args = parse_args()
    run(args)
