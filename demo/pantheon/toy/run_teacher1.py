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

import numpy as np
import paddle.fluid as fluid

from utils import parse_args, sample_generator, sample_list_generator, batch_generator
from paddleslim.pantheon import Teacher


def run(args):
    if args.out_path and args.out_port:
        raise ValueError("args.out_path and args.out_port should not be valid "
                         "at the same time")
    if not args.out_path and not args.out_port:
        raise ValueError("One of args.out_path and args.out_port be valid")

    # user-defined program: y = 2*x - 1 
    startup = fluid.Program()
    program = fluid.Program()
    with fluid.program_guard(program, startup):
        inp_x = fluid.layers.data(name='x', shape=[-1, 1], dtype="int64")
        y = inp_x * 2 - 1
        result = fluid.layers.assign(y)

    place = fluid.CUDAPlace(0) if args.use_cuda else fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(startup)

    teacher = Teacher(out_path=args.out_path, out_port=args.out_port)
    teacher.start()

    if args.generator_type == "sample_generator":
        reader_config = {
            "sample_generator": sample_generator(max_n=1000),
            "batch_size": args.batch_size,
            "drop_last": False
        }
    elif args.generator_type == "sample_list_generator":
        reader_config = {
            "sample_list_generator": sample_list_generator(
                max_n=1000, batch_size=args.batch_size)
        }
    else:
        reader_config = {
            "batch_generator": batch_generator(
                max_n=1000, batch_size=args.batch_size)
        }

    if args.test_send_recv:
        teacher.send("greetings from teacher1")
        teacher.send({"x": 1, "y": 2})
        teacher.send({3, 5})
        print("recved {}".format(teacher.recv()))

    teacher.start_knowledge_service(
        feed_list=[inp_x.name],
        schema={"x": inp_x,
                "2x-1": y,
                "result": result},
        program=program,
        reader_config=reader_config,
        exe=exe,
        use_fp16=True,
        times=args.serving_times)


if __name__ == '__main__':
    args = parse_args()
    run(args)
