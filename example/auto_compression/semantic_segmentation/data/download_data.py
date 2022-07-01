# Copyright (c) 20212  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
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

import sys
import os
from paddleslim.utils.download import download_file_and_uncompress

LOCAL_PATH = os.path.dirname(os.path.abspath(__file__))

urls = {
    "mini_humanseg":
    "https://paddleseg.bj.bcebos.com/humanseg/data/mini_supervisely.zip",
    "mini_cityscapes":
    "https://bj.bcebos.com/v1/paddle-slim-models/data/mini_cityscapes/mini_cityscapes.tar"
}

names = {"mini_humanseg": "humanseg", "mini_cityscapes": "cityscapes"}


def download_data(savepath, extrapath, url):
    download_file_and_uncompress(
        url=url, savepath=savepath, extrapath=extrapath)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(
            f"Usage: python {__file__} dataset_name\ndataset name should be in {list(urls.keys())}"
        )
        sys.exit(1)
    dataset_name = sys.argv[1]
    assert dataset_name in urls, f"dataset name should be in {list(urls.keys())}"
    download_file_and_uncompress(
        url=urls[dataset_name],
        savepath=LOCAL_PATH,
        extrapath=LOCAL_PATH,
        extraname=names[dataset_name])
    print("Data download finish!")
