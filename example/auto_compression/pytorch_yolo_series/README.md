# YOLO系列模型自动压缩示例

目录：
- [1.简介](#1简介)
- [2.Benchmark](#2Benchmark)
- [3.开始自动压缩](#自动压缩流程)
  - [3.1 环境准备](#31-准备环境)
  - [3.2 准备数据集](#32-准备数据集)
  - [3.3 准备预测模型](#33-准备预测模型)
  - [3.4 自动压缩并产出模型](#34-自动压缩并产出模型)
- [4.预测部署](#4预测部署)
- [5.FAQ](5FAQ)

## 1. 简介

本示例以[ultralytics/yolov5](https://github.com/ultralytics/yolov5)，[meituan/YOLOv6](https://github.com/meituan/YOLOv6) 和 [WongKinYiu/yolov7](https://github.com/WongKinYiu/yolov7) 目标检测模型为例，借助[X2Paddle](https://github.com/PaddlePaddle/X2Paddle)的能力，将PyTorch框架模型转换为Paddle框架模型，再使用ACT自动压缩功能进行模型压缩，压缩后的模型可使用Paddle Inference或者导出至ONNX，利用TensorRT部署。

## 2.Benchmark

| 模型            |  策略  | 输入尺寸 | mAP<sup>val<br>0.5:0.95 |  模型体积  | 预测时延<sup><small>FP32</small><sup><br><sup> |预测时延<sup><small>FP16</small><sup><br><sup> | 预测时延<sup><small>INT8</small><sup><br><sup> | 内存占用 | 显存占用  |                                                           配置文件                                                           |                                                                                      Inference模型                                                                                       |
|:--------------|:-------- |:--------: |:-----------------------:|:------:| :----------------: | :----------------: |:----------------: | :----------------: | :---------------: |:------------------------------------------------------------------------------------------------------------------------:|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
| YOLOv5s       |  Base模型 | 640*640  |          37.4           | 28.1MB | 5.95ms  |   2.44ms   |  -  | 1718MB | 705MB |                                                            -                                                             |                                                           [Model](https://paddle-slim-models.bj.bcebos.com/act/yolov5s.onnx)                                                           |
| YOLOv5s       |  离线量化 | 640*640  |          36.0           | 7.4MB  |   - |   -   |  1.87ms  | 736MB | 315MB  | [config](https://github.com/PaddlePaddle/PaddleSlim/tree/develop/example/post_training_quantization/pytorch_yolo_series) |                                                                                           -                                                                                            |
| YOLOv5s       |  ACT量化训练  | 640*640  |        **36.9**         | 7.4MB  |    - |   -   |  **1.87ms**  | 736MB | 315MB |                                         [config](./configs/yolov5s_qat_dis.yaml)                                         |      [Infer Model](https://bj.bcebos.com/v1/paddle-slim-models/act/yolov5s_quant.tar) &#124; [ONNX Model](https://bj.bcebos.com/v1/paddle-slim-models/act/yolov5s_quant_onnx.tar)      |
|               |  |  |                         |        |  |  |  |  |
| YOLOv6s       |  Base模型 | 640*640  |          42.4           | 65.9MB |   9.06ms  |   2.90ms   |  - | 1208MB   | 555MB  |                                                            -                                                             |                                                           [Model](https://paddle-slim-models.bj.bcebos.com/act/yolov6s.onnx)                                                           |
| YOLOv6s       |  KL离线量化 | 640*640  |          30.3           | 16.8MB |   - |   -   |  1.83ms  | 736MB   | 315MB | [config](https://github.com/PaddlePaddle/PaddleSlim/tree/develop/example/post_training_quantization/pytorch_yolo_series) |                                                                                           -                                                                                            |
| YOLOv6s       |  量化蒸馏训练 | 640*640  |        **41.3**         | 16.8MB |   - |   -   |  **1.83ms**  | 736MB   | 315MB |                                         [config](./configs/yolov6s_qat_dis.yaml)                                         |      [Infer Model](https://bj.bcebos.com/v1/paddle-slim-models/act/yolov6s_quant.tar) &#124; [ONNX Model](https://bj.bcebos.com/v1/paddle-slim-models/act/yolov6s_quant_onnx.tar)      |
|               |  |  |                         |        |  |  |  |  |
| YOLOv6s_v2    |  Base模型 | 640*640  |          43.4           | 67.4MB |   9.06ms  |   2.90ms   |  - | 1208MB   | 555MB  |                                                            -                                                             |                                                           [Model](https://github.com/meituan/YOLOv6/releases/download/0.2.0/yolov6s.onnx)                                                           |
| YOLOv6s_v2    |  量化蒸馏训练 | 640*640  |        **43.0**         | 16.8MB |   - |   -   |  **1.83ms**  | 736MB   | 315MB |                                       [config](./configs/yolov6s_v2_qat_dis.yaml)                                        | [Infer Model](https://bj.bcebos.com/v1/paddle-slim-models/act/yolov6s_v2_0_quant.tar) &#124; [ONNX Model](https://bj.bcebos.com/v1/paddle-slim-models/act/yolov6s_v2_0_quant_onnx.tar) |
|               |  |  |                         |        |  |  |  |  |
| YOLOv7        |  Base模型 | 640*640  |          51.1           | 141MB  |  26.84ms  |   7.44ms   |  -  | 1722MB  | 917MB |                                                            -                                                             |                                                           [Model](https://paddle-slim-models.bj.bcebos.com/act/yolov7.onnx)                                                            |
| YOLOv7        |  离线量化 | 640*640  |          50.2           |  36MB  |   - |   -   |  4.55ms  | 827MB  | 363MB  | [config](https://github.com/PaddlePaddle/PaddleSlim/tree/develop/example/post_training_quantization/pytorch_yolo_series) |                                                                                           -                                                                                            |
| YOLOv7        |  ACT量化训练 | 640*640  |        **50.9**         |  36MB  |   - |   -   |  **4.55ms**  | 827MB  | 363MB |                                         [config](./configs/yolov7_qat_dis.yaml)                                          |       [Infer Model](https://bj.bcebos.com/v1/paddle-slim-models/act/yolov7_quant.tar) &#124; [ONNX Model](https://bj.bcebos.com/v1/paddle-slim-models/act/yolov7_quant_onnx.tar)       |
|               |  |  |                         |        |  |  |  |  |
| YOLOv7-Tiny   |  Base模型 | 640*640  |          37.3           |  24MB  |  5.06ms  |   2.32ms   |  - | 738MB  | 349MB  |                                                            -                                                             |                                                         [Model](https://paddle-slim-models.bj.bcebos.com/act/yolov7-tiny.onnx)                                                         |
| YOLOv7-Tiny   |  离线量化 | 640*640  |          35.8           | 6.1MB  |   - |   -   |  1.68ms  | 729MB  | 315MB  |                                                            -                                                             |                                                                                           -                                                                                            |
| YOLOv7-Tiny   |  ACT量化训练 | 640*640  |        **37.0**         | 6.1MB  |  - |   -   |  **1.68ms**  | 729MB  | 315MB |                                       [config](./configs/yolov7_tiny_qat_dis.yaml)                                       |  [Infer Model](https://bj.bcebos.com/v1/paddle-slim-models/act/yolov7_tiny_quant.tar) &#124; [ONNX Model](https://bj.bcebos.com/v1/paddle-slim-models/act/yolov7_tiny_quant_onnx.tar)  |

说明：
- mAP的指标均在COCO val2017数据集中评测得到。
- YOLOv7模型在Tesla T4的GPU环境下开启TensorRT 8.4.1，batch_size=1， 测试脚本是[cpp_infer](./cpp_infer)。

## 3. 自动压缩流程

#### 3.1 准备环境
- PaddlePaddle >= 2.4.0版本 （可从[Paddle官网](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/install/pip/linux-pip.html)根据相应环境的安装指令进行安装）
- PaddleSlim develop 版本

（1）安装paddlepaddle
```
# CPU
pip install paddlepaddle
# GPU 以Ubuntu、CUDA 11.2为例
python -m pip install paddlepaddle_gpu -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html
```

（2）安装paddleslim>=2.4.0：
```shell
pip install paddleslim
```


#### 3.2 准备数据集

**选择(1)或(2)中一种方法准备数据即可。**

- （1）支持无标注图片，直接传入图片文件夹，但不支持评估模型mAP

  修改[config](./configs)中`image_path`路径为真实预测场景下的图片文件夹，图片数量依据数据集大小来定，尽量覆盖所有部署场景。
  ```yaml
  Global:
    image_path: dataset/coco/val2017
  ```

- （2）支持加载COCO格式数据集，**可支持实时评估模型mAP**

  可以从[MS COCO官网](https://cocodataset.org)下载[Train](http://images.cocodataset.org/zips/train2017.zip)、[Val](http://images.cocodataset.org/zips/val2017.zip)、[annotation](http://images.cocodataset.org/annotations/annotations_trainval2017.zip)。

  目录格式如下：
  ```
  dataset/coco/
  ├── annotations
  │   ├── instances_train2017.json
  │   ├── instances_val2017.json
  │   |   ...
  ├── train2017
  │   ├── 000000000009.jpg
  │   ├── 000000580008.jpg
  │   |   ...
  ├── val2017
  │   ├── 000000000139.jpg
  │   ├── 000000000285.jpg
  ```

  如果是自定义数据集，请按照如上COCO数据格式准备数据。

  准备好数据集后，修改[config](./configs)中`coco_dataset_dir`路径。
  ```yaml
  Global:
    coco_dataset_dir: dataset/coco/
    coco_train_image_dir: train2017
    coco_train_anno_path: annotations/instances_train2017.json
    coco_val_image_dir: val2017
    coco_val_anno_path: annotations/instances_val2017.json
  ```


#### 3.3 准备预测模型

（1）准备ONNX模型：

- YOLOv5:

  本示例模型使用[ultralytics/yolov5](https://github.com/ultralytics/yolov5)的master分支导出，要求v6.1之后的ONNX模型，可以根据官方的[导出教程](https://github.com/ultralytics/yolov5/issues/251)来准备ONNX模型。也可以下载准备好的[yolov5s.onnx](https://paddle-slim-models.bj.bcebos.com/act/yolov5s.onnx)。
  ```shell
  python export.py --weights yolov5s.pt --include onnx
  ```

- YOLOv6:

  可通过[meituan/YOLOv6](https://github.com/meituan/YOLOv6)官方的[导出教程](https://github.com/meituan/YOLOv6/blob/main/deploy/ONNX/README.md)来准备ONNX模型。也可以下载已经准备好的[yolov6s.onnx](https://paddle-slim-models.bj.bcebos.com/act/yolov6s.onnx)。

- YOLOv7: 可通过[WongKinYiu/yolov7](https://github.com/WongKinYiu/yolov7)的导出脚本来准备ONNX模型，具体步骤如下：
  ```shell
  git clone https://github.com/WongKinYiu/yolov7.git
  python export.py --weights yolov7-tiny.pt --grid
  ```

  **注意**：目前ACT支持**不带NMS**模型，使用如上命令导出即可。也可以直接下载我们已经准备好的[yolov7.onnx](https://paddle-slim-models.bj.bcebos.com/act/yolov7-tiny.onnx)。

#### 3.4 自动压缩并产出模型

蒸馏量化自动压缩示例通过run.py脚本启动，会使用接口```paddleslim.auto_compression.AutoCompression```对模型进行自动压缩。配置config文件中模型路径、蒸馏、量化、和训练等部分的参数，配置完成后便可对模型进行量化和蒸馏。

本示例启动自动压缩以YOLOv7-Tiny为例，如果想要更换模型，可修改`--config_path`路径即可，具体运行命令为：

- 单卡训练：
```
export CUDA_VISIBLE_DEVICES=0
python run.py --config_path=./configs/yolov7_tiny_qat_dis.yaml --save_dir='./output/'
```

- 多卡训练：
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m paddle.distributed.launch --log_dir=log --gpus 0,1,2,3 run.py \
          --config_path=./configs/yolov7_tiny_qat_dis.yaml --save_dir='./output/'
```


## 4.预测部署

执行完自动压缩后会生成:
```shell
├── model.pdiparams         # Paddle预测模型权重
├── model.pdmodel           # Paddle预测模型文件
├── ONNX
│   ├── quant_model.onnx      # 量化后转出的ONNX模型
│   ├── calibration.cache     # TensorRT可以直接加载的校准表
```

#### Paddle Inference部署测试

量化模型在GPU上可以使用TensorRT进行加速，在CPU上可以使用MKLDNN进行加速。

以下字段用于配置预测参数：

| 参数名 | 含义 |
|:------:|:------:|
| model_path | inference 模型文件所在目录，该目录下需要有文件 model.pdmodel 和 model.pdiparams 两个文件 |
| dataset_dir | eval时数据验证集路径， 默认`dataset/coco` |
| image_file | 如果只测试单张图片效果，直接根据image_file指定图片路径 |
| device | 使用GPU或者CPU预测，可选CPU/GPU   |
| use_trt | 是否使用 TesorRT 预测引擎   |
| use_mkldnn | 是否启用```MKL-DNN```加速库，注意```use_mkldnn```与```use_gpu```同时为```True```时，将忽略```enable_mkldnn```，而使用```GPU```预测  |
| cpu_threads | CPU预测时，使用CPU线程数量，默认10  |
| precision | 预测精度，包括`fp32/fp16/int8`  |

 TensorRT Python部署:

首先安装带有TensorRT的[Paddle安装包](https://www.paddlepaddle.org.cn/inference/v2.3/user_guides/download_lib.html#python)。

然后使用[paddle_inference_eval.py](./paddle_inference_eval.py)进行部署：

```shell
python paddle_inference_eval.py \
      --model_path=output \
      --reader_config=configs/yoloe_reader.yml \
      --use_trt=True \
      --precision=int8
```

- MKLDNN预测：

```shell
python paddle_inference_eval.py \
      --model_path=output \
      --reader_config=configs/yoloe_reader.yml \
      --device=CPU \
      --use_mkldnn=True \
      --cpu_threads=10 \
      --precision=int8
```

- 测试单张图片

```shell
python paddle_inference_eval.py --model_path=output --image_file=images/000000570688.jpg --use_trt=True --precision=int8
```

- C++部署

进入[cpp_infer](./cpp_infer)文件夹内，请按照[C++ TensorRT Benchmark测试教程](./cpp_infer/README.md)进行准备环境及编译，然后开始测试：
```shell
# 编译
bash compile.sh
# 执行
./build/trt_run --model_file yolov7_quant/model.pdmodel --params_file yolov7_quant/model.pdiparams --run_mode=trt_int8
```

#### 导出至ONNX使用TensorRT部署

加载`quant_model.onnx`和`calibration.cache`，可以直接使用TensorRT测试脚本进行验证，详细代码可参考[TensorRT部署](./TensorRT)

- python测试：
```shell
cd TensorRT
python trt_eval.py --onnx_model_file=output/ONNX/quant_model.onnx \
                   --calibration_file=output/ONNX/calibration.cache \
                   --image_file=../images/000000570688.jpg \
                   --precision_mode=int8
```

- 速度测试
```shell
trtexec --onnx=output/ONNX/quant_model.onnx --avgRuns=1000 --workspace=1024 --calib=output/ONNX/calibration.cache --int8
```

## 5.FAQ

- 如果想对模型进行离线量化，可进入[YOLO系列模型离线量化示例](https://github.com/PaddlePaddle/PaddleSlim/tree/develop/example/post_training_quantization/pytorch_yolo_series)中进行实验。
