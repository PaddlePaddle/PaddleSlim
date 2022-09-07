# Auto Compression of YOLO-Series Model

Contents：

- [1.Introduction](#1Introduction)
- [2.Benchmark](#2Benchmark)
- [3.Start the auto compression](#Start-the-auto-compression)
  - [3.1 Environment Preparation](#31Environment-Preparation)
  - [3.2 Dataset Preparation](#32-Dataset-Preparation)
  - [3.3 Inference Model Preparation](#33-Inference-Model-Preparation)
  - [3.4 Automatically compress and export model](#34-Automatically-compress-and-export-model)
  - [3.5 Testing Model Accuracy](#35-Testing-Model-Accuracy)
- [4.Inference Deployment](#4Inference-Deployment)
- [5.FAQ](5FAQ)

## 1. Introduction

This demo takes [ultralytics/yolov5](https://github.com/ultralytics/yolov5), [meituan/YOLOv6](https://github.com/meituan/YOLOv6) and [WongKinYiu/ yolov7](https://github.com/WongKinYiu/yolov7) object detection models as examples. With the help of [X2Paddle](https://github.com/PaddlePaddle/X2Paddle) to convert PyTorch framework models into PaddlePaddle framework, developers can use ACT to compress the model automatically. The compressed model can use Paddle Inference or export to ONNX, and TensorRT for deployment.

## 2.Benchmark

| Model       | Strategy                           | Input Size | mAP<sup>val<br>0.5:0.95 | Model Size | Predicted Latency<sup><small>FP32</small><sup><br><sup> | Predicted Latency<sup><small>FP16</small><sup><br><sup> | Predicted Latency<sup><small>INT8</small><sup><br><sup> | Memory Usage | Video Memory Usage | Config                                                                                                                   | Inference Model                                                                                                                                                                      |
|:----------- |:---------------------------------- |:----------:|:-----------------------:|:----------:|:-------------------------------------------------------:|:-------------------------------------------------------:|:-------------------------------------------------------:|:------------:|:------------------:|:------------------------------------------------------------------------------------------------------------------------:|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
| YOLOv5s     | Base Model                         | 640*640    | 37.4                    | 28.1MB     | 5.95ms                                                  | 2.44ms                                                  | -                                                       | 1718MB       | 705MB              | -                                                                                                                        | [Model](https://paddle-slim-models.bj.bcebos.com/act/yolov5s.onnx)                                                                                                                   |
| YOLOv5s     | Post-Training Model                | 640*640    | 36.0                    | 7.4MB      | -                                                       | -                                                       | 1.87ms                                                  | 736MB        | 315MB              | [config](https://github.com/PaddlePaddle/PaddleSlim/tree/develop/example/post_training_quantization/pytorch_yolo_series) | -                                                                                                                                                                                    |
| YOLOv5s     | ACT Quantization aware training    | 640*640    | **36.9**                | 7.4MB      | -                                                       | -                                                       | **1.87ms**                                              | 736MB        | 315MB              | [config](./configs/yolov5s_qat_dis.yaml)                                                                                 | [Infer Model](https://bj.bcebos.com/v1/paddle-slim-models/act/yolov5s_quant.tar) &#124; [ONNX Model](https://bj.bcebos.com/v1/paddle-slim-models/act/yolov5s_quant_onnx.tar)         |
|             |                                    |            |                         |            |                                                         |                                                         |                                                         |              |                    |                                                                                                                          |                                                                                                                                                                                      |
| YOLOv6s     | Base Model                         | 640*640    | 42.4                    | 65.9MB     | 9.06ms                                                  | 2.90ms                                                  | -                                                       | 1208MB       | 555MB              | -                                                                                                                        | [Model](https://paddle-slim-models.bj.bcebos.com/act/yolov6s.onnx)                                                                                                                   |
| YOLOv6s     | KL Post-Training Model             | 640*640    | 30.3                    | 16.8MB     | -                                                       | -                                                       | 1.83ms                                                  | 736MB        | 315MB              | [config](https://github.com/PaddlePaddle/PaddleSlim/tree/develop/example/post_training_quantization/pytorch_yolo_series) | -                                                                                                                                                                                    |
| YOLOv6s     | Quantitative distillation training | 640*640    | **41.3**                | 16.8MB     | -                                                       | -                                                       | **1.83ms**                                              | 736MB        | 315MB              | [config](./configs/yolov6s_qat_dis.yaml)                                                                                 | [Infer Model](https://bj.bcebos.com/v1/paddle-slim-models/act/yolov6s_quant.tar) &#124; [ONNX Model](https://bj.bcebos.com/v1/paddle-slim-models/act/yolov6s_quant_onnx.tar)         |
|             |                                    |            |                         |            |                                                         |                                                         |                                                         |              |                    |                                                                                                                          |                                                                                                                                                                                      |
| YOLOv7      | Base Model                         | 640*640    | 51.1                    | 141MB      | 26.84ms                                                 | 7.44ms                                                  | -                                                       | 1722MB       | 917MB              | -                                                                                                                        | [Model](https://paddle-slim-models.bj.bcebos.com/act/yolov7.onnx)                                                                                                                    |
| YOLOv7      | Post-Training Model                | 640*640    | 50.2                    | 36MB       | -                                                       | -                                                       | 4.55ms                                                  | 827MB        | 363MB              | [config](https://github.com/PaddlePaddle/PaddleSlim/tree/develop/example/post_training_quantization/pytorch_yolo_series) | -                                                                                                                                                                                    |
| YOLOv7      | ACT Quantization aware training    | 640*640    | **50.9**                | 36MB       | -                                                       | -                                                       | **4.55ms**                                              | 827MB        | 363MB              | [config](./configs/yolov7_qat_dis.yaml)                                                                                  | [Infer Model](https://bj.bcebos.com/v1/paddle-slim-models/act/yolov7_quant.tar) &#124; [ONNX Model](https://bj.bcebos.com/v1/paddle-slim-models/act/yolov7_quant_onnx.tar)           |
|             |                                    |            |                         |            |                                                         |                                                         |                                                         |              |                    |                                                                                                                          |                                                                                                                                                                                      |
| YOLOv7-Tiny | Base Model                         | 640*640    | 37.3                    | 24MB       | 5.06ms                                                  | 2.32ms                                                  | -                                                       | 738MB        | 349MB              | -                                                                                                                        | [Model](https://paddle-slim-models.bj.bcebos.com/act/yolov7-tiny.onnx)                                                                                                               |
| YOLOv7-Tiny | Post-Training Model                | 640*640    | 35.8                    | 6.1MB      | -                                                       | -                                                       | 1.68ms                                                  | 729MB        | 315MB              | -                                                                                                                        | -                                                                                                                                                                                    |
| YOLOv7-Tiny | ACT Quantization aware training    | 640*640    | **37.0**                | 6.1MB      | -                                                       | -                                                       | **1.68ms**                                              | 729MB        | 315MB              | [config](./configs/yolov7_tiny_qat_dis.yaml)                                                                             | [Infer Model](https://bj.bcebos.com/v1/paddle-slim-models/act/yolov7_tiny_quant.tar) &#124; [ONNX Model](https://bj.bcebos.com/v1/paddle-slim-models/act/yolov7_tiny_quant_onnx.tar) |

Note:

- The mAP indicators were all obtained by evaluation in the COCO val2017 dataset.
- The YOLOv7 model was tested in a Tesla T4 GPU environment with TensorRT 8.4.1 enabled, batch_size=1, and the test script was [cpp_infer](. /cpp_infer).

## 3. Start the auto compression

#### 3.1 Environment Preparation

- PaddlePaddle >= 2.3.2 version, Available from [PaddlePaddle Official website](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/install/pip/linux-pip.html) for installation instructions

- PaddleSlim develop version

（1）Install PaddlePaddle

```
# CPU
pip install paddlepaddle==2.3.2
# GPU Take Ubuntu、CUDA 11.2 as an example
python -m pip install paddlepaddle-gpu==2.3.2.post112 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html
```

（2）Install Paddleslim>=2.3.3：

```shell
pip install paddleslim==2.3.3
```

#### 3.2 Dataset Preparation

**Choose one of the methods to prepare the data**

- （1）Support direct transfer of unlabeled images to image folders, but does not support evaluation of model mAP
  
  Modify the `image_path` in [config](. /configs) to the `image_path` of the images folder in application. The number of images depends on the size of the dataset, with efforts to cover all deployment scenarios.
  
  ```yaml
  Global:
    image_path: dataset/coco/val2017
  ```

- （2）Supports loading of datasets in COCO format, **support real-time evaluation of model mAP** 
  
  Downlond [Train](http://images.cocodataset.org/zips/train2017.zip)、[Val](http://images.cocodataset.org/zips/val2017.zip)、[annotation](http://images.cocodataset.org/annotations/annotations_trainval2017.zip) from [MS COCO Official Website](https://cocodataset.org)
  
  The content format is as follows.
  
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
  
  If it is a customized dataset, please prepare the data as above.
  
  After preparing the dataset, modify the path to `coco_dataset_dir` in [config](. /configs).
  
  ```yaml
  Global:
    coco_dataset_dir: dataset/coco/
    coco_train_image_dir: train2017
    coco_train_anno_path: annotations/instances_train2017.json
    coco_val_image_dir: val2017
    coco_val_anno_path: annotations/instances_val2017.json
  ```

#### 3.3 Inference Model Preparation

（1）Prepare ONNX model：

- YOLOv5:
  
  This demo model is exported with the master branch of [ultralytics/yolov5](https://github.com/ultralytics/yolov5). It requires ONNX models from v6.1 onwards, which can be prepared according to the official [export tutorial](https://github.com/ ultralytics/yolov5/issues/251). A prepared model is also available on [yolov5s.onnx](https://paddle-slim-models.bj.bcebos.com/act/yolov5s.onnx).
  
  ```shell
  python export.py --weights yolov5s.pt --include onnx
  ```

- YOLOv6:
  
  Developers can prepare models according to the official tutorial on. [meituan/YOLOv6](https://github.com/meituan/YOLOv6). Developers can also download prepared [yolov6s.onnx](https://paddle-slim-models.bj.bcebos.com/act/yolov6s.onnx).

- YOLOv7: 
  
  The ONNX model can be prepared with the export script from [WongKinYiu/yolov7](https://github.com/WongKinYiu/yolov7), as follows.
  
  ```shell
  git clone https://github.com/WongKinYiu/yolov7.git
  python export.py --weights yolov7-tiny.pt --grid
  ```
  
  **NOTE**: 
  
  ACT currently supports models **without NMS**, which can be exported by command as above. It is also possible to directly download the prepared [yolov7.onnx](https://paddle-slim-models.bj.bcebos.com/act/yolov7-tiny.onnx).

#### 3.4 Automatically compress and export model

The distillation quantization demo is started via the run.py script and will automatically compress the model by using the interface ``paddleslim.auto_compression.AutoCompression``. Developers can configure the parameters for the model path, distillation, quantization, and training sections of the config file, and the model will be ready for quantization and distillation once configured.

This demo takes auto-compression with YOLOv7-Tiny as an example. If users eloper wants to change the model,  they can modify the `--config_path` by running the command as follows

- Single-Card training
  
  ```
  export CUDA_VISIBLE_DEVICES=0
  python run.py --config_path=./configs/yolov7_tiny_qat_dis.yaml --save_dir='./output/'
  ```

- Multi-Card training
  
  ```
  CUDA_VISIBLE_DEVICES=0,1,2,3 python -m paddle.distributed.launch --log_dir=log --gpus 0,1,2,3 run.py \
          --config_path=./configs/yolov7_tiny_qat_dis.yaml --save_dir='./output/'
  ```

#### 3.5 Testing Model Accuracy

修改[yolov7_qat_dis.yaml](./configs/yolov7_qat_dis.yaml)中`model_dir`字段为模型存储路径，然后使用eval.py脚本得到模型的mAP：

Modify the `model_dir` in [yolov7_qat_dis.yaml](. /configs/yolov7_qat_dis.yaml) to be the model storage path, then use the eval.py script to get the model's mAP.

```
export CUDA_VISIBLE_DEVICES=0
python eval.py --config_path=./configs/yolov7_tiny_qat_dis.yaml
```

## 4.Inference Deployment

After the auto compression is completed, the following files will be generated:

```shell
├── model.pdiparams         # Paddle Inference model weights
├── model.pdmodel           # Paddle Inference model file
├── calibration_table.txt   # Paddle post-quantitative calibration table
├── ONNX
│   ├── quant_model.onnx      # ONNX model after quantization
│   ├── calibration.cache     # TensorRT Calibration table that can be loaded directly by TensorRT
```

#### 

#### Export to ONNX for deployment with TensorRT

Load `quant_model.onnx`和`calibration.cache` to use TensorRT test script directly. For detailed code, please refer to [TensorRT deployment](/TensorRT)

- python test
  
  ```shell
  cd TensorRT
  python trt_eval.py --onnx_model_file=output/ONNX/quant_model.onnx \
                   --calibration_file=output/ONNX/calibration.cache \
                   --image_file=../images/000000570688.jpg \
                   --precision_mode=int8
  ```

- speed test
  
  ```shell
  trtexec --onnx=output/ONNX/quant_model.onnx --avgRuns=1000 --workspace=1024 --calib=output/ONNX/calibration.cache --int8
  ```

#### Paddle-TensorRT deployment

- C++ deployment

Open [cpp_infer](./cpp_infer) file, and please prepare the environment and compile according to [C++ TensorRT Benchmark testing turtorial](./cpp_infer/README.md), and then start the test：

```shell
# Compile
bash compile.sh
# Execute
./build/trt_run --model_file yolov7_quant/model.pdmodel --params_file yolov7_quant/model.pdiparams --run_mode=trt_int8
```

- Python deployment:

Please install [PaddlePaddle with TensorRT](https://www.paddlepaddle.org.cn/inference/v2.3/user_guides/download_lib.html#python)。

And deploy with [paddle_trt_infer.py](./paddle_trt_infer.py)

```shell
python paddle_trt_infer.py --model_path=output --image_file=images/000000570688.jpg --benchmark=True --run_mode=trt_int8
```

## 5.FAQ

- 
- If you want to conduct post-training quantization, please refer to [YOLO series Post-training quantization demo](https://github.com/PaddlePaddle/PaddleSlim/tree/develop/example/post_training_quantization/pytorch_yolo_series)
