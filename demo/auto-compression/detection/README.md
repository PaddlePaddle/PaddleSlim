# 目标检测模型自动压缩

预测模型保存接口：
动态图使用``paddle.jit.save``保存；
静态图使用``paddle.static.save_inference_model``保存。

本示例将介绍如何使用PaddleDetection中预测模型进行蒸馏量化训练。

## 模型量化蒸馏训练流程

### 1. 准备COCO格式数据

参考[COCO数据准备文档](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.4/docs/tutorials/PrepareDataSet.md#coco%E6%95%B0%E6%8D%AE)

### 2. 准备需要量化的环境

- PaddlePaddle >= 2.2
- PaddleDet >= 2.3

```shell
pip install paddledet
```

#### 3 准备待量化模型
- 下载代码
```
git clone https://github.com/PaddlePaddle/PaddleDetection.git
```
- 导出预测模型
```shell
python tools/export_model.py -c configs/yolov3/yolov3_mobilenet_v1_270e_coco.yml -o weights=https://paddledet.bj.bcebos.com/models/yolov3_mobilenet_v1_270e_coco.pdparams
```
或直接下载：
```shell
wget https://paddledet.bj.bcebos.com/models/slim/yolov3_mobilenet_v1_270e_coco.tar
tar -xf yolov3_mobilenet_v1_270e_coco.tar
```

#### 2.4 测试模型精度
拷贝``yolov3_mobilenet_v1_270e_coco``文件夹到``PaddleSlim/demo/auto-compression/``文件夹。
```
cd PaddleSlim/demo/auto-compression/
```
使用[run_main.py](run_main.py)脚本得到模型的mAP：
```
python3.7 run_main.py --config_path='./configs/yolov3_mbv1_qat_dis.yaml --eval=True
```

### 3. 进行多策略融合压缩

每一个小章节代表一种多策略融合压缩，不代表需要串行执行。

### 3.1 进行量化蒸馏压缩
蒸馏量化训练示例脚本为[run_main.py](run_main.py)，使用接口``paddleslim.auto_compression.AutoCompression``对模型进行量化训练。运行命令为：
```
python run_main.py --config_path='./configs/yolov3_mbv1_qat_dis.yaml --save_dir='./output/' --devices='gpu'
```
