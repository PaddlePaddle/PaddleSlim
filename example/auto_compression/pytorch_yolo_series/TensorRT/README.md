# TensorRT Python预测

### 验证COCO mAP

-FP16
```shell
python trt_eval.py --onnx_model_file=yolov7_tiny_quant_onnx/yolov7-tiny.onnx \
                   --precision_mode=fp16 \
                   --dataset_dir=dataset/coco/ \
                   --val_image_dir=val2017 \
                   --val_anno_path=annotations/instances_val2017.json
```

- INT8
```shell
python trt_eval.py --onnx_model_file=yolov7_tiny_quant_onnx/yolov7_tiny_quant.onnx \
                   --calibration_file=yolov7_tiny_quant_onnx/calibration.cache \
                   --precision_mode=int8 \
                   --dataset_dir=dataset/coco/ \
                   --val_image_dir=val2017 \
                   --val_anno_path=annotations/instances_val2017.json
```

### 验证单张图片

- FP16
```shell
python trt_eval.py --onnx_model_file=yolov7-tiny.onnx --image_file=../images/000000570688.jpg --precision_mode=fp16
```

- INT8
```shell
python trt_eval.py --onnx_model_file=yolov7_tiny_quant_onnx/yolov7_tiny_quant.onnx \
                   --calibration_file=yolov7_tiny_quant_onnx/calibration.cache \
                   --image_file=../images/000000570688.jpg \
                   --precision_mode=int8
```

### FAQ

- 测试内存和显存占用时，首次运行会将ONNX模型转换成TRT模型，耗时不准确，再此运行trt.eval.py可获取真实的内存和显存占用。
