## 量化FAQ

1. 量化训练或者离线量化后的模型体积为什么没有变小？
2. 量化训练或者离线量化后的模型使用fluid加载为什么没有加速？怎样才能加速？
3. 该怎么设置适合的量化配置？
4. 离线量化出现'KeyError: '报错
5. 离线量化或者量化训练时出现CUDNN或者CUDA错误
6. 量化训练时loss是nan
7. cpu上跑量化后的模型出nan

#### 1. 量化训练或者离线量化后的模型体积为什么没有变小？

答：这是因为量化后保存的参数是虽然是int8范围，但是类型是float。这是由于fluid没有int8 kernel, 为了方便量化后验证量化精度，必须能让fluid能够加载。

#### 2. 量化训练或者离线量化后的模型使用fluid加载为什么没有加速？怎样才能加速？

答：这是因为量化后保存的参数是虽然是int8范围，但是类型是float。fluid并不具备加速量化模型的能力。量化模型必须配合使用预测库才能加速。

- 如果量化模型在ARM上线，则需要使用[Paddle-Lite](https://paddle-lite.readthedocs.io/zh/latest/index.html).

    -  Paddle-Lite会对量化模型进行模型转化和优化，转化方法见[链接](https://paddle-lite.readthedocs.io/zh/latest/user_guides/model_quantization.html#paddle-lite)。

    - 转化之后可以像非量化模型一样使用[Paddle-Lite API](https://paddle-lite.readthedocs.io/zh/latest/user_guides/tutorial.html#lite)进行加载预测。

- 如果量化模型在GPU上线，则需要使用[Paddle-TensorRT 预测接口](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/advanced_guide/performance_improving/inference_improving/paddle_tensorrt_infer.html).

    - 和非量化模型的区别在于以下参数设置：

```python
config->EnableTensorRtEngine(1 << 20      /* workspace_size*/,  
                        batch_size        /* max_batch_size*/,  
                        3                 /* min_subgraph_size*/,
                        AnalysisConfig::Precision::kInt8 /* precision*/,
                        false             /* use_static*/,
                        false             /* use_calib_mode*/);
```

-  如果量化模型在x86上线，需要使用[INT8 MKL-DNN](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/fluid/contrib/slim/tests/slim_int8_mkldnn_post_training_quantization.md)

    - 首先对模型进行转化，可以参考[脚本](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/fluid/contrib/slim/tests/save_quant_model.py)

    - 转化之后可使用预测部署API进行加载。比如[c++ API](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/advanced_guide/inference_deployment/inference/native_infer.html)


#### 3. 该怎么设置适合的量化配置？

- 首先需要考虑量化模型上线的平台

   | 平台             | 支持weight量化方式             | 支持activation量化方式                | 支持量化的OP                                                 |
   | ---------------- | ------------------------------ | ------------------------------------- | ------------------------------------------------------------ |
   | ARM(Paddle-Lite) | channel_wise_abs_max， abs_max | moving_average_abs_max，range_abs_max | conv2d, depthwise_conv2d, mul                                |
   | x86(MKL-DNN)     | abs_max                        | moving_average_abs_max，range_abs_max | conv2d, depthwise_conv2d, mul, matmul                        |
   | GPU(TensorRT)    | channel_wise_abs_max           | moving_average_abs_max，range_abs_max | mul, conv2d, pool2d, depthwise_conv2d, elementwise_add, leaky_relu |

- 部分层跳过量化

   如果量化后精度损失较大，可以考虑跳过部分对量化敏感的计算不量化，比如最后一层或者attention计算。



#### 4. 离线量化出现'KeyError: '报错



答： 一般是reader没写对，导致离线量化是前向一次没跑，没有收集到中间的激活值。



#### 5. 离线量化或者量化训练时出现CUDNN或者CUDA错误



答：因为离线量化或者量化训练并没有涉及到对cuda或者cudnn做修改， 因此这个错误一般是机器上的cuda或者cudnn版本和Paddle所需的cuda或者cudnn版本不一致。



#### 6. 量化训练时loss是nan



答：需要适当调小学习率。如果小学习率依然不能解决问题，则需要考虑是否某些层对量化敏感，需要跳过量化，比如attention.



#### 7. cpu上跑量化后的模型出nan



答：可查看使用的paddle版本是否包含[pr](https://github.com/PaddlePaddle/Paddle/pull/22966)， 该pr修复了在对几乎是0的tensor进行量化时的bug。
