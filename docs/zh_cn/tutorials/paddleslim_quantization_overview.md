# PaddleSlim模型量化方法总览

PaddleSlim主要包含三种量化方法：在线量化、动态离线量化、静态离线量化

除此之外，PaddleSlim还有一种对embedding层量化的方法，将网络中embedding层参数从float32类型量化到int8类型。

下图展示了如何根据需要选择模型量化方法

![模型量化算法选择](https://user-images.githubusercontent.com/52520497/95644539-e7f23500-0ae9-11eb-80a8-596cfb285e17.png)

下表综合对比了模型量化方法的使用条件、易用性、精度损失和预期收益。

![模型量化算法对比](https://user-images.githubusercontent.com/52520497/95644609-59ca7e80-0aea-11eb-8897-208d7ccd5af1.png)

- [在线量化](quant_aware_training_tutorial.md)
- [动态离线量化](quant_post_dynamic_tutorial.md)
- [静态离线量化](quant_post_static_tutorial.md)
- [Embedding量化](embedding_quant_tutorial.md)
