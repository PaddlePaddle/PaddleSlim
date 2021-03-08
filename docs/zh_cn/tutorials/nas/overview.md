# Overview

PaddleSlim提供了4种网络结构搜索的方法：基于模拟退火进行网络结构搜索、基于强化学习进行网络结构搜索、基于梯度进行网络结构搜索和Once-For-All。

| 算法名称  |   算法简介   | 代表模型 |
|:---------:|:------------:|:--------:|
| [Once-For-All](https://paddleslim.readthedocs.io/zh_CN/latest/tutorials/nas/dygraph/nas_ofa.html)    | OFA是一种基于One-Shot NAS的压缩方案。这种方式比较高效，其优势是只需要训练一个超网络就可以从中选择满足不同延时要求的子模型。 | Once-For-All   |
| [SANAS](https://paddleslim.readthedocs.io/zh_CN/latest/quick_start/static/nas_tutorial.html)            | SANAS是基于模拟退火的方式进行网络结构搜索，在机器资源不多的情况下，选择这种方式一般能得到比强化学习更好的模型。             | \              |
| [RLNAS](https://paddleslim.readthedocs.io/zh_CN/latest/api_cn/static/nas/nas_api.html#rlnas)            | RLNAS是基于强化学习的方式进行网络结构搜索，这种方式需要耗费大量机器资源。 | ENAS、NasNet、MNasNet  |
| [DARTS](https://paddleslim.readthedocs.io/zh_CN/latest/api_cn/darts.html)            | DARTS是基于梯度的方式进行网络结构搜索，可以大大缩短搜索时长。             | DARTS、PCDARTS              |

## 参考文献
[1] H. Cai, C. Gan, T. Wang, Z. Zhang, and S. Han. Once for all: Train one network and specialize it for efficient deployment. In International Conference on Learning Representations, 2020.  
[2] Pham, H.; Guan, M. Y.; Zoph, B.; Le, Q. V.; and Dean, J. 2018. Efficient neural architecture search via parameter sharing. arXiv preprint arXiv:1802.03268.  
[3] Zoph B, Vasudevan V, Shlens J, et al. Learning transferable architectures for scalable image recognition[J]. arXiv preprint arXiv:1707.07012, 2017, 2(6).  
[4] Mingxing Tan, Bo Chen, Ruoming Pang, Vijay Vasudevan, and Quoc V Le. Mnasnet: Platform-aware neural architecture search for mobile. arXiv preprint arXiv:1807.11626, 2018.  
[5] H Liu, K Simonyan, Y Yang. Darts: Differentiable architecture search. arXiv preprint arXiv:1806.09055, 2018.  
[6] Xu, Y., Xie, L., Zhang, X., Chen, X., Qi, G.J., Tian, Q., Xiong, H.: PCDARTS: Partial Channel Connections for Memory-efficient Differentiable Architecture Search. In: International Conference on Learning Representations (2020)  
[7] Han Cai, Ligeng Zhu, and Song Han. ProxylessNAS: Direct neural architecture search on target task and hardware. In ICLR, 2019. URL https://arxiv.org/pdf/1812.00332.pdf. 3, 5, 6, 7, 8  
