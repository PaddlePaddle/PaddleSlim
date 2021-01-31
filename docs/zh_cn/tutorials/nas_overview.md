# PaddleSlim模型结构搜索总览

PaddleSlim提供了4种网络结构搜索的方法：基于模拟退火进行网络结构搜索、基于强化学习进行网络结构搜索、基于梯度进行网络结构搜索和Once-For-All。

## [Once-For-All](nas_ofa.md)
  OFA是一种基于One-Shot NAS的压缩方案。这种方式比较高效，其优势是只需要训练一个超网络就可以从中选择满足不同延时要求的子模型。

## [SANAS](nas_sa.md)
  SANAS是基于模拟退火的方式进行网络结构搜索，在机器资源不多的情况下，选择这种方式一般能得到比强化学习更好的模型。

## [RLNAS](nas_rl.md)
  RLNAS是基于强化学习的方式进行网络结构搜索，这种方式需要耗费大量机器资源。利用强化学习进行搜索的模型有：ENAS、NasNet、MNasNet等。

## [DARTS/PCDARTS](nas_darts.md)
  DARTS是基于梯度进行网络结构搜索，这种方式比较高效，大大减少了搜索时间和所需要的机器资源。利用梯度进行搜索的模型有：DARTS、PCDARTS、ProxylessNAS等。
