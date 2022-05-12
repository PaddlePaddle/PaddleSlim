## BiBERT: Accurate Fully Binarized BERT

Created by [Haotong Qin](https://htqin.github.io/), [Yifu Ding](https://yifu-ding.github.io/), [Mingyuan Zhang](https://scholar.google.com/citations?user=2QLD4fAAAAAJ&hl=en), Qinghua Yan, [Aishan Liu](https://liuaishan.github.io/), Qingqing Dang, [Ziwei Liu](https://liuziwei7.github.io/), and [Xianglong Liu](https://xlliu-beihang.github.io/) from Beihang University, Nanyang Technological University, and Baidu Inc.

![loading-ag-172](./overview.png)

## Introduction

This project is the official implementation of our accepted ICLR 2022 paper *BiBERT: Accurate Fully Binarized BERT* [[PDF](https://openreview.net/forum?id=5xEgrl_5FAJ)]. The large pre-trained BERT has achieved remarkable performance on Natural Language Processing (NLP) tasks but is also computation and memory expensive. As one of the powerful compression approaches, binarization extremely reduces the computation and memory consumption by utilizing 1-bit parameters and bitwise operations. Unfortunately, the full binarization of BERT (i.e., 1-bit weight, embedding, and activation) usually suffer a significant performance drop, and there is rare study addressing this problem. In this paper, with the theoretical justification and empirical analysis, we identify that the severe performance drop can be mainly attributed to the information degradation and optimization direction mismatch respectively in the forward and backward propagation, and propose BiBERT, an accurate fully binarized BERT, to eliminate the performance bottlenecks. Specifically, BiBERT introduces an efficient Bi-Attention structure for maximizing representation information statistically and a Direction-Matching Distillation (DMD) scheme to optimize the full binarized BERT accurately. Extensive experiments show that BiBERT outperforms both the straightforward baseline and existing state-of-the-art quantized BERTs with ultra-low bit activations by convincing margins on the NLP benchmark. As the first fully binarized BERT, our method yields impressive $59.2\times$ and $31.2\times$ saving on FLOPs and model size, demonstrating the vast advantages and potential of the fully binarized BERT model in real-world resource-constrained scenarios.

## Execution

First run run_fp.sh to fine-tune the bert-base-uncased model to obtain the full-precision teacher model. And then run run_binary.sh to use the full-precision teacher model distillation to train the binarized student model.


## Acknowledgement

The original code is borrowed from [TinyBERT](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/model_compression/tinybert).

## Citation

If you find our work useful in your research, please consider citing:

```shell
@inproceedings{Qin:iclr22,
  author    = {Haotong Qin and Yifu Ding and Mingyuan Zhang and Qinghua Yan and 
  Aishan Liu and Qingqing Dang and Ziwei Liu and Xianglong Liu},
  title     = {BiBERT: Accurate Fully Binarized BERT},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2022}
}
```