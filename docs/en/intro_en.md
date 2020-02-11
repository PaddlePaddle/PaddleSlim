

# Introduction

As a submodule of PaddlePaddle framework, PaddleSlim is an open-source library for deep model compression and architecture search. PaddleSlim supports current popular deep compression techniques such as pruning, quantization, and knowledge distillation. Further, it also automates the search of hyperparameters and the design of lightweight deep architectures. In the future, we will develop more practically useful compression techniques for industrial-level applications and transfer these techniques to models in NLP.


## Methods

- Pruning
  - Uniform pruning
  - Sensitivity-based pruning
  - Automated model pruning

- Quantization
  - Training-aware quantization: Quantize models with hyperparameters dynamically estimated from small batches of samples.
  - Training-aware quantization: Quantize models with the same hyperparameters estimated from training data.
  - Support global quantization of weights and Channel-Wise quantization

- Knowledge Distillation
  - Single-process knowledge distillation
  - Multi-process distributed knowledge distillation

- Network Architecture Search（NAS）
  - Simulated Annealing (SA)-based lightweight network architecture search method.（Light-NAS）
  - One-Shot network structure automatic search. (One-Shot-NAS)
  - PaddleSlim supports FLOPs and latency constrained search.
  - PaddleSlim supports the latency estimation on different hardware and platforms.
