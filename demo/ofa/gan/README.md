# gan compression
[GAN Compression: Efficient Architectures for Interactive Conditional GANs](https://arxiv.org/abs/2003.08936) based on PaddlePaddle.

including follow six steps:
1. replace resnet block with mobile resnet block in generator.
2. cut off the channels of generator from step 1 as student generator, distill the student generator with the teacher generator get from step 1.
3. student generator get from step2 as [Once-For-All](https://arxiv.org/abs/1908.09791) supernet to finetune different generator architectures.
4. search to get FLOPs and evaluation value of different generator architectures.
5. (optional) finetune the generator architectures get from steps 4, only suit for some model and some dataset.
6. export final model.

## quick start

1. prepare data cyclegan used, the format is like:
```
├── trainA dictionary of trainA data
│   ├── img1.jpg
│   ├── ...
│  
├── trainB dictionary of trainB data
│   ├── img1.jpg
│   ├── ...
│  
├── trainA.txt list file of trainA, every line represent a image in trainA
│  
├── trainB.txt list file of trainB, every line represent a image in trainB
```

2. start to get a compressed model, incluing steps(1~3)
```python
sh run.sh
```

3. search for suitable architectures.
```python
python search.py
```

4. (optional)finetune the model.
```python
python finetune.sh
```

5. export final model
```python
python export.py --h
```
