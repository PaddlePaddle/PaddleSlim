## 1. 安装依赖

### 1.1 安装PaddleSlim

```
git clone https://github.com/PaddlePaddle/PaddleSlim.git
cd PaddleSlim
python setup.py install
```

### 1.2 安装pytorch

```
pip install torch torchvision
```

## 2. Import依赖与环境设置


```python
from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.optim.lr_scheduler import StepLR
from paddleslim.dist import DML

args = {"batch-size": 256,
        "test-batch-size": 256,
        "epochs": 10,
        "lr": 1.0,
        "gamma": 0.7,
        "seed": 1,
        "log-interval": 10}



use_cuda = torch.cuda.is_available()
torch.manual_seed(args["seed"])
device = torch.device("cuda" if use_cuda else "cpu")
```

## 3. 准备数据



```python


kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                       ])),
        batch_size=args["batch_size"], shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('../data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                       ])),
        batch_size=args["test_batch_size"], shuffle=True, **kwargs)


```


## 4. 定义模型


```python
model = models.mobilenet_v2(num_classes=10).to(device)
optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
```

## 5. 添加DML修饰
### 5.1 将模型转为DML模型


```python
model = DML(model)
```

### 5.2 将优化器转为DML优化器


```python
optimizer = model.opt(optimizer)
scheduler = model.lr(scheduler)
```

### 6. 定义训练方法

将原来的交叉熵损失替换为DML损失，代码如下：


```python
def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = model.dml_loss(output, target)  
#        output = F.softmax(output, dim=1)
#        loss = F.cross_entropy(output, target)
#        loss.backward()
        optimizer.step()
        if batch_idx % args["log_interval"] == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
```

## 7. 定义测试方法


```python

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            output = F.softmax(output, dim=1)
            loss = F.cross_entropy(output, target, reduction="sum")
            test_loss += loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
```

## 8. 开始训练


```python
epochs = 10
for epoch in range(1, epochs + 1):
    train(args, model, device, train_loader, optimizer, epoch)
    test(model, device, test_loader)
    scheduler.step()
```
