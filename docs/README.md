# 文档构建与发布教程

## 1. 文档构成

PaddleSlim文档包含以下部分：

- 简介：概要介绍PaddleSlim功能。
- 安装：安装文档。
- 快速开始：各个策略示例，使用小数据，可快速完成执行。
- 高阶教程：包括在实际任务上的操作步骤、高级特性的使用教程。
- API文档：用户接口文档。

以上文档均包含中英两版，其中，**英文API文档根据代码注释自动生成**。

文档文件结构如下：

```bash
docs
├── en
│   ├── api_en # 英文API文档，该目录下文件为自动生成
│   ├── conf.py # 英文文档编译配置文件
│   ├── index_en.rst # 英文文档总导航页
│   ├── index.rst # 中英文切换功能的辅助文件，无实际内容
│   ├── install_en.md # 安装文档
│   ├── intro_en.md # 简介
│   ├── Makefile # 英文文档编译文件
│   ├── model_zoo_en.md # 模型库
│   ├── quick_start # 快速开始
│   └── tutorials # 进阶教程
├── requirements.txt # 文档编译所需依赖
└── zh_cn
    ├── algo # 算法原理
    ├── api_cn # 中文API文档
    ├── conf.py # 中文文档编译配置文件
    ├── index_en.rst # 中英文切换功能的辅助文件，无实际内容
    ├── index.rst # 中文文档总导航页
    ├── install.md # 安装文档
    ├── intro.md # 简介
    ├── Makefile # 编译文件
    ├── model_zoo.md # 模型库
    ├── quick_start # 快速开始
    └── tutorials # 进阶教程
```

## 2. 文档编译

编译文档前需要确保已正确安装PaddleSlim，且Python可正常执行`import paddleslim`。

执行以下命令安装文档编译依赖工具库：

```
pip install -r requirements.txt
```

##  2.1 编译中文文档

进入路径`./docs/zh_cn`

执行以下命令清理编译结果：

```
make clean
```

执行以下命令编译生成`html`:

```
make html
```

以上命令生成`html`文件到路径`./build/html/`。


## 2.2 预览文档

进入`PaddleSlim/docs/zh_cn/build/html`路径下。
执行`python -m SimpleHTTPServer 8883`。
假设当前机器IP为`server_ip`。

通过浏览器查看链接`server_ip:8883`即可预览文档。

## 2.3 编译英文文档

进入路径`PaddleSlim/docs/en`

编译文档前，需要先从代码注释生成API文档。

### 2.2.1 自动生成API

```
sphinx-apidoc -M -o api_en/ ../../paddleslim
```

如果有新增`package`，请将其添加到`./api_en/index_en.rst`文件中。


### 2.2.2 编译文档

与2.1节步骤一样。

# 3. 发布页面到Github

回到路径`PaddleSlim/`。

切换分支到`gh-pages`:

```
git checkout gh-pages
```

>注：直接切换到gh-pages分支可能会出现异常，可以尝试先切换到develop分支，再切到gh-pages分支。


```
rm docs/en/build/html/index.html
rm docs/zh_cn/build/html/index_en.html
cp -rf docs/en/build/html/* ./
cp -rf docs/zh_cn/build/html/* ./
```

执行以下命令，添加更新：
```
git add -u
```

如果有新增html页面，需要单独对其执行`git add`。

提交commit，并push到github。

```
git commit -m "update pages"
git push origin gh-pages
```

## 4. 其它

英文API文档格式请参考：https://wanghaoshuang.github.io/PaddleSlim/api_en/paddleslim.analysis.html
