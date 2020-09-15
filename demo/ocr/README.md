[English](README_en.md) | 简体中文

# SlimOCR模型库


## 模型

PaddleSlim对[PaddleOCR]()发布的模型进行了压缩，产出了如下一系列小模型：


<table>
<thead>
  <tr>
    <th>序号</th>
    <th>任务</th>
    <th>模型</th>
    <th>压缩策略</th>
    <th>精度(自建中文数据集)</th>
    <th>耗时(ms)</th>
    <th>整体耗时(ms)</th>
    <th>加速比</th>
    <th>整体模型大小(M)</th>
    <th>压缩比例</th>
    <th>下载链接</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td rowspan="2">0</td>
    <td>检测</td>
    <td>MobileNetV3_DB</td>
    <td>无</td>
    <td>61.7</td>
    <td>224</td>
    <td rowspan="2">375</td>
    <td rowspan="2">-</td>
    <td rowspan="2">8.6</td>
    <td rowspan="2">-</td>
    <td></td>
  </tr>
  <tr>
    <td>识别</td>
    <td>MobileNetV3_CRNN</td>
    <td>无</td>
    <td>62.0</td>
    <td>9.52</td>
    <td></td>
  </tr>
  <tr>
    <td rowspan="2">1</td>
    <td>检测</td>
    <td>SlimTextDet</td>
    <td>PACT量化训练</td>
    <td>62.1</td>
    <td>195</td>
    <td rowspan="2">348</td>
    <td rowspan="2">8%</td>
    <td rowspan="2">2.8</td>
    <td rowspan="2">67.82%</td>
    <td></td>
  </tr>
  <tr>
    <td>识别</td>
    <td>SlimTextRec</td>
    <td>PACT量化训练</td>
    <td>61.48</td>
    <td>8.6</td>
    <td></td>
  </tr>
  <tr>
    <td rowspan="2">2</td>
    <td>检测</td>
    <td>SlimTextDet_quat_pruning</td>
    <td>剪裁+PACT量化训练</td>
    <td>60.86</td>
    <td>142</td>
    <td rowspan="2">288</td>
    <td rowspan="2">30%</td>
    <td rowspan="2">2.8</td>
    <td rowspan="2">67.82%</td>
    <td></td>
  </tr>
  <tr>
    <td>识别</td>
    <td>SlimTextRec</td>
    <td>PACT量化训练</td>
    <td>61.48</td>
    <td>9.00</td>
    <td></td>
  </tr>
  <tr>
    <td rowspan="2">3</td>
    <td>检测</td>
    <td>SlimTextDet_pruning</td>
    <td>剪裁</td>
    <td>61.57</td>
    <td>138</td>
    <td rowspan="2">295</td>
    <td rowspan="2">27%</td>
    <td rowspan="2">2.9</td>
    <td rowspan="2">66.28%</td>
    <td></td>
  </tr>
  <tr>
    <td>识别</td>
    <td>SlimTextRec</td>
    <td>PACT量化训练</td>
    <td>61.48</td>
    <td>8.4</td>
    <td></td>
  </tr>
</tbody>
</table>


**注意**:

-   <a name="lantancy">[1]</a> 耗时评测环境为：骁龙855芯片+PaddleLite。
-   <a name="quant">[2]</a> 参考下面关于[OCR量化的说明](#OCR量化说明)。
-   <a name="prune">[3]</a> 参考下面关于[OCR剪裁的说明](#OCR剪裁说明)。


## OCR量化说明

待补充

分别针对检测和识别模型说明以下内容：
1. PACT量化所选参数：包括量化算法参数、训练轮数、优化器选择、学习率等信息
2. 跳过了哪些层


更多量化教程请参考[OCR模型量化压缩教程]()


## OCR剪裁说明

待补充

针对检测模型说明以下内容：
1. 剪裁了哪些层，以及对应的比例
2. 训练参数


更多OCR剪裁教程请参考[OCR模剪裁压缩教程]()
