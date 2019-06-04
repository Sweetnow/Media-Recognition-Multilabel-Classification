# 图像识别第28组代码使用方法

## 源代码文件
* `dataset.py`: 数据集预处理与`Pytorch`中`Dataset`的生成
* `demo.py`: 展示Demo与预测结果可视化
* `main.py`: 模型训练主程序
* `model.py`: VGG16/ResNet/DenseNet的预训练实现与部分层冻结
* `test.py`: 测试函数
* `train.py`: 训练函数
* `utils.py`: 工具函数与参数

## 使用方法
### 模型训练
模型训练主程序为`main.py`，采用`argparse`模块提供参数输入，训练得到的模型会保存为`<模型名称>_<模型参数个数>.pt`，训练过程会保存为`<模型名称>_<模型参数个数>.json`。
提供的可设置参数如下：
|参数|简介|
|-|-|
|--path|数据集文件夹路径|
|--model|选用的模型（可选ResNet18, ResNet34, ResNet50, DenseNet）|
|--layers|训练的层*|
|--lr|学习率|
|--batch|batch大小|
|--augmentation<br>--no-augmentation|是否启用数据增强|
|--fivecrop<br>--no-fivecrop|是否切分图像后测试|
|--worker|`DataLoader`对应参数|
|--early|提前终止训练所需的性能下降次数|
|--epoch|训练的最大epoch|
|--log|训练过程输出日志的batch间隔|
|--cuda|启用GPU|

其中--layers可选参数如下：
|参数|简介|
|-|-|
|fc|(默认)只训练全连接层|
|conv|训练全连接层与最后一层卷积|
|none|训练所有部分|

### Demo
Demo程序利用`dataset.py`生成数据集，采用给定的模型在测试集指定的图片上进行预测并显示预测结果，结果标题中label字样后为正确标签，pred字样后为预测标签。
Demo程序使用方法如下：
1. 根据是否使用图片切分，修改`use_fivecrop`（`True`|`False`）
2. 根据数据集文件夹路径、模型类型与模型路径，分别修改`dataset_path`、`used_model`与`model_path`
3. 启动程序，输入要预测的图片序号（从0开始，与图片名无关）
4. 程序显示结果，关闭窗口后可继续输入要预测的图片序号（输入-1可退出）