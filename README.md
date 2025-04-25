<h1 align="center">FFCNN
（用于图像去噪的特征融合卷积神经网络）</h1>

**使用方法：**
1. 本项目的文件结构如下：
   - datasets文件夹：存放训练集和验证集
   - demo文件夹：存放训练脚本(`train.py`)和推理脚本(`detect.py`)
   - model文件夹：存放数据加载(`dataload.py`)、模型结构(`net.py`)、损失计算(`loss.py`)、参数图像绘制(`graph.py`)的脚本
   - outputs文件夹
     - **推理结果**：存放每次推理的结果图像，其下文件夹以日期时间命名
     - **训练与性能情况**：存放每次训练的日志文件、参数曲线、每轮训练的效果图片，其下文件夹以日期时间命名
     - **weights**：存放每次训练的权重文件，其下文件夹以日期时间命名
  
2. 本项目的训练方法：
   - 首先将训练集和验证集放入datasets文件夹
   - 在`dataload.py`脚本中修改**train_images_dir**、**train_labels_dir**、**val_image_dir**、**val_lable_dir**，替换为现在训练集、验证集路径
   - 在`train.py`中设置轮次数**EPOCHES**、批次数**BATCHSIZE**；若是从头开始训练，则**LOAD_CP**值需为False，表示不需要加载之前的检查点，若从上次中断的检查点开始训练，则**LOAD_CP**需为True，同时设置**CP_PATH**为检查点权重文件路径。
   - 运行`train.py`即可开始训练，训练性能情况在**训练与性能情况**文件夹中；权重文件在**weights**文件夹中

3. 本项目的推理方法：
   - 直接修改`detect.py`脚本中的**weight_name**参数为推理所使用的权重文件路径，运行`detect.py`即可，推理结果在outputs文件夹下的**推理结果**文件夹中
