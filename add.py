import re
import torch

def extract_loss_values(log_file_path):
    loss_values = []
    with open(log_file_path, 'r', encoding='utf-8') as file:
        for line in file:
            # 查找包含 'loss_average' 的行
            if 'loss_average' in line:
                # 使用正则表达式提取 'loss_average' 之后的数字
                match = re.search(r'loss_average\s*:\s*([\d\.]+)', line)
                if match:
                    loss_value = float(match.group(1))
                    loss_values.append(loss_value)
    return loss_values

# 文件路径
log_file_path1 = 'E:\\vscodeProject\\Githubcode\\OwnNet\\outputs\\训练与性能情况\\12-02_19-50\\损失日志.log'
log_file_path2 = 'E:\\vscodeProject\\Githubcode\\OwnNet\\outputs\\训练与性能情况\\12-03_10-47\\损失日志.log'

# 提取损失值
loss_values1 = extract_loss_values(log_file_path1)
loss_values2 = extract_loss_values(log_file_path2)

# 合并两个文件中的损失值
all_loss_values = loss_values1 + loss_values2


# 假设你的模型文件路径为 'model.pth'
model_path = 'E:\\vscodeProject\\Githubcode\\OwnNet\\outputs\\weights\\12-03_10-47\\36weights.pth'

# 加载现有的模型文件
checkpoint = torch.load(model_path)

# 假设 losses 是你要添加的损失值列表
losses = all_loss_values # 示例损失值列表

# 在 checkpoint 字典中添加 "loss": losses 键值对
checkpoint["losses"] = losses

# 保存更新后的模型文件
torch.save(checkpoint, model_path)

print("模型文件已更新，增加了 'loss': losses 键值对")
print("提取的损失值列表：", all_loss_values)