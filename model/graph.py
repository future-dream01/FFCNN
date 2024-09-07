# 绘制各类性能曲线
import matplotlib.pyplot as plt
import os,sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), './..'))
sys.path.append(project_root)

def loss_graph(current_datetime,epoches,epoch_losses):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epoches, epoch_losses, 'b', label='Training loss')
    plt.title('Training Loss over Epochs')
    plt.xlabel('Epochs')    
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'{project_root}/outputs/训练与性能情况/{current_datetime}/经验误差.png')  # 保存训练损失图像
