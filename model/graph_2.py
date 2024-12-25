# 绘制各类性能曲线
import matplotlib.pyplot as plt
import os,sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), './..'))
sys.path.append(project_root)



def allgraph(current_datetime,epoches,train_losses,train_psnrs,train_ssims,val_losses,val_psnrs,val_ssims,power_datas):
    # 训练集损失曲线
    plt.figure(figsize=(20, 15))
    plt.subplot(3, 3, 1)
    plt.plot(epoches, train_losses, 'b', label='loss')
    plt.title('Training set Loss over Epochs')
    plt.xlabel('Epochs')    
    plt.ylabel('Loss')
    plt.legend()

    # 训练集PSNR曲线
    plt.subplot(3, 3, 2)
    plt.plot(epoches, train_psnrs, 'b', label='PSNR')
    plt.title('Training set PSNR over Epochs')
    plt.xlabel('Epochs')    
    plt.ylabel('PSNR')
    plt.legend()
    
    # 训练集SSIM曲线
    plt.subplot(3, 3, 3)
    plt.plot(epoches, train_ssims, 'b', label='SSIM')
    plt.title('Training set SSIM over Epochs')
    plt.xlabel('Epochs')    
    plt.ylabel('SSIM')
    plt.legend()

    # 验证集损失曲线
    plt.subplot(3, 3, 4)
    plt.plot(epoches, val_losses, 'r', label='loss')
    plt.title('Verification set loss over Epochs')
    plt.xlabel('Epochs')    
    plt.ylabel('Loss')
    plt.legend()

    # 验证集PSNR曲线
    plt.subplot(3, 3, 5)
    plt.plot(epoches, val_psnrs, 'r', label='PSNR')
    plt.title('Verification set PSNR over Epochs')
    plt.xlabel('Epochs')    
    plt.ylabel('PSNR')
    plt.legend()

    # 验证集SSIM曲线
    plt.subplot(3, 3, 6)
    plt.plot(epoches, val_ssims, 'r', label='SSIM')
    plt.title('Verification set SSIM over Epochs')
    plt.xlabel('Epochs')    
    plt.ylabel('SSIM')
    plt.legend()

    # 效果值power_data曲线
    plt.subplot(3, 3, 8)
    plt.plot(epoches, power_datas, 'g', label='power_data')
    plt.title('Verification set powerdata over Epochs')
    plt.xlabel('Epochs')    
    plt.ylabel('Power_data')
    plt.legend()

    plt.savefig(f'{project_root}/outputs/训练与性能情况/{current_datetime}/参数曲线.png')  # 保存训练损失图像
    plt.clf()