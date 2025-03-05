import torch
import numpy as np
import matplotlib.pyplot as plt
from data_zkd_loader import DataLoader as CustomDataLoader
from model_zkd import zkd_TSCN
import os
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader as TorchDataLoader
import scipy.io as sio

class RadarECGDataset(Dataset):
    def __init__(self, radar_data, position_data, ecg_data):
        self.radar_data = torch.FloatTensor(radar_data).cuda()
        self.position_data = torch.FloatTensor(position_data).cuda()
        self.ecg_data = torch.FloatTensor(ecg_data).cuda()

    def __len__(self):
        return len(self.radar_data)

    def __getitem__(self, idx):
        return {
            'radar': self.radar_data[idx],
            'position': self.position_data[idx],
            'ecg': self.ecg_data[idx]
        }

def plot_and_save_comparison(true_ecg, pred_ecg, save_path, index):
    """Plot and save comparison between true and predicted ECG values"""
    plt.figure(figsize=(15, 5))
    
    # Move tensors to CPU before converting to numpy
    true_ecg = true_ecg.detach().cpu().numpy().flatten()
    pred_ecg = pred_ecg.detach().cpu().numpy().flatten()
    
    # 创建时间轴
    time = np.arange(len(true_ecg)) / 200  # 假设采样率为200Hz
    
    # 绘制真实值
    plt.plot(time, true_ecg, 'b-', label='True ECG', alpha=0.7)
    # 绘制预测值
    plt.plot(time, pred_ecg, 'r-', label='Predicted ECG', alpha=0.7)
    
    plt.title(f'ECG Comparison - Sample {index}')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)
    
    # 保存图片
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # Configuration parameters
    config = {
        'data_path': 'E:/呼吸心跳公开数据集/中科大数据集/finalPartialPublicData20221108',
        'model_path': './checkpoints/best_model.pth',  # Load best model
        'output_dir': './test_results',  # Results save directory
        'batch_size': 10,  
        'device': 'cuda',  # Remove conditional, force CUDA usage
        'save_mat': True,  # Whether to save as mat file
        'mat_filename': 'ecg_results.mat'  # mat filename
    }
    
    # 创建输出目录
    os.makedirs(config['output_dir'], exist_ok=True)
    
    # 加载数据
    data_loader = CustomDataLoader(
        data_path=config['data_path'],
        num_files=35,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15
    )
    datasets = data_loader.load_and_process_data()
    
    # 创建测试数据集
    test_dataset = RadarECGDataset(
        datasets['test'][0],  # radar_data (RCG)
        datasets['test'][2],  # position_data
        datasets['test'][1]   # ecg_data
    )
    test_loader = TorchDataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False
    )
    
    # 加载模型
    model = zkd_TSCN().to(config['device'])
    # 直接加载状态字典
    model.load_state_dict(torch.load(config['model_path'], weights_only=True))
    model.eval()
    
    # 用于存储所有预测结果和真实值
    all_predictions = []
    all_true_values = []
    
    # Start prediction and save results
    print("Starting to generate test results...")
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(test_loader)):
            # 获取数据 (no need to move to device as dataset already on GPU)
            radar = batch['radar']
            position = batch['position']
            true_ecg = batch['ecg']
            
            # 预测
            pred_ecg = model(radar, position)
            
            # 保存对比图
            save_path = os.path.join(config['output_dir'], f'ecg_comparison_{idx}.png')
            plot_and_save_comparison(true_ecg[0], pred_ecg[0], save_path, idx)
            
            # 收集预测结果和真实值
            all_predictions.append(pred_ecg[0].detach().cpu().numpy().flatten())
            all_true_values.append(true_ecg[0].detach().cpu().numpy().flatten())
    
    # Save as mat file
    if config['save_mat']:
        # Create a cell array with N rows and 2 columns, each row stores a set of predicted and true values
        N = len(all_predictions)
        data = np.empty((N, 2), dtype=object)
        
        # Fill data
        for i in range(N):
            data[i, 0] = all_predictions[i]  # Predicted values
            data[i, 1] = all_true_values[i]  # True values
        
        # Save results to mat file
        mat_path = os.path.join(config['output_dir'], config['mat_filename'])
        sio.savemat(mat_path, {'data': data})
        print(f"Results saved to {mat_path}")
    
    print(f"Testing completed! Results saved to {config['output_dir']} directory")

if __name__ == "__main__":
    main() 