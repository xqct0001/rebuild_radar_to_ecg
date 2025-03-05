import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader as TorchDataLoader
import numpy as np
from model_zkd import zkd_TSCN
import logging
import os
from tqdm import tqdm
from data_zkd_loader import DataLoader as CustomDataLoader
import matplotlib.pyplot as plt
from datetime import datetime
from torch.optim.lr_scheduler import ReduceLROnPlateau

# 设置日志
def setup_logging(config):
    """设置日志配置"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = os.path.join(config['checkpoint_dir'], 'logs')
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'training_{timestamp}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return timestamp

def plot_training_history(history, save_path):
    """绘制训练历史并保存"""
    plt.figure(figsize=(12, 4))
    
    # 损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

class EarlyStopping:
    """早停机制"""
    def __init__(self, patience=7, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        
        return self.early_stop

class RadarECGDataset(Dataset):
    def __init__(self, radar_data, position_data, ecg_data, qrs_masks):
        """
        Args:
            radar_data: 雷达信号数据 [N, 50, 1, 640]
            position_data: 位置信息 [N, 50, 3]
            ecg_data: ECG标签数据 [N, 1, 640]
            qrs_masks: QRS区域掩码 [N, 1, 640]
        """
        self.radar_data = torch.FloatTensor(radar_data)
        self.position_data = torch.FloatTensor(position_data)
        self.ecg_data = torch.FloatTensor(ecg_data)
        self.qrs_masks = torch.FloatTensor(qrs_masks)

    def __len__(self):
        return len(self.radar_data)

    def __getitem__(self, idx):
        return {
            'radar': self.radar_data[idx],
            'position': self.position_data[idx],
            'ecg': self.ecg_data[idx],
            'qrs_mask': self.qrs_masks[idx]
        }

def generate_qrs_mask(ecg_data, threshold=0.7):
    """
    从ECG数据生成QRS掩码
    Args:
        ecg_data: ECG数据 [N, 1, 640]
        threshold: 阈值，用于确定QRS区域
    Returns:
        qrs_mask: QRS区域掩码 [N, 1, 640]
    """
    abs_signal = np.abs(ecg_data)
    masks = []
    for i in range(len(ecg_data)):
        signal = abs_signal[i, 0]
        adaptive_threshold = threshold * np.max(signal)
        mask = (signal > adaptive_threshold).astype(np.float32)
        expanded_mask = np.zeros_like(mask)
        for j in range(len(mask)):
            if mask[j] == 1:
                start_idx = max(0, j-2)
                end_idx = min(len(mask), j+3)
                expanded_mask[start_idx:end_idx] = 1
        masks.append(expanded_mask)
    masks = np.array(masks)[:, np.newaxis, :]
    return masks

def train_model(model, train_loader, val_loader, config):
    """训练模型"""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Please check your GPU setup.")
    
    device = torch.device("cuda")
    logging.info(f"Using device: {device} ({torch.cuda.get_device_name()})")
    model = model.to(device)
    
    logging.info(f"GPU Memory Usage:")
    logging.info(f"Allocated: {torch.cuda.memory_allocated(0)//1024//1024}MB")
    logging.info(f"Cached: {torch.cuda.memory_reserved(0)//1024//1024}MB")
    
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=5
    )
    early_stopping = EarlyStopping(patience=config['early_stopping_patience'])
    criterion = nn.MSELoss()
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'learning_rates': []
    }
    
    best_val_loss = float('inf')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    try:
        for epoch in range(config['epochs']):
            model.train()
            train_losses = []
            
            pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config["epochs"]}')
            for batch in pbar:
                radar = batch['radar'].to(device)
                position = batch['position'].to(device)
                ecg = batch['ecg'].to(device)
                
                optimizer.zero_grad()
                output = model(radar, position)
                
                # 只使用MSE损失
                loss = criterion(output, ecg)
                
                loss.backward()
                optimizer.step()
                
                train_losses.append(loss.item())
                pbar.set_postfix({
                    'train_loss': f'{np.mean(train_losses):.6f}',
                    'mse_loss': f'{loss.item():.6f}'
                })
            
            avg_train_loss = np.mean(train_losses)
            
            # 验证阶段
            model.eval()
            val_losses = []
            
            with torch.no_grad():
                for batch in val_loader:
                    radar = batch['radar'].to(device)
                    position = batch['position'].to(device)
                    ecg = batch['ecg'].to(device)
                    
                    output = model(radar, position)
                    loss = criterion(output, ecg)
                    val_losses.append(loss.item())
            
            avg_val_loss = np.mean(val_losses)
            
            # 记录历史
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(avg_val_loss)
            
            # 记录日志
            logging.info(
                f'Epoch {epoch+1}: '
                f'Train Loss = {avg_train_loss:.6f}, '
                f'Val Loss = {avg_val_loss:.6f}'
            )
            
            # 保存最佳模型
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), config['best_model_path'])
                logging.info(f'Saved new best model with validation loss: {best_val_loss:.6f}')
            
            # 每10轮保存一次模型
            if (epoch + 1) % 10 == 0:
                checkpoint_path = os.path.join(config['checkpoint_dir'], f'model_epoch_{epoch+1}.pth')
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': avg_train_loss,
                    'val_loss': avg_val_loss
                }, checkpoint_path)
                logging.info(f'Saved checkpoint at epoch {epoch+1}')
            
            # 早停检查
            if early_stopping(avg_val_loss):
                logging.info(f'Early stopping triggered after {epoch+1} epochs')
                break
            
            # 定期清理GPU缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            # 在验证阶段结束后
            scheduler.step(avg_val_loss)
            current_lr = optimizer.param_groups[0]['lr']
            logging.info(f'Current learning rate: {current_lr}')
            
    except RuntimeError as e:
        if "out of memory" in str(e):
            logging.error("GPU内存不足，尝试减小batch_size或模型大小")
        raise e

    # 保存最终模型
    torch.save({
        'epoch': config['epochs'],
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': avg_train_loss,
        'val_loss': avg_val_loss,
        'history': history
    }, config['final_model_path'])
    
    # 保存训练历史图表
    plots_dir = os.path.join(config['checkpoint_dir'], 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    plot_path = os.path.join(plots_dir, f'training_history_{timestamp}.png')
    plot_training_history(history, plot_path)
    
    return history

def validate_config(config):
    """验证配置参数"""
    required_fields = [
        'learning_rate', 'batch_size', 'epochs', 'data_path',
        'train_ratio', 'val_ratio', 'test_ratio'
    ]
    
    for field in required_fields:
        if field not in config:
            raise ValueError(f"配置缺少必要字段: {field}")
    
    if not os.path.exists(config['data_path']):
        raise ValueError(f"数据路径不存在: {config['data_path']}")
    
    if sum([config['train_ratio'], config['val_ratio'], config['test_ratio']]) != 1.0:
        raise ValueError("数据集划分比例之和必须等于1")

def main():
    config = {
        # 训练相关参数
        'learning_rate': 0.0005,        # 学习率
        'batch_size': 128,             # 批次大小
        'epochs': 100,                 # 训练轮数
        'grad_clip_norm': 1.0,         # 梯度裁剪范数
        'early_stopping_patience': 15,  # 早停耐心值
        
        # 数据处理参数
        'sequence_length': 640,        # 序列长度
        'window_size': 640,            # 窗口大小
        'sample_step': 30,             # 采样步长
        'step_size': 30,               # 滑动步长
        'sample_freq': 200,            # 采样频率
        'qrs_threshold': 0.7,          # QRS波检测阈值
        
        # 数据集配置
        'data_path': 'E:/呼吸心跳公开数据集/中科大数据集/finalPartialPublicData20221108',  # 数据路径
        'num_files': 35,               # 使用的文件数量
        'train_ratio': 0.7,            # 训练集比例
        'val_ratio': 0.15,             # 验证集比例
        'test_ratio': 0.15,            # 测试集比例
        'num_workers': 4,              # 数据加载线程数
        
        # 模型保存配置
        'checkpoint_dir': './checkpoints',                      # 检查点保存目录
        'best_model_path': './checkpoints/best_model.pth',      # 最佳模型保存路径
        'final_model_path': './checkpoints/final_model.pth',    # 最终模型保存路径
        
        # GPU配置
        'device': 'cuda',              # 使用设备
        'cuda_device_id': 0            # GPU设备ID
    }
    
    # 创建保存目录
    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    
    # 设置日志
    setup_logging(config)
    
    validate_config(config)
    
    try:
        # 加载数据
        data_loader = CustomDataLoader(
            data_path=config['data_path'],
            num_files=config['num_files'],
            train_ratio=config['train_ratio'],
            val_ratio=config['val_ratio'],
            test_ratio=config['test_ratio']
        )
        datasets = data_loader.load_and_process_data()
        
        # 添加数据验证
        for split, data in datasets.items():
            if any(d is None or len(d) == 0 for d in data):
                raise ValueError(f"数据集 {split} 包含空值或长度为0的数组")
            if not all(isinstance(d, np.ndarray) for d in data):
                raise ValueError(f"数据集 {split} 包含非numpy数组数据")
            # 检查数据维度
            radar_shape = data[0].shape
            ecg_shape = data[1].shape
            position_shape = data[2].shape
            logging.info(f"{split} DATA shape: Radar {radar_shape}, ECG {ecg_shape}, Position {position_shape}")
        
        # 创建数据加载器
        train_dataset = RadarECGDataset(
            datasets['train'][0],  # radar_data
            datasets['train'][2],  # position_data
            datasets['train'][1],  # ecg_data
            generate_qrs_mask(datasets['train'][1])  # qrs_masks
        )
        train_loader = TorchDataLoader(
            train_dataset, 
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=config['num_workers']
        )
        
        val_dataset = RadarECGDataset(
            datasets['val'][0],    # radar_data
            datasets['val'][2],    # position_data
            datasets['val'][1],     # ecg_data
            generate_qrs_mask(datasets['val'][1])  # qrs_masks
        )
        val_loader = TorchDataLoader(
            val_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=config['num_workers']
        )
        
        # 初始化模型
        model = zkd_TSCN()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        criterion = nn.MSELoss()  # 定义损失函数
        
        # 训练模型
        history = train_model(model, train_loader, val_loader, config)
        
        logging.info("Training completed successfully!")
        
    except Exception as e:
        logging.error(f"An error occurred during training: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main() 

   


