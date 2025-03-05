# # 使用当前目录作为起始路径
# list_files('.')
import numpy as np
import scipy.io as sio
import os
from sklearn.model_selection import train_test_split

class DataLoader:
    def __init__(self, data_path, num_files=35, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, split_by_file=True):
        """
        初始化数据加载器
        Args:
            data_path: 数据文件路径
            num_files: 要加载的文件数量
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            test_ratio: 测试集比例
            split_by_file: 是否按文件划分（True为按文件数量划分，False为按每个文件的数据比例划分）
        """
        self.data_path = data_path
        self.num_files = num_files  # 移除最大文件数限制
        self.window_size = 640
        self.step_size = 30
        
        # 检查比例合法性
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5, "比例之和必须等于1"
        
        # 计算每个集合应该包含的文件数量
        self.train_files = int(self.num_files * train_ratio)
        self.val_files = int(self.num_files * val_ratio)
        self.test_files = self.num_files - self.train_files - self.val_files
        self.split_by_file = split_by_file

    def load_and_process_data(self):
        """加载并处理所有数据"""
        if self.split_by_file:
            return self._load_and_split_by_files()
        else:
            return self._load_and_split_by_ratio()

    def _load_and_split_by_files(self):
        """按文件数量划分数据集（原有方法）"""
        train_rcg, train_ecg, train_pos = [], [], []
        val_rcg, val_ecg, val_pos = [], [], []
        test_rcg, test_ecg, test_pos = [], [], []

        # 遍历文件
        for file_idx in range(1, self.num_files + 1):
            file_path = os.path.join(self.data_path, f"{file_idx}.mat")
            try:
                # 加载mat文件
                mat_data = sio.loadmat(file_path)
                data = mat_data['data'][0, 0]  # 获取data结构体

                # 获取数据长度
                data_length = data['RCG'].shape[0]
                
                # 计算窗口数量
                num_windows = (data_length - self.window_size) // self.step_size + 1

                # 临时存储当前文件的所有窗口数据
                current_rcg = []
                current_ecg = []
                current_pos = []

                # 对每个窗口进行处理
                for win_idx in range(num_windows):
                    start_idx = win_idx * self.step_size
                    end_idx = start_idx + self.window_size

                    # 提取数据片段并转置
                    rcg_segment = data['RCG'][start_idx:end_idx].T  # 转置RCG
                    ecg_segment = data['ECG'][start_idx:end_idx].T  # 转置ECG
                    pos_xyz = data['posXYZ']  # posXYZ保持不变

                    current_rcg.append(rcg_segment)
                    current_ecg.append(ecg_segment)
                    current_pos.append(pos_xyz)

                # 根据文件索引决定将数据添加到哪个集合
                if file_idx <= self.train_files:
                    train_rcg.extend(current_rcg)
                    train_ecg.extend(current_ecg)
                    train_pos.extend(current_pos)
                elif file_idx <= self.train_files + self.val_files:
                    val_rcg.extend(current_rcg)
                    val_ecg.extend(current_ecg)
                    val_pos.extend(current_pos)
                else:
                    test_rcg.extend(current_rcg)
                    test_ecg.extend(current_ecg)
                    test_pos.extend(current_pos)

            except Exception as e:
                print(f"处理文件 {file_idx} 时出错: {str(e)}")
                continue

        # 转换为numpy数组
        datasets = {
            'train': (np.array(train_rcg), np.array(train_ecg), np.array(train_pos)),
            'val': (np.array(val_rcg), np.array(val_ecg), np.array(val_pos)),
            'test': (np.array(test_rcg), np.array(test_ecg), np.array(test_pos))
        }

        return datasets

    def _load_and_split_by_ratio(self):
        """按每个文件的数据比例划分数据集"""
        all_rcg, all_ecg, all_pos = [], [], []

        # 遍历文件
        for file_idx in range(1, self.num_files + 1):
            file_path = os.path.join(self.data_path, f"{file_idx}.mat")
            try:
                # 加载mat文件
                mat_data = sio.loadmat(file_path)
                data = mat_data['data'][0, 0]

                # 获取数据长度
                data_length = data['RCG'].shape[0]
                
                # 计算窗口数量
                num_windows = (data_length - self.window_size) // self.step_size + 1

                # 临时存储当前文件的所有窗口数据
                current_rcg = []
                current_ecg = []
                current_pos = []

                # 对每个窗口进行处理
                for win_idx in range(num_windows):
                    start_idx = win_idx * self.step_size
                    end_idx = start_idx + self.window_size

                    rcg_segment = data['RCG'][start_idx:end_idx].T
                    ecg_segment = data['ECG'][start_idx:end_idx].T
                    pos_xyz = data['posXYZ']

                    current_rcg.append(rcg_segment)
                    current_ecg.append(ecg_segment)
                    current_pos.append(pos_xyz)

                # 对当前文件的数据进行划分
                current_rcg = np.array(current_rcg)
                current_ecg = np.array(current_ecg)
                current_pos = np.array(current_pos)

                # 计算划分点
                n_samples = len(current_rcg)
                train_idx = int(n_samples * 0.8)
                val_idx = int(n_samples * 0.9)

                # 添加到总数据列表
                all_rcg.extend([
                    current_rcg[:train_idx],      # 训练集
                    current_rcg[train_idx:val_idx],  # 验证集
                    current_rcg[val_idx:]         # 测试集
                ])
                all_ecg.extend([
                    current_ecg[:train_idx],
                    current_ecg[train_idx:val_idx],
                    current_ecg[val_idx:]
                ])
                all_pos.extend([
                    current_pos[:train_idx],
                    current_pos[train_idx:val_idx],
                    current_pos[val_idx:]
                ])

            except Exception as e:
                print(f"处理文件 {file_idx} 时出错: {str(e)}")
                continue

        # 合并所有文件的数据
        train_idx = 0
        val_idx = 1
        test_idx = 2
        
        # 创建最终的数据集字典，包含训练、验证和测试集
        datasets = {
            # 训练集：合并所有标记为训练集的RCG、ECG和位置数据
            'train': (np.concatenate([x for i, x in enumerate(all_rcg) if i % 3 == train_idx]),  # RCG数据
                     np.concatenate([x for i, x in enumerate(all_ecg) if i % 3 == train_idx]),   # ECG数据
                     np.concatenate([x for i, x in enumerate(all_pos) if i % 3 == train_idx])),  # 位置数据
            
            # 验证集：合并所有标记为验证集的RCG、ECG和位置数据
            'val': (np.concatenate([x for i, x in enumerate(all_rcg) if i % 3 == val_idx]),     # RCG数据
                   np.concatenate([x for i, x in enumerate(all_ecg) if i % 3 == val_idx]),      # ECG数据
                   np.concatenate([x for i, x in enumerate(all_pos) if i % 3 == val_idx])),     # 位置数据
            
            # 测试集：合并所有标记为测试集的RCG、ECG和位置数据
            'test': (np.concatenate([x for i, x in enumerate(all_rcg) if i % 3 == test_idx]),   # RCG数据
                    np.concatenate([x for i, x in enumerate(all_ecg) if i % 3 == test_idx]),    # ECG数据
                    np.concatenate([x for i, x in enumerate(all_pos) if i % 3 == test_idx]))    # 位置数据
        }

        return datasets

# 使用示例
if __name__ == "__main__":
    # 设置数据路径
    DATA_PATH = 'E:/呼吸心跳公开数据集/中科大数据集/finalPartialPublicData20221108'
    
    # 创建数据加载器实例
    loader = DataLoader(
        data_path=DATA_PATH,
        num_files=60,  # 可以修改要加载的文件数量
        train_ratio=0.7,  # 70% 的文件用于训练
        val_ratio=0.15,  # 15% 的文件用于验证
        test_ratio=0.15,  # 15% 的文件用于测试
        split_by_file=True
    )
    
    # 加载和处理数据
    datasets = loader.load_and_process_data()
    
    # 打印数据集信息
    for split_name, (rcg, ecg, pos) in datasets.items():
        print(f"\n{split_name} 集合:")
        print(f"RCG shape: {rcg.shape}")
        print(f"ECG shape: {ecg.shape}")
        print(f"POS shape: {pos.shape}")