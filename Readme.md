# 基于时空卷积网络的雷达心电信号重建系统

## 项目概述

本项目实现了一个基于时空卷积网络(TSCN)的雷达信号到心电信号的重建系统。系统通过处理雷达反射信号和位置信息，重建出高质量的心电(ECG)信号。

## 系统架构

系统主要包含以下几个核心模块：

1. 时序编码器 (Temporal Encoder)
2. 空间编码器 (Spatial Encoder)
3. 特征扩展器 (Feature Expander)
4. 时序卷积网络解码器 (TCN Decoder)

### 数据流程

输入数据 \(\rightarrow\) 时序特征提取 \(\rightarrow\) 空间特征融合 \(\rightarrow\) 特征扩展 \(\rightarrow\) ECG信号重建

## 核心模块详解

### 1. 时序编码器 (TemporalEncoder)

时序编码器采用多层卷积结构，用于提取雷达信号的时序特征。

输入维度: \([B, 50, 1, 640]\)
输出维度: \([B, 50, 32, 80]\)

其中：
- B: 批次大小
- 50: 信号数量
- 640: 时间步长
- 32: 特征通道数

结构包含3个重复的卷积块，每个块包含：
- 两个1D卷积层
- 批归一化层
- ReLU激活
- 最大池化层

### 2. 空间编码器 (SpatialEncoder)

空间编码器使用Transformer结构处理位置信息和时序特征的融合。

主要组件：
- 时序特征投影层: \(32 \rightarrow 32\)
- 位置信息投影层: \(3 \rightarrow 32\)
- 3个Transformer块

每个Transformer块的注意力机制可表示为：

\[
Attention(Q,K,V) = softmax(\frac{QK^T}{\sqrt{d_k}})V
\]

### 3. 特征扩展器 (FeatureExpander)

将压缩的特征逐步扩展回原始信号维度。包含四层转置卷积结构：

1. 32 \(\rightarrow\) 16 通道, 80 \(\rightarrow\) 160 时间步
2. 16 \(\rightarrow\) 8 通道, 160 \(\rightarrow\) 320 时间步
3. 8 \(\rightarrow\) 4 通道, 320 \(\rightarrow\) 640 时间步
4. 4 \(\rightarrow\) 4 通道, 维持640时间步

### 4. TCN解码器 (TCNDecoder)

采用因果卷积网络进行最终的信号重建，主要特点：

- 膨胀因子: \(2^i\), i = 0,1,2,...
- 感受野: \(2^L - 1\), L为层数
- 残差连接
- 因果卷积保证时序性

## 训练策略

### 损失函数
模型仅采用均方误差(MSE)损失函数：

\[
L = L_{MSE} = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2
\]

其中：
- \(y_i\): 真实ECG信号值
- \(\hat{y}_i\): 模型预测的ECG信号值
- \(n\): 样本数量

损失函数实现：
```python
criterion = nn.MSELoss()
loss = criterion(output, ecg)
```

### 优化策略

1. 优化器配置：
```python
optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
```

2. 学习率调度：
```python
scheduler = ReduceLROnPlateau(
    optimizer, 
    mode='min',
    factor=0.5,
    patience=5
)
```

3. 早停机制：
```python
early_stopping = EarlyStopping(
    patience=config['early_stopping_patience'],
    min_delta=0
)
```

4. 训练过程监控：
- 训练损失
- 验证损失
- 学习率变化
- GPU内存使用

## 数据处理

数据集划分比例：
- 训练集: 70%
- 验证集: 15%
- 测试集: 15%

数据维度：
- 雷达信号: \([N, 50, 1, 640]\)
- 位置信息: \([N, 50, 3]\)
- ECG标签: \([N, 1, 640]\)

## 实验结果

模型评估指标包括：
- MSE损失
- 训练和验证曲线
- 信号重建质量可视化

## 环境要求

- Python 3.7+
- PyTorch 1.7+
- CUDA支持
- NumPy
- Matplotlib
- tqdm

## 使用说明

1. 配置参数在main函数中的config字典中设置
2. 数据路径设置
3. 模型训练启动
4. 结果可视化和评估
