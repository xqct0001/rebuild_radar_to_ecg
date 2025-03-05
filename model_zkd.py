import torch
import torch.nn as nn
import numpy as np
import math
from transformers import  ViTConfig,ViTModel
import cv2
from random import Random
import matplotlib.pyplot as plt
import warnings


# 自定义1D卷积层，处理4D输入张量 (batch_size, num_signals, channels, T)
class CustomConv1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding='same'):
        super(CustomConv1D, self).__init__()
        self.conv1d = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        
    def forward(self, inputs):
        # 获取输入形状
        batch_size, num_signals, channels, T = inputs.size()

        # 重塑输入为 (batch_size * num_signals, channels, T)
        inputs_reshaped = inputs.view(-1, channels, T)

        # 应用卷积操作
        convolved = self.conv1d(inputs_reshaped)

        # 重塑回原始维度 (batch_size, num_signals, channels', T')
        T_convolved = convolved.size(-1)
        convolved_reshaped = convolved.view(batch_size, num_signals, -1, T_convolved)

        return convolved_reshaped

# 自定义1D最大池化层
class CustomMaxPooling1D(nn.Module):
    def __init__(self, pool_size=2, strides=2):
        super(CustomMaxPooling1D, self).__init__()
        self.pool_size = pool_size
        self.strides = strides

    def forward(self, inputs):
        # 获取输入形状
        batch_size, num_signals, channels, T = inputs.size()

        # 重塑为 (batch_size * num_signals, channels, T)
        inputs_reshaped = inputs.view(-1, channels, T)

        # 应用最大池化
        pooled = nn.functional.max_pool1d(inputs_reshaped, kernel_size=self.pool_size, stride=self.strides)

        # 重塑回原始维度
        T_pooled = pooled.size(-1)
        pooled_reshaped = pooled.view(batch_size, num_signals, -1, T_pooled)
        return pooled_reshaped
      
# 自定义1D转置卷积层
class CustomConv1DTranspose(nn.Module):
    def __init__(self, padding, input_channels ,filters, kernel_size, strides=1, activation='relu'):
        super(CustomConv1DTranspose, self).__init__()
        self.input_channels = input_channels
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.activation = activation

        self.conv_transpose = nn.ConvTranspose1d(
            in_channels=input_channels,
            out_channels=filters,
            kernel_size=kernel_size,
            stride=strides,
            padding=padding
        )

    def forward(self, inputs):
        inputs = inputs.contiguous()

        # 获取输入形状
        batch_size, num_signals, channels, T = inputs.size()

        # 重塑为 (batch_size * num_signals, channels, T)
        inputs_reshaped = inputs.view(-1, channels, T)

        # 应用转置卷积
        conv_transposed = self.conv_transpose(inputs_reshaped)

        # 重塑回原始维度
        T_transposed = conv_transposed.size(-1)
        transposed_reshaped = conv_transposed.view(batch_size, num_signals, self.filters, T_transposed)

        return transposed_reshaped    

# 自定义批归一化层
class CustomBatchNorm1d(nn.Module):
    def __init__(self, channels):
        super(CustomBatchNorm1d, self).__init__()
        self.channels = channels
        self.batchnorm = nn.BatchNorm1d(channels)

    def forward(self, x):
        batch_size = x.size(0)
        num_signals = x.size(1)
        T = x.size(3)
        
        # 重塑为 (batch_size * num_signals, channels, T)
        x_reshaped = x.view(-1, self.channels, T)

        # 应用批归一化
        output = self.batchnorm(x_reshaped)

        # 重塑回原始维度
        output = output.view(batch_size, num_signals, -1, T)

        return output

# 时序编码器CNN网络
class TemporalEncoder(nn.Module):
    def __init__(self):
        super(TemporalEncoder, self).__init__()
        
        # 创建3个重复的层（改为3层而不是4层）
        self.layers = nn.ModuleList()
        input_channels = 1
        output_channels = 8  # 调整初始输出通道，使得最终能达到32通道
        
        for _ in range(3):  # 改为3次循环
            layer = nn.ModuleDict({
                'conv1': CustomConv1D(input_channels, output_channels, kernel_size=7, padding='same'),
                'bn1': CustomBatchNorm1d(output_channels),
                'conv2': CustomConv1D(output_channels, output_channels, kernel_size=7, padding='same'),
                'bn2': CustomBatchNorm1d(output_channels),
                'pool': CustomMaxPooling1D(pool_size=2, strides=2)
            })
            self.layers.append(layer)
            input_channels = output_channels
            output_channels *= 2
            
        # 添加最后一层，但不包含池化
        self.final_layer = nn.ModuleDict({
            'conv1': CustomConv1D(input_channels, 32, kernel_size=7, padding='same'),
            'bn1': CustomBatchNorm1d(32),
            'conv2': CustomConv1D(32, 32, kernel_size=7, padding='same'),
            'bn2': CustomBatchNorm1d(32)
        })
            
        self.relu = nn.ReLU()

    def forward(self, x):
        # 输入 x 形状: (batch_size, 50, 1, 640)
        for layer in self.layers:
            x = self.relu(layer['bn1'](layer['conv1'](x)))
            x = self.relu(layer['bn2'](layer['conv2'](x)))
            x = layer['pool'](x)
        
        # 最后一层没有池化
        x = self.relu(self.final_layer['bn1'](self.final_layer['conv1'](x)))
        x = self.relu(self.final_layer['bn2'](self.final_layer['conv2'](x)))
        
        # 输出形状: (batch_size, 50, 32, 80)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, dim=32, num_heads=4, ff_dim=128):
        super(TransformerBlock, self).__init__()
        # 多头注意力，4个头，Q/K/V维度为64
        self.attention = nn.MultiheadAttention(dim, num_heads)
        # 层归一化
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        # 前馈网络，维度为128
        self.ff = nn.Sequential(
            nn.Linear(dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, dim)
        )

    def forward(self, x):
        # 输入形状: [seq_len, batch_size, dim]
        # 多头自注意力
        attended, _ = self.attention(x, x, x)
        # 第一个残差连接和层归一化
        x = self.norm1(x + attended)
        # 前馈网络
        ff_output = self.ff(x)
        # 第二个残差连接和层归一化
        x = self.norm2(x + ff_output)
        return x

class SpatialEncoder(nn.Module):
    def __init__(self):
        super(SpatialEncoder, self).__init__()
        # 时序特征线性投影
        self.temporal_proj = nn.Linear(32, 32)
        # 3D位置信息线性嵌入
        self.position_proj = nn.Linear(3, 32)
        
        # 3个Transformer块
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(dim=32, num_heads=4, ff_dim=128)
            for _ in range(3)
        ])

    def forward(self, temporal_features, position_info):
        # temporal_features: [batch_size, 50, 32, 80]
        # position_info: [batch_size, 50, 3]
        batch_size = temporal_features.size(0)
        
        # 处理时序特征：平均池化时间维度
        temp_feat = temporal_features.mean(dim=-1)  # [batch_size, 50, 32]
        temp_feat = self.temporal_proj(temp_feat)
        
        # 处理位置信息
        pos_feat = self.position_proj(position_info)  # [batch_size, 50, 32]
        # print(pos_feat.shape)   



        # 特征融合
        x = temp_feat + pos_feat  # [batch_size, 50, 32]
        
        # 调整维度顺序以适应Transformer
        x = x.permute(1, 0, 2)  # [50, batch_size, 32]
        
        # 通过Transformer块
        for block in self.transformer_blocks:
            x = block(x)
            
        # 恢复原始维度顺序
        x = x.permute(1, 0, 2)  # [batch_size, 50, 32]
            
        return x  # [batch_size, 50, 32]


class FeatureExpander(nn.Module):
    def __init__(self):
        super(FeatureExpander, self).__init__()
        
        # 第一层 (32 -> 16) 80 -> 160
        self.conv_trans1 = CustomConv1DTranspose(
            padding=3,  # 调整padding
            input_channels=32,
            filters=16,
            kernel_size=8,  # 调整kernel_size
            strides=2
        )
        self.conv1_1 = CustomConv1D(16, 16, kernel_size=7, padding='same')
        self.conv1_2 = CustomConv1D(16, 16, kernel_size=7, padding='same')
        self.bn1 = CustomBatchNorm1d(16)
        
        # 第二层 (16 -> 8) 160 -> 320
        self.conv_trans2 = CustomConv1DTranspose(
            padding=3,
            input_channels=16,
            filters=8,
            kernel_size=8,
            strides=2
        )
        self.conv2_1 = CustomConv1D(8, 8, kernel_size=7, padding='same')
        self.conv2_2 = CustomConv1D(8, 8, kernel_size=7, padding='same')
        self.bn2 = CustomBatchNorm1d(8)
        
        # 第三层 (8 -> 4) 320 -> 640
        self.conv_trans3 = CustomConv1DTranspose(
            padding=3,
            input_channels=8,
            filters=4,
            kernel_size=8,
            strides=2
        )
        self.conv3_1 = CustomConv1D(4, 4, kernel_size=7, padding='same')
        self.conv3_2 = CustomConv1D(4, 4, kernel_size=7, padding='same')
        self.bn3 = CustomBatchNorm1d(4)
        
        # 第四层 (4 -> 4) 640 -> 640
        self.conv4 = CustomConv1D(4, 4, kernel_size=7, padding='same')
        self.conv4_1 = CustomConv1D(4, 4, kernel_size=7, padding='same')
        self.conv4_2 = CustomConv1D(4, 4, kernel_size=7, padding='same')
        self.bn4 = CustomBatchNorm1d(4)
        
        # 空间特征扩展
        self.spatial_proj = nn.Linear(32, 4 * 640)
        
        self.relu = nn.ReLU()
        
    def forward(self, temporal_features, spatial_features):
        batch_size = temporal_features.size(0)
        

        # print("Input temporal shape:", temporal_features.shape)
        
        # 第一层 80 -> 160
        x = self.conv_trans1(temporal_features)
        x = x[..., :160]  # 截断到正确的长度
        x = self.relu(self.bn1(self.conv1_1(x)))
        x = self.relu(self.bn1(self.conv1_2(x)))
        # print("After layer1:", x.shape)
        
        # 第二层 160 -> 320
        x = self.conv_trans2(x)
        x = x[..., :320]  # 截断到正确的长度
        x = self.relu(self.bn2(self.conv2_1(x)))
        x = self.relu(self.bn2(self.conv2_2(x)))
        # print("After layer2:", x.shape)
        
        # 第三层 320 -> 640
        x = self.conv_trans3(x)
        x = x[..., :640]  # 截断到正确的长度
        x = self.relu(self.bn3(self.conv3_1(x)))
        x = self.relu(self.bn3(self.conv3_2(x)))
        # print("After layer3:", x.shape)
        
        # 第四层 保持640
        x = self.conv4(x)
        x = self.relu(self.bn4(self.conv4_1(x)))
        x = self.relu(self.bn4(self.conv4_2(x)))
        # print("After layer4:", x.shape)
        
        # 空间特征扩展
        spatial_expanded = self.spatial_proj(spatial_features)
        spatial_expanded = spatial_expanded.view(batch_size, 50, 4, 640)
        # print("Spatial expanded shape:", spatial_expanded.shape)
        
        # 特征融合
        fused = x * spatial_expanded
        # print("After fusion:", fused.shape)
        
        # 调整维度顺序并压缩50维
        fused = fused.mean(dim=1)  # 在50维度上取平均
        # print("Final output:", fused.shape)
        
        return fused

import torch
import torch.nn as nn

class DilatedCausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super(DilatedCausalConv1d, self).__init__()
        self.padding = (kernel_size - 1) * dilation  # 因果卷积的padding
        self.conv = nn.Conv1d(
            in_channels, 
            out_channels,
            kernel_size,
            padding=self.padding,
            dilation=dilation
        )

    def forward(self, x):
        # 应用卷积
        out = self.conv(x)
        # 移除多余的padding (保持因果性)
        return out[:, :, :-self.padding]

class TCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super(TCNBlock, self).__init__()
        self.conv1 = DilatedCausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.conv2 = DilatedCausalConv1d(out_channels, out_channels, kernel_size, dilation)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        
        # 残差连接
        self.downsample = None
        if in_channels != out_channels:
            self.downsample = nn.Conv1d(in_channels, out_channels, 1)

    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            residual = self.downsample(x)
            
        # 确保残差连接的维度匹配
        if residual.size(-1) > out.size(-1):
            residual = residual[:, :, :out.size(-1)]
            
        out += residual
        out = self.relu(out)
        return out

class TCNDecoder(nn.Module):
    def __init__(self, input_channels=4, output_channels=1, num_layers=9, kernel_size=3):
        super(TCNDecoder, self).__init__()
        self.num_layers = num_layers
        channels = 64  # 可以根据需要调整通道数
        
        # 初始投影层
        self.input_proj = nn.Conv1d(input_channels, channels, 1)
        
        # TCN层
        self.tcn_layers = nn.ModuleList([
            TCNBlock(
                channels, 
                channels,
                kernel_size=kernel_size,
                dilation=2**i  # 膨胀因子: 1,2,4,8,...
            ) for i in range(num_layers)
        ])
        
        # 输出投影
        self.output_proj = nn.Conv1d(channels, output_channels, 1)
        
    def forward(self, x, future_steps=None):
        """
        Args:
            x: 输入张量 [batch_size, channels, time_steps]
            future_steps: 预测未来的步数(用于推理)
        """
        if self.training or future_steps is None:
            # 训练模式: 并行处理整个序列
            x = self.input_proj(x)
            for layer in self.tcn_layers:
                x = layer(x)
            return self.output_proj(x)
        else:
            # 推理模式: 自回归生成
            predictions = []
            current_input = x
            
            for _ in range(future_steps):
                # 通过网络前向传播
                h = self.input_proj(current_input)
                for layer in self.tcn_layers:
                    h = layer(h)
                next_pred = self.output_proj(h)
                
                # 只取最后一个时间步的预测
                next_pred = next_pred[:, :, -1:]
                predictions.append(next_pred)
                
                # 更新输入序列
                current_input = torch.cat([current_input[:, :, 1:], next_pred], dim=2)
            
            return torch.cat(predictions, dim=2)




class zkd_TSCN(nn.Module):
    def __init__(self):
        super(zkd_TSCN, self).__init__()
        self.temporal_encoder = TemporalEncoder()
        self.spatial_encoder = SpatialEncoder() 
        self.feature_expander = FeatureExpander()
        self.ecg_decoder = TCNDecoder()

    def forward(self, x, position_info):
        """
        Args:
            x: 输入雷达信号 [batch_size, num_signals, time_steps]
               转换为 [batch_size, num_signals, 1, time_steps]
            position_info: 位置信息 [batch_size, num_signals, 3]
        Returns:
            ecg: 重建的心电信号
        """
        # 时序特征提取
        # 确保输入维度正确 [batch_size, num_signals, time_steps] -> [batch_size, num_signals, 1, time_steps]
        if len(x.shape) == 3:
            x = x.unsqueeze(2)
        
        
        temporal_features = self.temporal_encoder(x)
        
        # 空间特征提取
        spatial_features = self.spatial_encoder(temporal_features, position_info)
        
        # 特征扩展
        expanded_features = self.feature_expander(temporal_features, spatial_features)
        
        # 解码生成ECG
        ecg = self.ecg_decoder(expanded_features)
        
        return ecg

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = zkd_TSCN().to(device)
    x = torch.randn(1, 50,640).to(device)  # 雷达信号输入
    position_info = torch.randn(1, 50, 3).to(device)  # 位置信息

    ecg = model(x, position_info)
    print(ecg.shape)



