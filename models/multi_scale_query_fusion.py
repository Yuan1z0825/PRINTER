import torch
import torch.nn as nn
import torch.nn.functional as F
from .position_encoding import PositionEmbeddingSine  # 确保已定义


class MultiScaleQueryFusionModule(nn.Module):
    def __init__(self, hidden_dim, num_queries, num_heads=8, num_layers=6, dropout=0.1):
        super(MultiScaleQueryFusionModule, self).__init__()
        self.num_queries = num_queries
        self.hidden_dim = hidden_dim

        # 可学习的查询向量
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        # Transformer 解码器层
        decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=num_heads,
                                                   dim_feedforward=hidden_dim * 4, dropout=dropout)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # 位置编码
        self.position_encoding = PositionEmbeddingSine(hidden_dim // 2, normalize=True)

        # 卷积层用于将融合后的查询转换为与生成器特征匹配的形状
        self.fusion_conv = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1)
        self.norm = nn.LayerNorm(hidden_dim)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, multi_scale_features, feature_sizes):
        """
        multi_scale_features: list of [B, C, H_i, W_i]，目标图像的多尺度特征图
        feature_sizes: list of tuples [(H_i, W_i), ...]，对应特征图的空间尺寸
        返回：
            fused_queries: [B, num_queries, C, H, W]，用于上采样的融合查询
            attn_weights: list of attention weights for visualization
        """
        B = multi_scale_features[0].size(0)
        device = multi_scale_features[0].device

        # 生成位置编码
        pos_encodings = [self.position_encoding(f).flatten(2).permute(2, 0, 1) for f in multi_scale_features]

        # 将多尺度特征拼接
        memory = torch.cat([f.flatten(2).permute(2, 0, 1) for f in multi_scale_features], dim=0)  # [sum(H_i*W_i), B, C]
        pos_memory = torch.cat(pos_encodings, dim=0)  # [sum(H_i*W_i), B, C]

        # 初始化查询向量
        queries = self.query_embed.weight.unsqueeze(1).repeat(1, B, 1)  # [num_queries, B, C]

        # Transformer 解码器
        decoder_output = self.transformer_decoder(tgt=queries, memory=memory, memory_key_padding_mask=None,
                                                  pos=pos_memory)  # [num_queries, B, C]

        # 将查询输出转换为与生成器特征匹配的形状
        # 假设最高层的空间尺寸用于上采样指导
        # 可以根据需求调整，这里假设使用所有尺度的融合查询
        fused_queries = decoder_output.permute(1, 0, 2).unsqueeze(-1).unsqueeze(-1)  # [B, num_queries, C, 1, 1]
        fused_queries = self.fusion_conv(
            fused_queries.view(B, self.num_queries * self.hidden_dim, 1, 1))  # [B, C, 1, 1]
        fused_queries = self.norm(fused_queries.view(B, self.hidden_dim, 1, 1)).unsqueeze(2)  # [B, C, 1, 1, 1]
        fused_queries = self.activation(fused_queries)

        # 返回融合后的查询，用于上采样过程
        return fused_queries  # [B, C, 1, 1, 1], 需根据上采样模块调整
