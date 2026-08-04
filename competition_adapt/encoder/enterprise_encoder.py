"""
================================================================================
企业属性编码器（Enterprise MLP Encoder）
================================================================================
替代原版的 FinTemporalEncoder + HeteroChannelEncoder。

在原版中，异构通道通过 GATv2Conv 在 4 种节点类型间传递消息实现特征感知。
赛题数据无 financial_state / lawsuit_type / scf_type 节点，无法构建异构图，
因此改为纯 MLP 对企业静态特征做非线性编码。

设计思路:
  - 企业 24 维静态特征 → 2 层 MLP + LayerNorm → h_attr (N, 128)
  - 输出维度与超图通道对齐，可以直接送入 FusionGate

与原版异构通道的等效关系:
  - 原版: 财务指标 → FeatureGate → HeteroConv(GATv2) → h_feat
  - 新版: 企业属性(注册资本+社保+年限+司法统计+行业+地域) → MLP → h_attr
================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ENT_ENCODER_HIDDEN, ENT_ENCODER_LAYERS, DROPOUT, HIDDEN_DIM


class EnterpriseEncoder(nn.Module):
    """
    ==========================================================================
    企业静态属性编码器。

    输入:  (N, ENT_FEAT_DIM)  企业 24 维静态特征
    输出:  (N, HIDDEN_DIM)    属性感知的企业表示
    ==========================================================================
    """
    def __init__(self, in_dim: int, hidden: int = ENT_ENCODER_HIDDEN,
                 out_dim: int = HIDDEN_DIM, num_layers: int = ENT_ENCODER_LAYERS,
                 dropout: float = DROPOUT):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        layers = []
        current_dim = in_dim
        for i in range(num_layers):
            layers.append(nn.Linear(current_dim, hidden))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden))
            layers.append(nn.Dropout(dropout))
            current_dim = hidden

        # 最后一层 → out_dim
        layers.append(nn.Linear(current_dim, out_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x_ent: torch.Tensor) -> torch.Tensor:
        """
        x_ent: (N, in_dim)  企业静态特征
        返回:  (N, out_dim)  属性编码
        """
        return self.mlp(x_ent)
