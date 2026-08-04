"""
================================================================================
双通道融合门（Fusion Gate）— 赛题适配版
================================================================================
从 heterogeneous_hypergraph/fusion/fusion_gate.py 重写。

与原版的关键区别:
  - 原版: 融合 超图结构(h_struct) + 异构特征(h_feat)
  - 新版: 融合 超图结构(h_struct) + 企业属性(h_attr)

门控语义:
  gate → 1.0: 该企业的预测更依赖超图结构位置（供应链网络位置、行业集群）
  gate → 0.0: 该企业的预测更依赖自身属性（注册资本、司法记录、经营年限等）
================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import FUSION_HIDDEN, DROPOUT


class FusionGate(nn.Module):
    """
    ==========================================================================
    双通道注意力融合: 超图结构 ⊗ 企业属性

    输入:
      h_struct: (N, D_struct)  超图通道输出的结构角色表示
      h_attr:   (N, D_attr)    企业属性编码器的输出
      struct_hint: (N, S)      图结构统计量（度中心性、是否涉诉等）

    输出:
      h_fusion: (N, hidden)    融合后的统一表示
    ==========================================================================
    """
    def __init__(self, struct_dim: int, attr_dim: int,
                 hidden: int = FUSION_HIDDEN, hint_dim: int = 8,
                 dropout: float = DROPOUT):
        super().__init__()

        # 维度对齐
        self.struct_proj = nn.Linear(struct_dim, hidden) if struct_dim != hidden else nn.Identity()
        self.attr_proj = nn.Linear(attr_dim, hidden) if attr_dim != hidden else nn.Identity()

        # 门控网络: [h_struct || h_attr || struct_hint] → gate
        gate_in = hidden * 2 + hint_dim
        self.gate_net = nn.Sequential(
            nn.Linear(gate_in, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
            nn.Sigmoid(),
        )

        # 融合后投影
        self.fusion_proj = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, h_struct: torch.Tensor, h_attr: torch.Tensor,
                struct_hint: torch.Tensor = None) -> torch.Tensor:
        """
        h_struct: (N, D_struct)
        h_attr:   (N, D_attr)
        struct_hint: (N, S) or None

        返回: h_fusion (N, hidden)
        """
        hs = self.struct_proj(h_struct)
        ha = self.attr_proj(h_attr)

        # 构建门控输入
        if struct_hint is not None:
            gate_in = torch.cat([hs, ha, struct_hint], dim=-1)
        else:
            gate_in = torch.cat([hs, ha], dim=-1)

        gate = self.gate_net(gate_in)  # (N, 1)

        # 融合
        h_fused = gate * hs + (1.0 - gate) * ha
        h_fusion = self.fusion_proj(h_fused)
        h_fusion = self.dropout(h_fusion)

        return h_fusion
