"""
================================================================================
多任务预测头 — 赛题适配版
================================================================================
简化自 heterogeneous_hypergraph/classifier.py。

输入: FusionGate 输出的 h_fusion (N, 64)
输出: 两个预测概率
  - 信用风险预测（二分类）
  - 企业分级（五分类）

去掉了原版的风险预测头（原版 y_risk 依赖坏账比例，赛题数据无）。
================================================================================
"""

import torch
import torch.nn as nn

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import GRADE_CLASSES, DROPOUT


class MultiTaskHeads(nn.Module):
    """
    ==========================================================================
    两个并行的 MLP 预测头，支持分批推理防 OOM。
    ==========================================================================
    """
    def __init__(self, in_dim: int, hidden: int = 64, dropout: float = DROPOUT):
        super().__init__()
        self.default_batch = 4096  # 分批推理 chunk 大小

        # 信用风险预测头（二分类）
        self.credit_head = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

        # 企业分级头（五分类: S/A/B/C/D）
        self.grade_head = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, GRADE_CLASSES),
        )

    def forward(self, z_v: torch.Tensor, batch_size: int = None):
        """
        输入:  z_v (N, in_dim)
        输出:
          logit_credit: (N, 1)   信用风险 logits
          logit_grade:  (N, 5)   分级 logits
        """
        bs = batch_size if batch_size is not None else self.default_batch
        N = z_v.size(0)

        if N <= bs:
            return (
                self.credit_head(z_v),
                self.grade_head(z_v),
            )

        credit_out, grade_out = [], []
        for start in range(0, N, bs):
            z = z_v[start:start + bs]
            credit_out.append(self.credit_head(z))
            grade_out.append(self.grade_head(z))

        return (
            torch.cat(credit_out, dim=0),
            torch.cat(grade_out, dim=0),
        )
