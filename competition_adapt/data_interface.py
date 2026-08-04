"""
================================================================================
数据接口定义 — 赛题数据适配版
================================================================================
简化自 heterogeneous_hypergraph/data_interface.py。

与原版的关键区别：
  - 仅 enterprise 一种节点类型，无 heterogeneous 特征节点
  - 超边 5 视图（supply / industry / legal_risk / geographic / capital）
  - 无时序快照（snapshots）和 x_seq
  - 保留 risk_propagation 所需字段
================================================================================
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import torch


@dataclass
class CompetitionGraphData:
    """
    ==========================================================================
    赛题数据接口 — 多视角超图 + 企业属性
    ==========================================================================
    """

    # ═══════════════════════════════════════════
    # 必填字段
    # ═══════════════════════════════════════════

    # ---- 节点特征（仅 enterprise） ----
    x_ent: torch.Tensor
    # (N, 24) 企业静态属性：注册资本、参保人数、经营年限、司法统计等

    # ---- 边索引（2 种） ----
    edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor]
    # {("enterprise", "trade",      "enterprise"): (2, E_trade),
    #  ("enterprise", "legal_risk", "enterprise"): (2, E_legal)}

    # ---- 标签（仅 enterprise 节点） ----
    y_credit: torch.Tensor   # (N,) 0/1  信用风险标签（从征信排序值构建）
    y_grade: torch.Tensor    # (N,) 0-4  企业分级标签

    # ---- Mask ----
    train_mask: torch.Tensor
    val_mask: torch.Tensor
    test_mask: torch.Tensor

    # ═══════════════════════════════════════════
    # 可选字段
    # ═══════════════════════════════════════════

    # ---- 超边（5 视图） ----
    hyperedges: Dict[str, List[torch.Tensor]] = field(default_factory=dict)
    # {"supply": [tensor([A,B,C]), ...],
    #  "industry": [...], "legal_risk": [...],
    #  "geographic": [...], "capital": [...]}

    # ---- 结构特征（用于融合门输入） ----
    struct_hint: Optional[torch.Tensor] = None  # (N, 8)

    # ---- 元信息 ----
    num_enterprises: int = 0
    total_nodes: int = 0

    # ---- 企业元数据（用于结果分析） ----
    enterprise_names: Optional[List[str]] = None
    credit_rank_raw: Optional[torch.Tensor] = None  # (N,) 原始征信排序值

    def to(self, device: str) -> "CompetitionGraphData":
        """一键移到 GPU/CPU"""
        self.x_ent = self.x_ent.to(device)
        self.edge_index_dict = {
            k: v.to(device) for k, v in self.edge_index_dict.items()
        }
        self.y_credit = self.y_credit.to(device)
        self.y_grade = self.y_grade.to(device)
        self.train_mask = self.train_mask.to(device)
        self.val_mask = self.val_mask.to(device)
        self.test_mask = self.test_mask.to(device)
        if self.struct_hint is not None:
            self.struct_hint = self.struct_hint.to(device)
        if self.credit_rank_raw is not None:
            self.credit_rank_raw = self.credit_rank_raw.to(device)
        return self

    def summary(self):
        """打印数据集概览"""
        print("=" * 60)
        print("  赛题数据概览 (competition_adapt)")
        print("=" * 60)
        print(f"  enterprise: {self.x_ent.shape[0]} 节点, "
              f"{self.x_ent.shape[1]} 维特征")
        for etype, ei in self.edge_index_dict.items():
            print(f"  {etype[0]} --{etype[1]}--> {etype[2]}: {ei.shape[1]} 条边")
        for view_name, he_list in self.hyperedges.items():
            sizes = [len(he) for he in he_list]
            if sizes:
                print(f"  超图 {view_name}: {len(he_list)} 条超边, "
                      f"平均大小 {sum(sizes)/len(sizes):.1f}, 最大 {max(sizes)}")
        print(f"  信用风险正样本: {(self.y_credit == 1).sum().item()}")
        print(f"  训练/验证/测试: {self.train_mask.sum().item()}/"
              f"{self.val_mask.sum().item()}/{self.test_mask.sum().item()}")
