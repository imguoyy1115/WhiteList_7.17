"""
================================================================================
数据接口定义 — 超图异构双通道模型 v5
================================================================================
扩展自 model_v4/data_interface.py，新增：
  - 超边列表（每个视图独立）
  - 特征节点（financial_state, lawsuit_type, scf_type）
  - 半年报时序快照（替代 v4 的 Q4 重复 x_seq）
================================================================================
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import torch


@dataclass
class HeteroGraphData:
    """
    ==========================================================================
    异构图 + 超图数据 — v5 模型唯一标准输入接口
    ==========================================================================
    """

    # ═══════════════════════════════════════════
    # 必填字段（无默认值）
    # ═══════════════════════════════════════════

    # ---- 节点特征（4 种节点类型） ----
    x_dict: Dict[str, torch.Tensor]
    # {"enterprise": (N_ent, 11),         ← 纯结构: SCF(8) + 诉讼(2) + 预留(1)
    #  "financial_state": (N_fin, 8),     ← 统计描述 embedding
    #  "lawsuit_type": (N_law, 8),        ← 诉讼类型 embedding
    #  "scf_type": (N_scf, 8)}            ← SCF 类型 embedding

    # ---- 边索引（6 种边类型） ----
    edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor]

    # ---- 标签（仅 enterprise 节点） ----
    y_white: torch.Tensor     # (N_ent,) 0/1
    y_risk: torch.Tensor      # (N_ent,) 0/1
    y_grade: torch.Tensor     # (N_ent,) 0-4

    # ---- Mask（与 model_v4 兼容） ----
    train_mask: torch.Tensor
    val_mask: torch.Tensor
    test_mask: torch.Tensor

    # ═══════════════════════════════════════════
    # 可选字段（有默认值，必须在必填字段之后）
    # ═══════════════════════════════════════════

    # ---- 超边（4 个视图） ----
    hyperedges: Dict[str, List[torch.Tensor]] = field(default_factory=dict)
    # {"supply": [tensor([A,B,C,D]), tensor([E,F,G]), ...],  每条超边是一个企业 ID 列表
    #  "equity": [...],
    #  "legal_rep": [...],
    #  "industry": [...]}

    # ---- 时序快照（4 个半年，真实时序） ----
    snapshots: List[Dict] = field(default_factory=list)
    # [{"period": "2023H2", "x_dict": ..., "edge_index_dict": ..., "hyperedges": ...},
    #  {"period": "2024H1", ...},
    #  {"period": "2024H2", ...},
    #  {"period": "2025H1", ...}]

    # ---- Adaptive Feature Gate 数据（保留 v4 兼容） ----
    x_struct: Optional[Dict[str, torch.Tensor]] = None
    x_missing: Optional[Dict[str, torch.Tensor]] = None
    struct_hint: Optional[Dict[str, torch.Tensor]] = None

    # ---- 半年报时序原始特征（替代 v4 的 x_seq） ----
    x_seq: Optional[torch.Tensor] = None  # (N_ent, 4, 13)  仅非财务特征

    # ---- SCF 打分卡风险分数（模型训练后传播用） ----
    risk_scf: Optional[torch.Tensor] = None  # (N_ent,) ∈ [0,1], 越高越危险

    # ---- 元信息 ----
    num_enterprises: int = 0
    num_listed: int = 0
    total_nodes: int = 0

    def to(self, device: str) -> "HeteroGraphData":
        """一键移到 GPU/CPU"""
        self.x_dict = {k: v.to(device) for k, v in self.x_dict.items()}
        self.edge_index_dict = {k: v.to(device) for k, v in self.edge_index_dict.items()}
        # 超边保持 CPU（不需要 GPU 矩阵运算）
        self.y_white = self.y_white.to(device)
        self.y_risk = self.y_risk.to(device)
        self.y_grade = self.y_grade.to(device)
        self.train_mask = self.train_mask.to(device)
        self.val_mask = self.val_mask.to(device)
        self.test_mask = self.test_mask.to(device)
        if self.x_struct:
            self.x_struct = {k: v.to(device) for k, v in self.x_struct.items()}
        if self.x_missing:
            self.x_missing = {k: v.to(device) for k, v in self.x_missing.items()}
        if self.struct_hint:
            self.struct_hint = {k: v.to(device) for k, v in self.struct_hint.items()}
        if self.x_seq is not None:
            self.x_seq = self.x_seq.to(device)
        if self.risk_scf is not None:
            self.risk_scf = self.risk_scf.to(device)
        return self

    def summary(self):
        """打印数据集概览"""
        print("=" * 60)
        print("  超图异构数据概览 (v5)")
        print("=" * 60)
        for ntype, x in self.x_dict.items():
            print(f"  {ntype}: {x.shape[0]} 节点, {x.shape[1]} 维特征")
        for etype, ei in self.edge_index_dict.items():
            print(f"  {etype[0]} --{etype[1]}--> {etype[2]}: {ei.shape[1]} 条边")
        for view_name, he_list in self.hyperedges.items():
            sizes = [len(he) for he in he_list]
            if sizes:
                print(f"  超图 {view_name}: {len(he_list)} 条超边, "
                      f"平均大小 {sum(sizes)/len(sizes):.1f}, 最大 {max(sizes)}")
        print(f"  白名单正样本: {(self.y_white==1).sum().item()}")
        print(f"  训练/验证/测试: {self.train_mask.sum().item()}/"
              f"{self.val_mask.sum().item()}/{self.test_mask.sum().item()}")
        if self.snapshots:
            print(f"  时序快照: {len(self.snapshots)} 个半年期 "
                  f"({self.snapshots[0]['period']} ~ {self.snapshots[-1]['period']})")
