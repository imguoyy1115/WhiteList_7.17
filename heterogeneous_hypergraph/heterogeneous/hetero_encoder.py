"""
================================================================================
异构通道编码器（Heterogeneous Channel Encoder）
================================================================================
在包含 4 种节点类型（enterprise / financial_state / lawsuit_type / scf_type）
的异构图上做 HeteroConv 消息传递，使用 GATv2 注意力聚合替代等权平均。

GATv2 相比 SAGEConv 的关键差异:
  - SAGEConv: 所有邻居等权平均，信息密度高的邻居被稀释
  - GATv2Conv: 每个节点学习关注哪些邻居，有财务数据的大企业自动获得更高权重
  - 动态注意力（GATv2 修复了 GAT 的静态注意力问题，query/key 先变换再打分）

企业与财务/诉讼/SCF 节点的边让 GNN 学到：
  - "有高盈利能力的企业的交易对手，信用更可靠"
  - "大量涉诉企业的关键供应商，风险更高"
  - "使用应收账款保理的企业的供应商，更稳定"

输入: x_dict (4 种节点) + edge_index_dict (6 种边)
输出: h_feat (N_ent, hidden)  特征感知的企业表示
================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, HeteroConv, Linear
from config import HIDDEN_DIM, DROPOUT, NODE_TYPES, HETERO_LAYERS, GAT_HEADS


class HeteroChannelEncoder(nn.Module):
    """
    ==========================================================================
    异构通道：4 种节点 + 6 种边的 HeteroConv 编码器

    与 v4 DualOutputEncoder 的区别：
      - 输入节点类型从 1 种变为 4 种
      - 去掉了 FeatureGate（FeatureGate 在异构通道之前独立工作）
      - 只输出 enterprise 节点的表示 h_feat
    ==========================================================================
    """
    def __init__(self, in_dims: dict, edge_types: list,
                 hidden: int = HIDDEN_DIM, num_layers: int = HETERO_LAYERS,
                 dropout: float = DROPOUT):
        super().__init__()
        self.hidden = hidden
        self.edge_types = edge_types

        # ── 节点投影：每种节点类型 → hidden ──
        self.proj = nn.ModuleDict()
        for ntype in NODE_TYPES:
            dim = in_dims.get(ntype, 8)  # 默认 8 维（特征节点）
            if dim is not None and dim > 0:
                self.proj[ntype] = Linear(dim, hidden)

        # ── HeteroConv 层（GATv2 注意力替换 SAGEConv 等权聚合） ──
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv_dict = {}
            for etype in edge_types:
                src, rel, dst = etype
                if src in self.proj and dst in self.proj:
                    # GATv2: 每头 hidden // heads 维，concat 后 = hidden，残差维度一致
                    conv_dict[etype] = GATv2Conv(
                        hidden, hidden // GAT_HEADS,
                        heads=GAT_HEADS, concat=True,
                        dropout=dropout,
                    )
            self.convs.append(HeteroConv(conv_dict, aggr="mean"))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x_dict: dict, edge_index_dict: dict) -> torch.Tensor:
        """
        ==========================================================================
        输入:
          x_dict: {ntype: (N_t, d_in)}  4 种节点的原始特征
          edge_index_dict: {(src, rel, dst): [2, E]}

        返回:
          h_feat: (N_ent, hidden)  仅 enterprise 节点的特征感知表示
        ==========================================================================
        """
        # ── 投影 ──
        h = {}
        for ntype in self.proj:
            if ntype in x_dict:
                h[ntype] = self.proj[ntype](x_dict[ntype])

        # ── 多层消息传递 ──
        for conv in self.convs:
            h_new = conv(h, edge_index_dict)
            h_new = {k: self.dropout(F.relu(v)) for k, v in h_new.items()}
            # 残差（仅对已有表示的节点类型）
            for ntype in h_new:
                if ntype in h:
                    h[ntype] = h_new[ntype] + h[ntype]
                else:
                    h[ntype] = h_new[ntype]

        return h.get("enterprise", None)
