"""
================================================================================
全局配置 — 赛题数据适配版（多视角超图 + 企业属性编码器）
================================================================================
基于 heterogeneous_hypergraph/config.py 重写。

与原版的关键区别：
  - 无时序编码（赛题数据无财务报表，去掉 FinGRU）
  - 无异构图通道（无 financial_state / lawsuit_type / scf_type 节点）
  - 新增 EnterpriseEncoder（MLP 编码企业静态属性，替代异构通道）
  - 超图视图扩展为 5 个：supply / industry / legal_risk / geographic / capital
  - 标签重构：从 征信排序值 构建信用风险标签
================================================================================
"""

import os

# ============================================================================
# 路径配置
# ============================================================================
DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "创新实践大赛", "研究生金融科技创新大赛配套材料", "10万笔数据交付"
)
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================================
# 模型超参数
# ============================================================================
HIDDEN_DIM = 128           # 统一隐藏维度
DROPOUT = 0.35

# ── 超图通道 ──
NUM_HYPER_VIEWS = 5        # 超图视图数: supply, industry, legal_risk, geographic, capital
HYPER_HIDDEN = 128         # 超图卷积隐藏维度
HYPER_LAYERS = 2           # 超图卷积层数
HYPER_AGGR = "attention"   # 视图融合方式: mean / attention

# ── 企业属性编码器（替代原异构通道） ──
ENT_FEAT_DIM = 24          # 企业静态特征维度
ENT_ENCODER_HIDDEN = 128   # MLP 隐藏维度
ENT_ENCODER_LAYERS = 2     # MLP 层数

# ── 融合门 ──
FUSION_HIDDEN = 64

# ── 分类头 ──
NUM_CLASSES = 2            # 信用风险二分类（好/坏）
GRADE_CLASSES = 5          # 企业分级五档

# ============================================================================
# 训练配置
# ============================================================================
SEED = 1234
DEVICE = "cuda"
EPOCHS = 500
LR = 1e-3
LR_HYPER = 3e-4            # 超图通道学习率（更保守）
WEIGHT_DECAY = 1.5e-3
EARLY_STOP_PATIENCE = 50

USE_AMP = True             # AMP 混合精度

BATCH_SIZE = 4096          # 预测头推理 batch（防 OOM）

# ── 损失权重 ──
LAMBDA_RISK = 0.3          # 风险辅助任务权重
LAMBDA_GRADE = 0.2         # 分级辅助任务权重
LAMBDA_STRUCT = 0.0        # 结构正则（暂关闭）

# ============================================================================
# 超图定义（5 个视图）
# ============================================================================
HYPER_CONFIG = {
    "supply": {
        "name": "供应链超图",
        "source_edges": ["trade"],
        "direction": "both",            # both: 上下游分开建超边
        "min_hyperedge_size": 3,
        "max_hyperedge_size": 500,
    },
    "industry": {
        "name": "行业超图",
        "source_edges": [],             # 从行业代码直接分组
        "direction": "undirected",
        "min_hyperedge_size": 5,
        "max_hyperedge_size": 2000,
    },
    "legal_risk": {
        "name": "司法风险关联超图",
        "source_edges": ["legal_risk"], # 共同涉诉 / 共同被执行
        "direction": "undirected",
        "min_hyperedge_size": 2,
        "max_hyperedge_size": 200,
    },
    "geographic": {
        "name": "地域超图",
        "source_edges": [],             # 从省份/城市直接分组
        "direction": "undirected",
        "min_hyperedge_size": 10,
        "max_hyperedge_size": 5000,
    },
    "capital": {
        "name": "注册资本超图",
        "source_edges": [],             # 从注册资本区间分组
        "direction": "undirected",
        "min_hyperedge_size": 5,
        "max_hyperedge_size": 3000,
    },
}

# ============================================================================
# 边类型定义（赛题数据：仅企业间同构边）
# ============================================================================
EDGE_TYPES = [
    ("enterprise", "trade",      "enterprise"),   # 招投标供应链边
    ("enterprise", "legal_risk", "enterprise"),   # 共涉诉关联边
]

EDGE_TYPE_NAMES = [et[1] for et in EDGE_TYPES]

# ============================================================================
# 节点类型（仅 enterprise 一种）
# ============================================================================
NODE_TYPES = ["enterprise"]

# ============================================================================
# 征信排序值 → 风险标签 映射配置
# ============================================================================
# 征信排序值越小越优（排名靠前=信用好），按分位数切为二分类标签
CREDIT_RANK_LABEL_QUANTILE = 0.3  # 前 30% 为白名单（好企业），后 70% 为风险企业

# ============================================================================
# 司法涉诉特征配置
# ============================================================================
JUDICIAL_DIR = DATA_DIR  # 司法涉诉 CSV 已平移到根目录
JUDICIAL_FILES = {
    "dishonest":     "失信被执行人.csv",
    "court_announce": "法院公告.csv",
    "executee":      "被执行人.csv",
    "consumption":   "限制高消费.csv",
}

# 司法严重程度权重
JUDICIAL_SEVERITY = {
    "dishonest":     5.0,   # 失信被执行人 — 最严重
    "executee":      3.0,   # 被执行人
    "consumption":   2.0,   # 限制高消费
    "court_announce": 1.0,  # 法院公告
}
