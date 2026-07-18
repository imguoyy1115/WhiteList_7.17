"""
================================================================================
全局配置 — 超图异构双通道模型 v5
================================================================================
与 model_v4/config.py 的关系：独立重新设计，保留部分兼容字段。
model_v4 代码保留在 ../model_v4/ 下作为参考。
================================================================================
"""

import os

# ============================================================================
# 路径配置
# ============================================================================
DATA_DIR = None
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 参考：model_v4 数据管线路径
MODEL_V4_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "model_v4")

# ============================================================================
# 模型超参数
# ============================================================================
HIDDEN_DIM = 128           # 统一隐藏维度
DROPOUT = 0.2

# ── 同构通道（超图） ──
NUM_HYPER_VIEWS = 4        # 超图视图数: supply, equity, legal_rep, industry
HYPER_HIDDEN = 128         # 超图卷积隐藏维度
HYPER_LAYERS = 2           # 超图卷积层数
HYPER_AGGR = "attention"   # 超边内聚合方式: mean / attention

# ── 异构通道（特征图） ──
NUM_FIN_STATES = 18        # 财务状态节点数（6指标 × 3档）
NUM_LAWSUIT_TYPES = 8      # 诉讼类型节点数
NUM_SCF_TYPES = 6          # SCF合约类型节点数
HETERO_LAYERS = 1          # 异构图卷积层数（特征查找节点 1 层足够，2 层过拟合）

# ── 时序编码器 ──
SEQ_LEN = 4                # 半年报时序步数（4 个半年 = 2 年）
GRU_HIDDEN = 32            # GRU 隐藏维度
GRU_LAYERS = 2

# ── 融合门 ──
FUSION_HIDDEN = 64

# ── 分类头 ──
NUM_CLASSES = 2
GRADE_CLASSES = 5
HEAD_BATCH_SIZE = 4096

# ============================================================================
# 训练配置
# ============================================================================
SEED = 42
DEVICE = "cuda"
EPOCHS = 500
LR = 1e-3
LR_HYPER = 3e-4            # 超图通道学习率（单独设置，更保守）
WEIGHT_DECAY = 5e-4
EARLY_STOP_PATIENCE = 80

USE_AMP = True             # AMP 混合精度（T4 开 AMP 加速 ~1.5-2x，省显存）

# ── 阶段性训练 ──
PHASE1_EPOCHS = 80         # Phase 1: 预热异构通道
PHASE2_EPOCHS = 150        # Phase 2: 加入超图通道
PHASE3_EPOCHS = 50         # Phase 3: 时序微调

BATCH_SIZE = 512

# ── 损失权重 ──
LAMBDA_RISK = 0.5
LAMBDA_GRADE = 0.3
LAMBDA_STRUCT = 0.05       # 超图结构一致性正则（同超边内预测平滑）

# ── 消融实验（默认全部关闭 = 完整模型） ──
ABLATION_NO_TEMPORAL = False      # True = 去掉 GRU 时序，退化为 MLP 投影
ABLATION_NO_FEATURE_SPLIT = False  # False = v5.2 特征分工（同构通道 SCF+诉讼 10 维 / 异构通道 财务 2 维）

# ── 特征分工（v5.2：同构通道看关系，异构通道看财务 + 财务时序GRU） ──
# 同构通道取 SCF(8) + 诉讼(2) = 10 维，排除财务特征以避免中小企业零值噪声
STRUCT_FEATURE_INDICES = [0, 1, 2, 3, 4, 5, 6, 7, 10, 11]
# 异构通道取营收增长率(8) + 资产周转率(9) = 2 维，聚焦经营表现
FINANCIAL_FEATURE_INDICES = [8, 9]
# col 12 为预留维度，暂归入同构通道（修改 STRUCT_FEATURE_INDICES 即可调整）

# x_seq 中对应 FINANCIAL_FEATURE_INDICES 的财务指标索引
# INDICATOR_ORDER: CR,DAR,ICR,ROA,ROE,ART,APT,TAGR,REVGR,CFONI,DFL,DOL
# FINANCIAL_FEATURE_INDICES[0]=8(营收增长率=REVGR) → x_seq idx 8
# FINANCIAL_FEATURE_INDICES[1]=9(资产周转率=ART)    → x_seq idx 5
FIN_SEQ_INDICES = [8, 5]

# ============================================================================
# 边类型定义（v5 扩展为 6 种）
# ============================================================================
EDGE_TYPES = [
    # 同构图边（Enterprise ↔ Enterprise）
    ("enterprise", "trade",            "enterprise"),
    ("enterprise", "equity",           "enterprise"),
    ("enterprise", "legal_rep",        "enterprise"),
    # 异构特征边（Enterprise → 特征节点）
    ("enterprise", "has_financial",    "financial_state"),
    ("enterprise", "has_lawsuit",      "lawsuit_type"),
    ("enterprise", "uses_scf",         "scf_type"),
]

EDGE_TYPE_NAMES = [et[1] for et in EDGE_TYPES]

# ── 节点类型 ──
NODE_TYPES = ["enterprise", "financial_state", "lawsuit_type", "scf_type"]

# ============================================================================
# 超图定义
# ============================================================================
HYPER_CONFIG = {
    "supply": {
        "name": "供应链超图",
        "source_edges": ["trade"],     # 从哪些边构建超边
        "direction": "both",           # both: 上游+下游分开建  /  undirected: 无向
        "min_hyperedge_size": 3,       # 超边最少节点数（太少没意义）
        "max_hyperedge_size": 500,     # 超边最多节点数（防爆炸）
    },
    "equity": {
        "name": "投资控制超图",
        "source_edges": ["equity"],
        "direction": "undirected",
        "min_hyperedge_size": 2,
        "max_hyperedge_size": 200,
    },
    "legal_rep": {
        "name": "法人关联超图",
        "source_edges": ["legal_rep"],
        "direction": "undirected",
        "min_hyperedge_size": 2,
        "max_hyperedge_size": 100,
    },
    "industry": {
        "name": "行业同构超图",
        "source_edges": [],            # 行业超边不从边表构建，从行业代码直接分组
        "direction": "undirected",
        "min_hyperedge_size": 5,
        "max_hyperedge_size": 1000,
    },
}
