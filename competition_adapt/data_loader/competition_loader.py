"""
================================================================================
赛题数据加载器 — 多视角超图 + 企业属性编码
================================================================================
从 10万笔数据交付 目录加载数据，构建 CompetitionGraphData。

数据管线:
  Step 1: 加载工商数据 → 企业基础信息 + 征信排序值（标签）
  Step 2: 加载司法涉诉 → 企业司法特征 + 共涉诉关联边
  Step 3: 加载招投标数据 → 供应链边
  Step 4: 构建企业静态特征（24 维）
  Step 5: 构建超图（5 视图）
  Step 6: 标签构建 + 数据切分 + 组装

用法:
  from data_loader.competition_loader import load_competition_data
  data = load_competition_data()
================================================================================
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import torch
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_interface import CompetitionGraphData
from config import (
    DATA_DIR, JUDICIAL_DIR, JUDICIAL_FILES, JUDICIAL_SEVERITY,
    CREDIT_RANK_LABEL_QUANTILE, SEED,
)

np.random.seed(SEED)


# ═══════════════════════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════════════════════
def _clean_name(name):
    """清洗企业名称：去空格、统一括号、去常见后缀干扰"""
    if pd.isna(name):
        return ""
    s = str(name).replace("（", "(").replace("）", ")")
    s = s.replace(" ", "").replace("*", "").replace("\t", "").strip()
    return s


def _safe_read_csv(filepath, encodings=("utf-8", "utf-8-sig", "gbk", "gb18030", "gb2312")):
    """尝试多种编码读取 CSV"""
    for enc in encodings:
        try:
            return pd.read_csv(filepath, encoding=enc, low_memory=False)
        except (UnicodeDecodeError, UnicodeError):
            continue
    raise ValueError(f"无法读取文件: {filepath}")


# ═══════════════════════════════════════════════════════════
# Step 1: 加载工商数据
# ═══════════════════════════════════════════════════════════
def load_enterprise_data():
    """
    加载工商数据.xlsx，提取企业基础信息。

    返回:
      df_ent: DataFrame, 索引为 enterprise_id
      credit_rank: (N,) numpy array, 征信排序值（越小越好）
    """
    print("[1/6] 加载工商数据...")
    path = os.path.join(DATA_DIR, "工商数据.xlsx")
    df = pd.read_excel(path)
    print(f"  原始记录: {len(df)} 条, {df.shape[1]} 列")

    # 去除 征信排序值 为空的企业
    df = df.dropna(subset=["征信排序值"])
    df = df.reset_index(drop=True)
    print(f"  有效记录（有征信排序值）: {len(df)} 条")

    credit_rank = df["征信排序值"].astype(float).values

    return df, credit_rank


# ═══════════════════════════════════════════════════════════
# Step 2: 加载司法涉诉数据 + 构建企业司法特征
# ═══════════════════════════════════════════════════════════
def load_judicial_data(df_ent, name_to_id):
    """
    加载 4 类司法涉诉 CSV，按企业名称匹配，统计每个企业的司法风险指标。

    返回:
      judicial_features: (N, 6)  numpy array
        列: 失信次数, 被执行次数, 限高次数, 公告次数,
             加权风险分, 是否涉诉(0/1)
      co_litigation_edges: set of (ent_id_a, ent_id_b)
        共同涉诉/被执行的企业对
    """
    print("\n[2/6] 加载司法涉诉数据...")

    N = len(df_ent)
    # 每种涉诉类型的次数
    count_dishonest = np.zeros(N, dtype=np.float32)
    count_executee = np.zeros(N, dtype=np.float32)
    count_consumption = np.zeros(N, dtype=np.float32)
    count_court = np.zeros(N, dtype=np.float32)

    # 共涉诉边：同一案件中涉及的企业对
    co_litigation_edges = set()

    for jud_type, fname in JUDICIAL_FILES.items():
        filepath = os.path.join(JUDICIAL_DIR, fname)
        if not os.path.exists(filepath):
            print(f"  [跳过] 文件不存在: {filepath}")
            continue

        df = _safe_read_csv(filepath)
        print(f"  {jud_type}: {len(df)} 条记录")

        # 查找企业名称列
        name_cols = ["企业名称", "当事人", "被执行人名称", "姓名",
                      "被申请人名称", "执行当事人名称", "公告当事人",
                      "CompanyName", "Name"]
        name_col = None
        for col in name_cols:
            if col in df.columns:
                name_col = col
                break
        if name_col is None:
            # 尝试模糊匹配
            for col in df.columns:
                if "名称" in col or "姓名" in col or "当事人" in col:
                    name_col = col
                    break
        if name_col is None:
            print(f"    [警告] 未找到企业名称列，跳过。列名: {list(df.columns[:10])}")
            continue

        # ── 频率统计 ──
        names = df[name_col].astype(str).apply(_clean_name).values
        matched_count = 0
        for nm in names:
            if nm in name_to_id:
                eid = name_to_id[nm]
                if jud_type == "dishonest":
                    count_dishonest[eid] += 1
                elif jud_type == "executee":
                    count_executee[eid] += 1
                elif jud_type == "consumption":
                    count_consumption[eid] += 1
                elif jud_type == "court_announce":
                    count_court[eid] += 1
                matched_count += 1
        print(f"    matched {matched_count} / {len(names)}")

        # ── 共涉诉边：按案件 ID 分组 ──
        case_cols = ["案号", "执行案号", "公告案号", "CaseNumber", "FenHao"]
        case_col = None
        for col in case_cols:
            if col in df.columns:
                case_col = col
                break
        if case_col is not None:
            case_groups = defaultdict(list)
            for i, nm in enumerate(names):
                if nm in name_to_id:
                    case_id = str(df.iloc[i][case_col])
                    if case_id and case_id != "nan":
                        case_groups[case_id].append(name_to_id[nm])
            for case_id, eids in case_groups.items():
                for i in range(len(eids)):
                    for j in range(i + 1, len(eids)):
                        a, b = eids[i], eids[j]
                        if a != b:
                            co_litigation_edges.add((min(a, b), max(a, b)))
        print(f"    共涉诉边累计: {len(co_litigation_edges)}")

    # ── 加权风险分 ──
    weighted_score = (
        JUDICIAL_SEVERITY["dishonest"]      * count_dishonest +
        JUDICIAL_SEVERITY["executee"]       * count_executee +
        JUDICIAL_SEVERITY["consumption"]    * count_consumption +
        JUDICIAL_SEVERITY["court_announce"] * count_court
    )

    # ── 是否涉诉标记 ──
    has_litigation = (
        (count_dishonest + count_executee + count_consumption + count_court) > 0
    ).astype(np.float32)

    judicial_features = np.stack([
        count_dishonest,
        count_executee,
        count_consumption,
        count_court,
        np.log1p(weighted_score),     # log1p 变换防长尾
        has_litigation,
    ], axis=1)

    print(f"  司法特征: {judicial_features.shape[1]} 维")
    print(f"  有涉诉记录企业: {has_litigation.sum():.0f} / {N}")
    print(f"  共涉诉边总数: {len(co_litigation_edges)}")

    return judicial_features, co_litigation_edges


# ═══════════════════════════════════════════════════════════
# Step 3: 加载招投标数据 → 供应链边
# ═══════════════════════════════════════════════════════════
def load_bidding_data(df_ent, name_to_id):
    """
    加载招投标数据.xlsx，构建供应商→采购人的供应链边。

    返回:
      trade_edges: set of (supplier_id, buyer_id)
    """
    print("\n[3/6] 加载招投标数据...")
    path = os.path.join(DATA_DIR, "招投标数据.xlsx")
    if not os.path.exists(path):
        print(f"  [警告] 文件不存在: {path}")
        return set()

    df = pd.read_excel(path)
    print(f"  原始记录: {len(df)} 条, {df.shape[1]} 列")

    trade_edges = set()

    # 查找供应商/采购人列
    supplier_col = None
    buyer_col = None
    for col in df.columns:
        if "供应商" in col or "投标人" in col or "Supplier" in col.lower():
            supplier_col = col
        if "采购" in col or "招标" in col or "Buyer" in col.lower() or "业主" in col:
            buyer_col = col

    if supplier_col is None or buyer_col is None:
        print(f"  [警告] 未找到供应商/采购人列。列名: {list(df.columns[:15])}")
        return trade_edges

    suppliers = df[supplier_col].astype(str).apply(_clean_name).values
    buyers = df[buyer_col].astype(str).apply(_clean_name).values

    matched_both = 0
    for s_name, b_name in zip(suppliers, buyers):
        s_id = name_to_id.get(s_name, -1)
        b_id = name_to_id.get(b_name, -1)
        if s_id >= 0 and b_id >= 0 and s_id != b_id:
            trade_edges.add((s_id, b_id))
            matched_both += 1

    print(f"  双向匹配成功: {matched_both} / {len(df)}")
    print(f"  供应链边总数: {len(trade_edges)}")

    return trade_edges


# ═══════════════════════════════════════════════════════════
# Step 4: 构建企业静态特征（24 维）
# ═══════════════════════════════════════════════════════════
def build_enterprise_features(df_ent, judicial_features):
    """
    ==========================================================================
    从工商数据 + 司法统计构建 24 维企业静态特征。

    特征设计（24 维）:
      col  0:  注册资本(log1p)
      col  1:  参保人数(log1p)
      col  2:  经营年限
      col  3:  行业编码(门类 → 数值)
      col  4:  行业编码(大类 → 数值)
      col  5:  省份编码 → 数值
      col  6:  城市编码 → 数值
      col  7:  注册资本区间分档 (1-5)
      col  8-13: 司法特征 (6 维, 来自 Step 2)
      col 14:  经营年限分档 (1-5)
      col 15:  社保密度(参保人数/注册资本 log1p 比)
      col 16-21: 司法特征归一化后 (6 维, 去量纲)
      col 22:  供应链参与度（有多少条边的度中心性，归一到 [0,1]）
      col 23:  预留位
    ==========================================================================
    """
    print("\n[4/6] 构建企业静态特征 (24 维)...")
    N = len(df_ent)
    DIM = 24
    X = np.zeros((N, DIM), dtype=np.float32)

    # ── 4.1 注册资本(log1p) ──
    if "注册资本" in df_ent.columns:
        reg_cap = pd.to_numeric(df_ent["注册资本"], errors="coerce").fillna(0).values
        X[:, 0] = np.log1p(np.clip(reg_cap, 0, 1e12))
    else:
        X[:, 0] = 0.0

    # ── 4.2 参保人数(log1p) ──
    if "参保人数" in df_ent.columns:
        insurance = pd.to_numeric(df_ent["参保人数"], errors="coerce").fillna(0).values
        X[:, 1] = np.log1p(np.clip(insurance, 0, 1e7))
    else:
        X[:, 1] = 0.0

    # ── 4.3 经营年限 ──
    if "成立日期" in df_ent.columns:
        establish = pd.to_datetime(df_ent["成立日期"], errors="coerce")
        ref_date = pd.Timestamp("2025-01-01")
        age_years = (ref_date - establish).dt.days / 365.25
        X[:, 2] = np.clip(age_years.fillna(0).values, 0, 100)
    else:
        X[:, 2] = 0.0

    # ── 4.4-4.5 行业编码 ──
    if "行业分类_门类" in df_ent.columns:
        ind_door = df_ent["行业分类_门类"].astype("category")
        X[:, 3] = ind_door.cat.codes.fillna(-1).values + 1  # 0 留给未知
    if "行业分类_大类" in df_ent.columns:
        ind_big = df_ent["行业分类_大类"].astype("category")
        X[:, 4] = ind_big.cat.codes.fillna(-1).values + 1

    # ── 4.5-4.6 地域编码 ──
    if "省份" in df_ent.columns:
        prov = df_ent["省份"].astype("category")
        X[:, 5] = prov.cat.codes.fillna(-1).values + 1
    if "城市" in df_ent.columns:
        city = df_ent["城市"].astype("category")
        X[:, 6] = city.cat.codes.fillna(-1).values + 1

    # ── 4.7 注册资本分档 ──
    reg_vals = X[:, 0]
    bins = np.percentile(reg_vals[reg_vals > 0], [20, 40, 60, 80])
    X[:, 7] = np.digitize(reg_vals, [0] + list(bins)).astype(np.float32)

    # ── 4.8-4.13 司法特征（原始） ──
    X[:, 8:14] = judicial_features

    # ── 4.14 经营年限分档 ──
    age_bins = np.percentile(X[:, 2][X[:, 2] > 0], [20, 40, 60, 80])
    X[:, 14] = np.digitize(X[:, 2], [0] + list(age_bins)).astype(np.float32)

    # ── 4.15 社保密度 ──
    ins_vals = X[:, 1]
    X[:, 15] = np.where(reg_vals > 0, ins_vals / (reg_vals + 1e-8), 0.0)

    # ── 4.16-4.21 司法特征（归一化） ──
    jud_raw = judicial_features
    for j in range(6):
        col_data = jud_raw[:, j]
        mx = col_data.max()
        if mx > 0:
            X[:, 16 + j] = col_data / mx

    # ── 4.22-23 供应链参与度 + 预留 ──
    # 供应链参与度在构建边后填充（Step 5.5）
    X[:, 22] = 0.0
    X[:, 23] = 0.0

    # ── 归一化 ──
    # 对连续特征做 Z-score（类别特征 col 3-6, 7, 14 除外）
    cat_cols = [3, 4, 5, 6, 7, 14, 23]
    cont_cols = [c for c in range(DIM) if c not in cat_cols]

    mask_real = (X[:, cont_cols].std(axis=1) > 1e-8)  # 有有效数据的企业
    if mask_real.sum() > 30:
        scaler = StandardScaler().fit(X[:, cont_cols][mask_real])
        X[:, cont_cols] = scaler.transform(X[:, cont_cols])

    # 填充 NaN（全零行归一化后出现）
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    print(f"  特征维度: {DIM}")
    print(f"  有效特征企业: {mask_real.sum()} / {N}")

    return X


# ═══════════════════════════════════════════════════════════
# Step 5: 构建超图（5 视图）
# ═══════════════════════════════════════════════════════════
def build_hyperedges(df_ent, edge_index_dict, N, co_litigation_edges):
    """
    ==========================================================================
    从企业属性 + 边构建 5 张超图。

    supply:      以每个企业为核心，其 trade 邻居 + 自身构成超边
    industry:    同一行业门类的企业 → 超边
    legal_risk:  共同涉诉的连通分量 → 超边
    geographic:  同一省份的企业 → 超边
    capital:     同一注册资本区间的企业 → 超边
    ==========================================================================
    """
    print("\n[5/6] 构建超图 (5 视图)...")
    hyperedges = {}

    # ── 5.1 supply 超图 ──
    trade_ei = edge_index_dict.get(("enterprise", "trade", "enterprise"))
    supply_he = []
    if trade_ei is not None:
        adj = defaultdict(set)
        src = trade_ei[0].numpy()
        dst = trade_ei[1].numpy()
        for s, d in zip(src, dst):
            adj[int(s)].add(int(d))
            adj[int(d)].add(int(s))
        for core, neighbors in adj.items():
            he_nodes = [core] + list(neighbors)
            if len(he_nodes) >= 3:
                supply_he.append(torch.tensor(he_nodes))
    hyperedges["supply"] = supply_he
    print(f"  supply: {len(supply_he)} 条超边")

    # ── 5.2 industry 超图 ──
    industry_he = []
    if "行业分类_门类" in df_ent.columns:
        ind_vals = df_ent["行业分类_门类"].astype(str).values
        ind_groups = defaultdict(list)
        for i, ind in enumerate(ind_vals):
            if ind and ind != "nan":
                ind_groups[ind].append(i)
        for ind, eids in ind_groups.items():
            if len(eids) >= 5:
                industry_he.append(torch.tensor(eids))
    hyperedges["industry"] = industry_he
    print(f"  industry: {len(industry_he)} 条超边 ({len(set(df_ent['行业分类_门类'].dropna()))} 个门类)")

    # ── 5.3 legal_risk 超图 ──
    legal_he = []
    if co_litigation_edges:
        # 并查集聚类
        parent = list(range(N))
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py
        for a, b in co_litigation_edges:
            if a < N and b < N:
                union(a, b)
        comps = defaultdict(list)
        for i in range(N):
            comps[find(i)].append(i)
        for comp_nodes in comps.values():
            if len(comp_nodes) >= 2:
                legal_he.append(torch.tensor(comp_nodes))
    hyperedges["legal_risk"] = legal_he
    print(f"  legal_risk: {len(legal_he)} 条超边")

    # ── 5.4 geographic 超图 ──
    geo_he = []
    if "省份" in df_ent.columns:
        prov_vals = df_ent["省份"].astype(str).values
        prov_groups = defaultdict(list)
        for i, prov in enumerate(prov_vals):
            if prov and prov != "nan":
                prov_groups[prov].append(i)
        for prov, eids in prov_groups.items():
            if len(eids) >= 10:
                geo_he.append(torch.tensor(eids))
    hyperedges["geographic"] = geo_he
    print(f"  geographic: {len(geo_he)} 条超边")

    # ── 5.5 capital 超图 ──
    capital_he = []
    if "注册资本" in df_ent.columns:
        cap_vals = pd.to_numeric(df_ent["注册资本"], errors="coerce").fillna(0).values
        cap_log = np.log1p(np.clip(cap_vals, 0, 1e12))
        # 按 log-注册资本 分位数分 6 档
        cap_bins = np.percentile(cap_log[cap_log > 0], [16.7, 33.3, 50, 66.7, 83.3])
        cap_tier = np.digitize(cap_log, [0] + list(cap_bins))
        for tier in range(7):
            eids = np.where(cap_tier == tier)[0]
            if len(eids) >= 5:
                capital_he.append(torch.tensor(eids.tolist()))
    hyperedges["capital"] = capital_he
    print(f"  capital: {len(capital_he)} 条超边")

    return hyperedges


# ═══════════════════════════════════════════════════════════
# Step 6: 标签构建 + 组装
# ═══════════════════════════════════════════════════════════
def build_labels_and_assemble(df_ent, credit_rank, X_ent, trade_edges,
                                co_litigation_edges, hyperedges, judicial_features):
    """
    ==========================================================================
    从征信排序值构建信用风险标签，组装 CompetitionGraphData。

    标签逻辑:
      - 征信排序值越小 → 信用越好
      - 前 CREDIT_RANK_LABEL_QUANTILE (30%) 的企业 → y_credit = 1（白名单）
      - 其余 → y_credit = 0（风险企业）
      - y_grade: 按征信排序值五等分
    ==========================================================================
    """
    print("\n[6/6] 标签构建 + 组装...")
    N = len(df_ent)

    # ── 6.1 信用风险标签 ──
    #     注意：部分 征信排序值可能为 0 或缺失，这些企业在加载时已过滤
    sorted_idx = np.argsort(credit_rank)  # 升序：排名靠前 = 好企业
    threshold_idx = int(N * CREDIT_RANK_LABEL_QUANTILE)
    good_ids = set(sorted_idx[:threshold_idx])

    y_credit = np.zeros(N, dtype=np.int64)
    for i in good_ids:
        y_credit[i] = 1

    # ── 6.2 企业分级标签（五档） ──
    n_per_grade = N // 5
    y_grade = np.full(N, 4, dtype=np.int64)
    for grade in range(5):
        start = grade * n_per_grade
        end = start + n_per_grade if grade < 4 else N
        for idx in sorted_idx[start:end]:
            y_grade[int(idx)] = grade

    print(f"  正样本（白名单，前{CREDIT_RANK_LABEL_QUANTILE*100:.0f}%）: {y_credit.sum():.0f} / {N}")
    print(f"  分级分布: ", {g: (y_grade == g).sum() for g in range(5)})

    # ── 6.3 分层随机切分 (70/15/15) ──
    all_idx = np.arange(N)
    train_idx, rest_idx = train_test_split(
        all_idx, test_size=0.3, stratify=y_credit, random_state=SEED)
    y_rest = y_credit[rest_idx]
    val_idx, test_idx = train_test_split(
        rest_idx, test_size=0.5, stratify=y_rest, random_state=SEED)

    train_mask = torch.zeros(N, dtype=bool)
    train_mask[train_idx] = True
    val_mask = torch.zeros(N, dtype=bool)
    val_mask[val_idx] = True
    test_mask = torch.zeros(N, dtype=bool)
    test_mask[test_idx] = True

    print(f"  训练/验证/测试: {train_idx.shape[0]}/{val_idx.shape[0]}/{test_idx.shape[0]}")

    # ── 6.4 构建边表 ──
    ei = {}
    if trade_edges:
        ei[("enterprise", "trade", "enterprise")] = torch.tensor(
            list(trade_edges)).long().t()
    if co_litigation_edges:
        leg_list = [(a, b) for a, b in co_litigation_edges if a < N and b < N]
        if leg_list:
            leg_t = torch.tensor(leg_list).long().t()
            # 双向边
            ei[("enterprise", "legal_risk", "enterprise")] = torch.cat([
                leg_t, leg_t.flip(0)
            ], dim=1)

    # ── 6.5 结构特征（8 维） ──
    struct_hint = np.zeros((N, 8), dtype=np.float32)

    # 度统计
    if trade_edges:
        trade_deg = np.zeros(N, dtype=np.float32)
        for a, b in trade_edges:
            if a < N:
                trade_deg[a] += 1
            if b < N:
                trade_deg[b] += 1
        struct_hint[:, 0] = trade_deg
        struct_hint[:, 6] = np.log1p(trade_deg)

    if co_litigation_edges:
        leg_deg = np.zeros(N, dtype=np.float32)
        for a, b in co_litigation_edges:
            if a < N:
                leg_deg[a] += 1
            if b < N:
                leg_deg[b] += 1
        struct_hint[:, 1] = leg_deg

    struct_hint[:, 2] = struct_hint[:, 0] + struct_hint[:, 1]  # 总度数

    # 是否涉诉
    struct_hint[:, 3] = judicial_features[:, 5]

    # 是否有供应链关系
    struct_hint[:, 4] = (struct_hint[:, 0] > 0).astype(np.float32)

    # 企业规模分档(注册资本)
    struct_hint[:, 5] = np.digitize(
        np.log1p(pd.to_numeric(df_ent["注册资本"], errors="coerce").fillna(0).values),
        [5, 10, 15, 20]
    ).astype(np.float32)

    # log1p 总度数
    struct_hint[:, 7] = np.log1p(struct_hint[:, 2])

    # 归一化
    for c in [0, 1, 2, 6, 7]:
        mx = struct_hint[:, c].max()
        if mx > 0:
            struct_hint[:, c] /= mx

    # ── 6.6 组装 ──
    data = CompetitionGraphData(
        x_ent=torch.tensor(X_ent),
        edge_index_dict=ei,
        hyperedges=hyperedges,
        y_credit=torch.tensor(y_credit),
        y_grade=torch.tensor(y_grade),
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        num_enterprises=N,
        total_nodes=N,
        struct_hint=torch.tensor(struct_hint),
        credit_rank_raw=torch.tensor(credit_rank),
    )

    data.summary()
    return data


# ═══════════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════════
def load_competition_data():
    """
    端到端赛题数据加载管线。
    """
    stages = []
    t0 = time.time()

    # ── Step 1: 工商数据 ──
    df_ent, credit_rank = load_enterprise_data()
    N = len(df_ent)

    # 企业名称 → ID 映射
    if "企业名称" in df_ent.columns:
        names = df_ent["企业名称"].astype(str).apply(_clean_name).values
    else:
        names = [f"ENT_{i}" for i in range(N)]
    name_to_id = {nm: i for i, nm in enumerate(names) if nm}
    print(f"  企业名称→ID 映射: {len(name_to_id)} 去重")

    stages.append(("工商数据", time.time() - t0))
    t0 = time.time()

    # ── Step 2: 司法数据 ──
    judicial_features, co_litigation_edges = load_judicial_data(df_ent, name_to_id)
    stages.append(("司法涉诉", time.time() - t0))
    t0 = time.time()

    # ── Step 3: 招投标数据 ──
    trade_edges = load_bidding_data(df_ent, name_to_id)
    stages.append(("招投标", time.time() - t0))
    t0 = time.time()

    # ── Step 4: 企业特征 ──
    X_ent = build_enterprise_features(df_ent, judicial_features)
    stages.append(("企业特征", time.time() - t0))
    t0 = time.time()

    # ── Step 5: 构建边表（预构建，用于超图） ──
    ei_temp = {}
    if trade_edges:
        ei_temp[("enterprise", "trade", "enterprise")] = torch.tensor(
            list(trade_edges)).long().t()
    if co_litigation_edges:
        leg_list = [(a, b) for a, b in co_litigation_edges if a < N and b < N]
        if leg_list:
            leg_t = torch.tensor(leg_list).long().t()
            ei_temp[("enterprise", "legal_risk", "enterprise")] = torch.cat([
                leg_t, leg_t.flip(0)
            ], dim=1)

    # ── 填充供应链参与度（根据实际边） ──
    if trade_edges:
        trade_deg = np.zeros(N, dtype=np.float32)
        for a, b in trade_edges:
            if a < N:
                trade_deg[a] += 1
            if b < N:
                trade_deg[b] += 1
        max_deg = trade_deg.max()
        if max_deg > 0:
            X_ent[:, 22] = trade_deg / max_deg

    # ── Step 6: 超图 ──
    hyperedges = build_hyperedges(df_ent, ei_temp, N, co_litigation_edges)
    stages.append(("超图", time.time() - t0))
    t0 = time.time()

    # ── Step 7: 标签 + 组装 ──
    data = build_labels_and_assemble(
        df_ent, credit_rank, X_ent, trade_edges,
        co_litigation_edges, hyperedges, judicial_features
    )
    stages.append(("组装", time.time() - t0))

    print("\n时序:")
    for name, sec in stages:
        print(f"  {name}: {sec:.1f}s")

    return data


if __name__ == "__main__":
    t0 = time.time()
    data = load_competition_data()
    print(f"\n总耗时: {time.time() - t0:.1f}s")
