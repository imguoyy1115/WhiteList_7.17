"""
================================================================================
CSMAR 数据加载器 v5 — 超图异构双通道
================================================================================
基于 model_v4/data_loader/csmar_loader.py 重写。
CSV 加载 + 实体发现逻辑内嵌（避免跨版本 import 冲突），
新增：
  1. 财务状态节点（FinancialState）— 离散化为高/中/低三档
  2. 诉讼类型节点（LawsuitType）— 按案由聚类
  3. SCF 类型节点（SCFType）— 按产品类型
  4. 四张超图（supply / equity / legal_rep / industry）
  5. 半年报时序快照（4 步真实序列）

Enterprise 特征从 25 维压缩到 13 维（财务指标移到特征节点）。
================================================================================
"""
import os, sys, time
import numpy as np
import pandas as pd
import torch
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# ── 导入 v5 数据接口 ──
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_interface import HeteroGraphData

CSMAR_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "csmar-7.16")
SEED = 42
np.random.seed(SEED)

# ═══════════════════════════════════════════════════════════
# 过滤配置
# ═══════════════════════════════════════════════════════════
EXCLUDE_FINANCE = True
MIN_TRADE_DATE = "2022-01-01"
MIN_LAWSUIT_DATE = "2020-01-01"

# ── helpers（从 model_v4 复制，避免 import 冲突） ──
def _clean(s):
    return str(s).replace("（","(").replace("）",")").replace(" ","").replace("*","").strip()

def _read(folder, fname):
    p = os.path.join(CSMAR_DIR, folder, fname)
    return pd.read_csv(p, encoding="utf-8-sig", low_memory=False, on_bad_lines="skip")

def _stock(s):
    s = str(s).strip()
    return s.zfill(6) if s.replace('.','').isdigit() else s

_ent_cache = {}
def ent_id(code_or_name, E):
    key = str(code_or_name)
    if key in _ent_cache:
        return _ent_cache[key]
    code = _stock(key) if key[:1].isdigit() else ""
    result = -1
    if code in E["code2id"]:
        result = E["code2id"][code]
    else:
        nm = _clean(key)
        if nm in E["cp_matched"]:
            result = E["code2id"][E["cp_matched"][nm]]
        elif nm in E["s2id"]:
            result = E["s2id"][nm]
    _ent_cache[key] = result
    return result


# ═══════════════════════════════════════════════════════════
# Step 1: Load（从 model_v4 复制，避免跨版本 import 冲突）
# ═══════════════════════════════════════════════════════════
def load_all():
    print("[1/5] Loading CSVs...")
    T = {}
    T["solvency"]   = _read("偿债能力133357457","FI_T1.csv")
    T["profit"]     = _read("盈利能力134632506","FI_T5.csv")
    T["operation"]  = _read("经营能力135259886","FI_T4.csv")
    T["growth"]     = _read("发展能力135640386","FI_T8.csv")
    T["cashflow"]   = _read("现金流分析140224916","FI_T6.csv")
    T["risklevel"]  = _read("风险水平140608073","FI_T7.csv")
    T["shareholder"]= _read("十大股东文件144239277","HLD_Shareholders.csv")
    T["controller"] = _read("上市公司控制人文件144846229","HLD_Contrshr.csv")
    T["lawsuit"]    = _read("诉讼仲裁明细表145241458","LA_DETAIL.csv")
    T["scf_ov"]     = _read("供应链金融总体情况表143535774","SCF_Overview.csv")
    # T["scf_stats"]  = _read("供应链金融业务统计表163124673","SCF_BusinessStats.csv")  # 未下载
    T["scf_credit"] = _read("商业信用及话语权表141420435","SCF_TradeCreditPower.csv")
    T["scf_ar"]     = _read("应收账款主要欠款人信息表142009823","SCF_ARMajorDebtors.csv")
    T["scf_ap"]     = _read("预付账款主要欠款人信息表143057203","SCF_APMajorCreditors.csv")
    for k,v in T.items(): print(f"  {k}: {len(v)} rows")
    return T


# ═══════════════════════════════════════════════════════════
# Step 2: Entity discovery（从 model_v4 复制）
# ═══════════════════════════════════════════════════════════
def build_entities(T):
    print("\n[2/5] Entity discovery...")

    finance_codes = set()
    if EXCLUDE_FINANCE:
        for dfk in ["scf_ov", "scf_credit"]:
            df = T[dfk]
            if "IndustryCode" not in df.columns or "Symbol" not in df.columns:
                continue
            mask_fin = df["IndustryCode"].astype(str).str.strip().str.startswith("J")
            codes = df.loc[mask_fin, "Symbol"].astype(str).str.strip().str.zfill(6)
            finance_codes.update(codes[codes.str.isdigit()].values)
        print(f"  金融业企业: {len(finance_codes)}")

    st_codes = set()
    for dfk in ["scf_credit", "scf_ar", "scf_ap"]:
        df = T[dfk]
        if "StateTypeCode" not in df.columns or "Symbol" not in df.columns:
            continue
        mask_st = df["StateTypeCode"].astype(str).str.strip() != "1"
        codes = df.loc[mask_st, "Symbol"].astype(str).str.strip().str.zfill(6)
        st_codes.update(codes[codes.str.isdigit()].values)
    print(f"  ST/异常状态企业: {len(st_codes)}")

    listed = set()
    for k in ["solvency","profit","operation","growth","cashflow","risklevel"]:
        codes = T[k]["Stkcd"].dropna().astype(str).str.strip().str.zfill(6)
        listed.update(codes[codes.str.isdigit()].values)
    if EXCLUDE_FINANCE:
        listed -= finance_codes
    print(f"  Listed: {len(listed)} (剔除金融业 {len(finance_codes)} 家)")

    name2code = {}
    pairs = [("scf_ov","Symbol","ShortName"),("scf_credit","Symbol","ShortName"),
             ("shareholder","Stkcd","S0301a"),("controller","Stkcd","S0701b")]  # 2026-07-16 修正：S0701a(依据代码)→S0701b(实际控制人名称)
    for dfk, code_col, name_col in pairs:
        df = T[dfk]
        if code_col not in df.columns or name_col not in df.columns:
            continue
        codes = df[code_col].dropna().astype(str).str.strip().str.zfill(6)
        names = (df[name_col].dropna().astype(str)
                 .str.replace("（","(",regex=False)
                 .str.replace("）",")",regex=False)
                 .str.replace(" ","",regex=False)
                 .str.replace("*","",regex=False).str.strip())
        mask = codes.str.isdigit() & (names.str.len() > 0)
        for c,n in zip(codes[mask].values, names[mask].values):
            name2code[n] = c
    print(f"  name2code: {len(name2code)} entries")

    cp_seen = set()
    for dfk in ["scf_ar","scf_ap"]:
        df = T[dfk]
        nms = (df["DebtName"].dropna().astype(str)
               .str.replace("（","(",regex=False)
               .str.replace("）",")",regex=False)
               .str.replace(" ","",regex=False)
               .str.replace("*","",regex=False).str.strip())
        cp_seen.update(nms[nms.str.len() > 1].values)
    print(f"  CP names (unique): {len(cp_seen)}")

    cp_matched, cp_standalone = {}, set()
    for nm in cp_seen:
        if nm in name2code and name2code[nm] in listed:
            cp_matched[nm] = name2code[nm]
        else:
            cp_standalone.add(nm)
    print(f"  Matched: {len(cp_matched)}, Standalone: {len(cp_standalone)}")

    sh_df = T["shareholder"]
    sh_names_raw = sh_df["S0301a"].dropna().astype(str)
    sh_names_clean = (sh_names_raw
        .str.replace("（","(",regex=False)
        .str.replace("）",")",regex=False)
        .str.replace(" ","",regex=False)
        .str.replace("*","",regex=False).str.strip())
    existing = cp_seen | set(cp_matched.keys()) | cp_standalone | set(name2code.keys())
    sh_new = 0
    for nm in sh_names_clean.unique():
        nm_str = str(nm)
        if len(nm_str) <= 1 or nm_str in existing:
            continue
        cp_standalone.add(nm_str)
        sh_new += 1
    print(f"  Shareholder names added: {sh_new}")

    all_codes = sorted(listed | set(cp_matched.values()))
    code2id = {c:i for i,c in enumerate(all_codes)}
    s2id = {nm: len(all_codes)+i for i,nm in enumerate(sorted(cp_standalone))}
    n_total = len(all_codes) + len(cp_standalone)

    industry_codes = {}
    for dfk in ["scf_ov", "scf_credit"]:
        df = T[dfk]
        if "IndustryCode" not in df.columns or "Symbol" not in df.columns:
            continue
        codes = df["Symbol"].astype(str).str.strip().str.zfill(6)
        inds = df["IndustryCode"].astype(str).str.strip()
        for c, ind in zip(codes, inds):
            if c in code2id:
                eid = code2id[c]
                if eid not in industry_codes or len(ind) > len(str(industry_codes.get(eid, ""))):
                    industry_codes[eid] = ind

    print(f"  Total nodes: {n_total}")

    return {
        "n_total": n_total, "n_listed": len(listed),
        "code2id": code2id, "s2id": s2id,
        "name2code": name2code, "cp_matched": cp_matched,
        "st_codes": st_codes, "finance_codes": finance_codes,
        "industry_codes": industry_codes,
    }


# ═══════════════════════════════════════════════════════════
# 财务指标分档配置
# ═══════════════════════════════════════════════════════════
# 每项财务指标离散化为 3 档：高(H) / 中(M) / 低(L)
# 分界点：P33（下三分位）和 P67（上三分位），从上市公司数据计算
FIN_DISCRETIZE_CONFIG = {
    # {指标key: [CSMAR列名, 节点名前缀, 越高越好?]}
    # 2026-07-17 修正：所有F编码与fin_map对齐（新下载csmar-7.16数据集）
    "solvency_cr":     ("F010101A", "偿债_流动比率", True),
    "solvency_dar":    ("F011201A", "偿债_资产负债率", False),   # F011201A=资产负债率, 负债率越低越好
    "profit_roa":      ("F050202B", "盈利_ROA", True),
    "profit_roe":      ("F050502B", "盈利_ROE", True),
    "operation_art":   ("F040202B", "经营_应收周转率", True),
    "operation_apt":   ("F040802B", "经营_应付周转率", False),   # 过高可能意味着拖欠
    "growth_tagr":     ("F080601A", "发展_总资产增长率", True),
    "growth_revgr":    ("F081602C", "发展_营收增长率", True),
    "cashflow_cfoni":  ("F060101B", "现金流_经营现金流", True),
    "risklevel_dfl":   ("F070101B", "风险_财务杠杆", False),
    "risklevel_dol":   ("F070201B", "风险_经营杠杆", False),
}

# SCF 产品类型映射
SCF_PRODUCT_TYPES = [
    "应收账款保理", "预付账款融资", "存货质押融资",
    "订单融资", "信用保险融资", "其他SCF",
]

# 诉讼案由类型
LAWSUIT_CATEGORIES = [
    "合同纠纷", "知识产权", "劳动争议",
    "金融借款", "股权纠纷", "行政处罚",
    "执行案件", "其他",
]


# ═══════════════════════════════════════════════════════════
# Step 3: 节点特征（v5 版本 — 仅 13 维非财务特征 + 财务状态节点）
# ═══════════════════════════════════════════════════════════
def build_features_v5(T, E):
    """
    ==========================================================================
    与 v4 的关键区别：
      - Enterprise 特征仅包含非财务维度（SCF + 诉讼 + 营收/资产），共 13 维
      - 财务指标不再压入向量，而是用来构建 FinancialState 节点
      - 返回额外的 financial_states 字典用于后续建边
    ==========================================================================
    """
    print("\n[3/5] Node features (v5.4: 11-dim Enterprise, 纯结构特征)...")
    n, nl = E["n_total"], E["n_listed"]
    DIM_ENT = 11  # SCF(8) + 诉讼(2) + 预留(1), 财务特征全部移入 x_seq
    X_ent = np.zeros((n, DIM_ENT), dtype=np.float32)
    M_ent = np.ones((n, DIM_ENT), dtype=np.float32)
    col = 0

    # ── 3.1 财务指标：不再填入 X_ent，改为构建 financial_records ──
    fin_map = {
        "solvency": {"F010101A": "CR", "F011201A": "DAR", "F010701B": "ICR"},  # 2026-07-16 修正：F011201A=资产负债率, F010701B=利息保障倍数
        "profit":    {"F050202B": "ROA", "F050502B": "ROE"},                    # 2026-07-16 修正：新下载数据F编码01→02
        "operation": {"F040202B": "ART", "F040802B": "APT"},                    # 2026-07-16 修正：新下载数据F编码01→02
        "growth":    {"F080601A": "TAGR", "F081602C": "REVGR"},                 # 2026-07-16 修正：REVGR编码F081601B→F081602C
        "cashflow":  {"F060101B": "CFONI"},
        "risklevel": {"F070101B": "DFL", "F070201B": "DOL"},
    }

    financial_records = defaultdict(dict)  # {eid: {"CR": value, ...}}  最新一期（用于 FinancialState 节点）
    fin_records_by_period = defaultdict(lambda: defaultdict(dict))  # {eid: {accper: {"CR": value, ...}}}

    for tab_key, fmap in fin_map.items():
        df = T[tab_key].copy()
        df["code"] = df["Stkcd"].astype(str).apply(_stock)
        if "Typrep" in df.columns:
            df = df[df["Typrep"].astype(str).str.upper() == "B"]  # 半年报
        df = df.sort_values("Accper")

        for code, group in df.groupby("code"):
            eid = ent_id(code, E)
            if eid < 0 or eid >= nl:
                continue
            eid = int(eid)
            # 取最后 4 期
            recent = group.tail(4)
            for _, row in recent.iterrows():
                accper = str(row["Accper"])
                for fcol, short_name in fmap.items():
                    v = pd.to_numeric(row[fcol], errors="coerce")
                    if pd.notna(v):
                        fin_records_by_period[eid][accper][short_name] = float(np.clip(v, -10, 100))
                    else:
                        fin_records_by_period[eid][accper][short_name] = np.nan

        # 最新一期用于 FinancialState 节点
        df_last = df.groupby("code").last().reset_index()
        eid_arr = df_last["code"].apply(lambda c: ent_id(c, E)).values
        for fcol, short_name in fmap.items():
            v = pd.to_numeric(df_last[fcol], errors="coerce").fillna(0).clip(-10, 100).values
            for i in range(len(df_last)):
                eid = eid_arr[i]
                if 0 <= eid < nl:
                    financial_records[int(eid)][short_name] = float(v[i])

    # ── 3.2 构建 FinancialState 节点（离散化） ──
    fin_nodes, fin_thresholds = _build_financial_state_nodes(financial_records, nl)

    print(f"  FinancialState nodes: {len(fin_nodes)}")
    for name, node in sorted(fin_nodes.items()):
        print(f"    {name}: {node['num_connected']} enterprises, "
              f"mean={node['stat_mean']:.3f}")

    # ── 3.3 Trade credit (8 维) ──
    df_cr = T["scf_credit"].copy()
    eid_cr = df_cr["Symbol"].astype(str).apply(lambda x: ent_id(x, E)).values
    cr_cols = ["AccountPayable", "Prepayment", "AccountReceivable", "TotalAssets",
               "ProvidedTradeCredit", "ObtainedTradeCredit", "BankLoanSize", "SupplierPower1"]
    for fcol in cr_cols:
        if fcol in df_cr.columns:
            v = pd.to_numeric(df_cr[fcol], errors="coerce").fillna(0).values
            for i in range(len(df_cr)):
                eid = eid_cr[i]
                if 0 <= eid < nl:
                    X_ent[eid, col] = v[i]
                    M_ent[eid, col] = 0.0
        col += 1

    # ── 3.4 财务特征不再写入 X_ent（v5.4: 全部走 x_seq → FinTemporalEncoder） ──
    #     col 当前 = 8（SCF 8 维已填），接下去是诉讼 2 维 + 预留 1 维 = 11 维

    # ── 3.5 诉讼严重程度特征 (2 维, col 8-9) ──
    la_full = T["lawsuit"]
    min_la_ts = pd.Timestamp(MIN_LAWSUIT_DATE)
    la_dates = None
    for dc in ["EventSignDate", "DeclareDate"]:
        if dc in la_full.columns:
            la_dates = pd.to_datetime(la_full[dc], errors="coerce")
            break
    la_val = pd.to_numeric(la_full["LAValue"], errors="coerce")
    la_median = la_val.median()
    la_val = la_val.fillna(la_median).clip(lower=1, upper=1_420_000_000)
    la_src = la_full["Symbol"].astype(str).apply(lambda x: ent_id(x, E)).values

    ent_la_sum = defaultdict(float)
    ent_severity = defaultdict(float)
    for i in range(len(la_full)):
        if la_dates is not None:
            d = la_dates.iloc[i]
            if pd.notna(d) and d < min_la_ts:
                continue
        eid = int(la_src[i])
        if eid < 0 or eid >= nl:
            continue
        v = la_val.iloc[i]
        ent_la_sum[eid] += v
        if v < 5_000_000:
            ent_severity[eid] += 1
        elif v < 50_000_000:
            ent_severity[eid] += 3
        else:
            ent_severity[eid] += 10

    for i in range(n):
        X_ent[i, col] = np.log1p(ent_la_sum.get(i, 0))
        M_ent[i, col] = 0.0
    col += 1
    for i in range(n):
        X_ent[i, col] = np.log1p(ent_severity.get(i, 0))
        M_ent[i, col] = 0.0
    col += 1

    # ── 3.6b SCF 打分卡（在归一化前，使用原始值） ──
    risk_scf = _compute_scf_risk_scores(financial_records, X_ent, n, nl)

    # ── 3.7 归一化 ──
    mask_normal = np.ones(nl, dtype=bool)
    st_ids = set()
    for code in E.get("st_codes", []):
        sid = ent_id(code, E)
        if 0 <= sid < nl:
            st_ids.add(sid)
    for sid in st_ids:
        mask_normal[sid] = False
    mask_real = (X_ent[:nl].sum(axis=1) > 0)
    mask_fit = mask_real & mask_normal
    scaler = StandardScaler().fit(X_ent[:nl][mask_fit])
    X_ent = scaler.transform(X_ent)

    print(f"  Enterprise feature dim: {DIM_ENT}, coverage: {mask_real.sum()}/{nl}"
          f" (ST: {len(st_ids & set(range(nl)))})")

    return X_ent, M_ent, fin_nodes, scaler, fin_records_by_period, risk_scf


def _build_financial_state_nodes(financial_records, nl):
    """
    ==========================================================================
    将连续财务指标离散化为 FinancialState 节点。
    分界点 P33/P67 从上市公司数据计算。
    ==========================================================================
    """
    fin_nodes = {}
    thresholds = {}
    node_id = 0

    # short_name 映射
    indicator_short_names = {
        "solvency_cr": "CR", "solvency_dar": "DAR",
        "profit_roa": "ROA", "profit_roe": "ROE",
        "operation_art": "ART", "operation_apt": "APT",
        "growth_tagr": "TAGR", "growth_revgr": "REVGR",
        "cashflow_cfoni": "CFONI",
        "risklevel_dfl": "DFL", "risklevel_dol": "DOL",
    }

    for key, (csm_col, prefix, higher_better) in FIN_DISCRETIZE_CONFIG.items():
        short_name = indicator_short_names[key]
        vals = []
        eid_to_val = {}
        for eid in range(nl):
            v = financial_records.get(eid, {}).get(short_name)
            if v is not None and not np.isnan(v) and not np.isinf(v):
                vals.append(v)
                eid_to_val[eid] = v

        if len(vals) < 30:
            continue

        vals = np.array(vals)
        p33, p67 = np.percentile(vals, [33, 67])
        thresholds[key] = (p33, p67)

        for level, (lo, hi, suffix) in enumerate([
            (-np.inf, p33, "L"),
            (p33, p67, "M"),
            (p67, np.inf, "H"),
        ]):
            node_name = f"{prefix}_{suffix}"
            connected = [eid for eid, v in eid_to_val.items() if lo <= v < hi]

            if len(connected) < 5:
                continue

            fin_nodes[node_name] = {
                "id": node_id,
                "stat_mean": float(np.mean([eid_to_val[e] for e in connected])),
                "stat_std": float(np.std([eid_to_val[e] for e in connected])),
                "num_connected": len(connected),
                "connected_enterprises": connected,
            }
            node_id += 1

    return fin_nodes, thresholds


# ═══════════════════════════════════════════════════════════
# SCF 打分卡：风险分数计算（归一化前调用，使用原始值）
# ═══════════════════════════════════════════════════════════
def _compute_scf_risk_scores(financial_records, X_ent_raw, n_total, nl):
    """
    ==========================================================================
    国内供应链金融打分卡 — 四维度加权计算 0-1 风险分数。

    四个子维度:
      1. 财务健康 (30%): CR(流动比率) + DAR(资产负债率) + ICR(利息保障倍数)
      2. 供应链地位 (25%): BankLoanSize(融资能力) + SupplierPower1(议价力)
      3. 诉讼合规 (25%): 诉讼金额 + 诉讼严重程度
      4. 关联企业质量 (20%): 上市/非上市身份

    每个指标用百分位映射（自适应数据分布）：
      - 越高越好的指标: 子风险分 = 1 - 百分位（高值→低风险）
      - 越高越差的指标: 子风险分 = 百分位（高值→高风险）
      - 缺失值/非上市企业: 子风险分 = 0.5（中性）

    返回: risk_scf (N,) ∈ [0, 1], 越高越危险
    ==========================================================================
    """
    N = n_total

    def _percentile_scores(raw_vals, higher_better, score_all=False):
        """将原始值按百分位映射到 0-1 风险子分数。

        score_all=False: 仅对上市公司 (0..nl-1) 打分，非上市默认 0.5
        score_all=True:  对所有企业打分（用于诉讼等全量可用的数据）
        """
        valid = [v for v in raw_vals if v is not None and not np.isnan(v)]
        if len(valid) < 30:
            return np.full(N, 0.5, dtype=np.float32)

        valid_arr = np.array(valid)
        scores = np.full(N, 0.5, dtype=np.float32)
        score_end = N if score_all else nl
        for i in range(score_end):
            v = raw_vals[i]
            if v is not None and not np.isnan(v):
                pct = (valid_arr < v).mean()
                if higher_better:
                    scores[i] = 1.0 - pct     # 高值 → 低风险
                else:
                    scores[i] = pct            # 高值 → 高风险
        return scores

    # ── 维度1: 财务健康 (30%) —— 仅上市公司有财报数据 ──
    cr_vals  = [financial_records.get(i, {}).get("CR")   for i in range(N)]
    dar_vals = [financial_records.get(i, {}).get("DAR")  for i in range(N)]
    icr_vals = [financial_records.get(i, {}).get("ICR")  for i in range(N)]

    fin_cr  = _percentile_scores(cr_vals,  higher_better=True)    # CR 越高越好
    fin_dar = _percentile_scores(dar_vals, higher_better=False)    # DAR 越高越差
    fin_icr = _percentile_scores(icr_vals, higher_better=True)    # ICR 越高越好

    dim_financial = (fin_cr + fin_dar + fin_icr) / 3.0

    # ── 维度2: 供应链地位 (25%) —— 仅上市公司有 SCF 特征数据 ──
    bank_loan = [float(X_ent_raw[i, 6]) if i < nl and X_ent_raw[i, 6] != 0 else None
                 for i in range(N)]
    supplier_power = [float(X_ent_raw[i, 7]) if i < nl and X_ent_raw[i, 7] != 0 else None
                      for i in range(N)]

    scf_bank   = _percentile_scores(bank_loan,      higher_better=True)
    scf_power  = _percentile_scores(supplier_power,  higher_better=True)

    dim_scf = (scf_bank + scf_power) / 2.0

    # ── 维度3: 诉讼合规 (25%) —— 全量企业可用，score_all=True ──
    lawsuit_amount  = [float(X_ent_raw[i, 8]) if X_ent_raw[i, 8] > 0 else None
                       for i in range(N)]
    lawsuit_severity = [float(X_ent_raw[i, 9]) if X_ent_raw[i, 9] > 0 else None
                        for i in range(N)]

    law_amount  = _percentile_scores(lawsuit_amount,  higher_better=False, score_all=True)
    law_severity = _percentile_scores(lawsuit_severity, higher_better=False, score_all=True)

    dim_lawsuit = (law_amount + law_severity) / 2.0

    # ── 维度4: 关联企业质量 (20%) —— v5.5 修复: 连续化，不再一刀切 ──
    # 上市公司: 基准 0.25，财务越健康 → 越低（0.18~0.45）
    # 非上市:   利用诉讼维度做区分（唯一全量可用的信号），无诉讼→0.42，多诉讼→0.72
    dim_peer = np.full(N, 0.55, dtype=np.float32)
    dim_peer[:nl] = np.clip(0.18 + 0.27 * dim_financial[:nl], 0.18, 0.50)
    # 非上市: dim_lawsuit 偏离 0.5 的程度反映诉讼风险
    lawsuit_deviation = (dim_lawsuit[nl:] - 0.5) * 2.0  # [-1, +1]
    dim_peer[nl:] = np.clip(0.42 + 0.30 * lawsuit_deviation, 0.35, 0.75)

    # ── 加权汇总 ──
    risk_scf = (
        0.30 * dim_financial +
        0.25 * dim_scf +
        0.25 * dim_lawsuit +
        0.20 * dim_peer
    ).astype(np.float32)

    risk_scf = np.clip(risk_scf, 0.0, 1.0)

    print(f"\n  [SCF 打分卡 v5.5] 风险分数统计:")
    print(f"    上市公司 (n={nl}):       mean={risk_scf[:nl].mean():.4f}, "
          f"median={np.median(risk_scf[:nl]):.4f}, "
          f"std={risk_scf[:nl].std():.4f}")
    print(f"    非上市企业 (n={N-nl}):   mean={risk_scf[nl:].mean():.4f}, "
          f"median={np.median(risk_scf[nl:]):.4f}, "
          f"std={risk_scf[nl:].std():.4f}")
    print(f"    子维度均值(上市): 财务={dim_financial[:nl].mean():.3f}, "
          f"供应链={dim_scf[:nl].mean():.3f}, "
          f"诉讼={dim_lawsuit[:nl].mean():.3f}, "
          f"关联={dim_peer[:nl].mean():.3f}")
    print(f"    子维度均值(非上市): 诉讼={dim_lawsuit[nl:].mean():.3f}, "
          f"关联={dim_peer[nl:].mean():.3f}")

    return risk_scf


# ═══════════════════════════════════════════════════════════
# Step 4: 边构建（v5 版本 — 6 种边 + 超边）
# ═══════════════════════════════════════════════════════════
def build_edges_v5(T, E, fin_nodes):
    """
    ==========================================================================
    构建 6 种边：
      - trade, equity (从 v4 复用)
      - has_financial (Enterprise → FinancialState)
      - has_lawsuit   (Enterprise → LawsuitType)
      - uses_scf      (Enterprise → SCFType)
      - legal_rep     (Enterprise → Enterprise, 如果有数据)
    ==========================================================================
    """
    print("\n[4/5] Building edges (v5: 6 edge types)...")
    ei = {}

    # ── 4.1 trade（复用 v4 逻辑） ──
    trade_set = set()
    trade_skipped = 0
    min_trade_ts = pd.Timestamp(MIN_TRADE_DATE)
    for dfk in ["scf_ar", "scf_ap"]:
        df = T[dfk]
        if "Symbol" not in df.columns or "DebtName" not in df.columns:
            continue
        if "EndDate" in df.columns:
            dates = pd.to_datetime(df["EndDate"], errors="coerce")
        else:
            dates = None
        src = df["Symbol"].astype(str).apply(lambda x: ent_id(x, E)).values
        dst = df["DebtName"].astype(str).apply(lambda x: ent_id(x, E)).values
        for i in range(len(df)):
            if dates is not None:
                d = dates.iloc[i]
                if pd.notna(d) and d < min_trade_ts:
                    trade_skipped += 1
                    continue
            s, d_id = int(src[i]), int(dst[i])
            if s >= 0 and d_id >= 0 and s != d_id:
                trade_set.add((s, d_id))
    print(f"  trade: {len(trade_set)} edges (跳过 {trade_skipped} 条过期)")

    # ── 4.2 equity（复用 v4 逻辑） ──
    eq_set = set()
    df_sh = T["shareholder"]
    # 2026-07-16: 新数据已裁剪到2024-2025（16.9万行），无需采样
    dst = df_sh["Stkcd"].astype(str).apply(lambda x: ent_id(x, E)).values
    ratio = pd.to_numeric(df_sh["S0304a"], errors="coerce").fillna(0).values  # 2026-07-16 修正：S0306a(排名)→S0304a(持股比例%)
    names = df_sh["S0301a"].astype(str)
    for i in range(len(df_sh)):
        d, r = int(dst[i]), float(ratio[i])
        if d < 0 or r <= 0:
            continue
        s = ent_id(names.iloc[i], E)
        if s < 0:
            continue
        if s != d:
            eq_set.add((int(s), d))
    print(f"  equity: {len(eq_set)} edges")

    # ── 4.3 has_financial (Enterprise → FinancialState) ──
    fin_edge_set = set()
    for node_name, node_info in fin_nodes.items():
        fin_id = node_info["id"]
        for eid in node_info["connected_enterprises"]:
            fin_edge_set.add((eid, fin_id))
    print(f"  has_financial: {len(fin_edge_set)} edges → {len(fin_nodes)} FinancialState 节点")

    # ── 4.4 has_lawsuit (Enterprise → LawsuitType) ──
    law_edge_set = set()
    la_full = T["lawsuit"]
    la_src = la_full["Symbol"].astype(str).apply(lambda x: ent_id(x, E)).values
    # 尝试从案由列分类（可能叫 CaseReason / CaseType 等）
    case_col = None
    for col_name in ["LATypes", "CaseReason", "CaseType", "ActionType", "ExecutnStus"]:
        if col_name in la_full.columns:
            case_col = col_name
            break

    if case_col:
        case_vals = la_full[case_col].fillna("其他").astype(str).values
        for i in range(len(la_full)):
            eid = int(la_src[i])
            if eid < 0:
                continue
            case_str = str(case_vals[i])
            # 简单分类
            category = _classify_lawsuit(case_str)
            law_edge_set.add((eid, category))
    else:
        # 没有案由列，全部分配到"其他"
        for i in range(len(la_full)):
            eid = int(la_src[i])
            if eid >= 0:
                law_edge_set.add((eid, 7))  # 7 = "其他"
    print(f"  has_lawsuit: {len(law_edge_set)} edges → {len(LAWSUIT_CATEGORIES)} 诉讼类型节点")

    # ── 4.5 uses_scf (Enterprise → SCFType) ──
    scf_edge_set = set()
    df_scf = T.get("scf_credit")
    if df_scf is not None and "Symbol" in df_scf.columns:
        scf_src = df_scf["Symbol"].astype(str).apply(lambda x: ent_id(x, E)).values
        # 根据 SCF 产品使用情况判断（简化：有 SCF 数据就标记为使用）
        for i in range(len(df_scf)):
            eid = int(scf_src[i])
            if eid < 0:
                continue
            # 简化：分配到"应收账款保理"（最常用的 SCF 类型）
            scf_edge_set.add((eid, 0))
    # 也检查 SCF overview
    df_ov = T.get("scf_ov")
    if df_ov is not None and "Symbol" in df_ov.columns:
        ov_src = df_ov["Symbol"].astype(str).apply(lambda x: ent_id(x, E)).values
        if "IsSCFBusiness" in df_ov.columns:
            is_scf = pd.to_numeric(df_ov["IsSCFBusiness"], errors="coerce").fillna(0).values
            for i in range(len(df_ov)):
                if is_scf[i] > 0:
                    eid = int(ov_src[i])
                    if eid >= 0:
                        scf_edge_set.add((eid, 0))
    print(f"  uses_scf: {len(scf_edge_set)} edges → {len(SCF_PRODUCT_TYPES)} SCF 类型节点")

    # ── 4.6 legal_rep（如果有控制人数据） ──
    legal_set = set()
    df_ctrl = T.get("controller")
    if df_ctrl is not None and "Stkcd" in df_ctrl.columns:
        ctrl_src = df_ctrl["Stkcd"].astype(str).apply(lambda x: ent_id(x, E)).values
        if "LegalRepName" in df_ctrl.columns or "S0302a" in df_ctrl.columns:
            leg_col = "LegalRepName" if "LegalRepName" in df_ctrl.columns else "S0302a"
            leg_names = df_ctrl[leg_col].astype(str).values
            # 按法人姓名分组，同一法人的企业互相连边
            leg_groups = defaultdict(list)
            for i in range(len(df_ctrl)):
                eid = int(ctrl_src[i])
                if eid >= 0 and len(leg_names[i]) > 1:
                    leg_groups[leg_names[i]].append(eid)
            for name, eids in leg_groups.items():
                for a in range(len(eids)):
                    for b in range(a + 1, len(eids)):
                        legal_set.add((eids[a], eids[b]))
                        legal_set.add((eids[b], eids[a]))
    print(f"  legal_rep: {len(legal_set)} edges")

    # ── 转为 tensor ──
    if trade_set:
        ei[("enterprise", "trade", "enterprise")] = torch.tensor(
            list(trade_set)).long().t()
    if eq_set:
        ei[("enterprise", "equity", "enterprise")] = torch.tensor(
            list(eq_set)).long().t()
    if fin_edge_set:
        ei[("enterprise", "has_financial", "financial_state")] = torch.tensor(
            list(fin_edge_set)).long().t()
    if law_edge_set:
        ei[("enterprise", "has_lawsuit", "lawsuit_type")] = torch.tensor(
            [(s, d) for s, d in law_edge_set]).long().t()
    if scf_edge_set:
        ei[("enterprise", "uses_scf", "scf_type")] = torch.tensor(
            list(scf_edge_set)).long().t()
    if legal_set:
        ei[("enterprise", "legal_rep", "enterprise")] = torch.tensor(
            list(legal_set)).long().t()

    return ei


def _classify_lawsuit(case_str):
    """根据案由字符串分类到预定义类型"""
    case_str = str(case_str).lower()
    if any(w in case_str for w in ["合同", "买卖", "租赁", "承包", "承揽"]):
        return 0  # 合同纠纷
    elif any(w in case_str for w in ["知识产权", "专利", "商标", "著作权", "版权"]):
        return 1  # 知识产权
    elif any(w in case_str for w in ["劳动", "劳务", "工伤", "社保"]):
        return 2  # 劳动争议
    elif any(w in case_str for w in ["借款", "贷款", "金融", "融资", "担保"]):
        return 3  # 金融借款
    elif any(w in case_str for w in ["股权", "股东", "出资"]):
        return 4  # 股权纠纷
    elif any(w in case_str for w in ["行政", "处罚", "税务", "工商"]):
        return 5  # 行政处罚
    elif any(w in case_str for w in ["执行", "强制", "查封"]):
        return 6  # 执行案件
    return 7  # 其他


# ═══════════════════════════════════════════════════════════
# Step 4.5: 超边构建
# ═══════════════════════════════════════════════════════════
def build_hyperedges(E, edge_index_dict, nl):
    """
    ==========================================================================
    从 6 种边中构建 4 张超图。

    supply 超图: 每条超边 = 同一核心企业 + 其所有 trade 邻居
    equity 超图: 每条超边 = 同一控制人下的所有企业
    legal_rep 超图: 每条超边 = 同一法人旗下的所有企业
    industry 超图: 每条超边 = 同一行业代码的所有企业（从 E 的行业信息构建）
    ==========================================================================
    """
    print("\n[4.5/5] Building hyperedges (4 views)...")
    n = E["n_total"]
    hyperedges = {}

    # ── supply 超图 ──
    trade_ei = edge_index_dict.get(("enterprise", "trade", "enterprise"))
    supply_hyperedges = []
    if trade_ei is not None:
        src = trade_ei[0].numpy()
        dst = trade_ei[1].numpy()
        # 以每个节点作为核心，其邻居 + 自己构成超边
        adj = defaultdict(set)
        for s, d in zip(src, dst):
            adj[int(s)].add(int(d))
            adj[int(d)].add(int(s))
        for core, neighbors in adj.items():
            hyperedge = [core] + list(neighbors)
            if len(hyperedge) >= 3:  # 至少 3 个节点才有超边意义
                supply_hyperedges.append(torch.tensor(hyperedge))
    hyperedges["supply"] = supply_hyperedges
    print(f"  supply: {len(supply_hyperedges)} 条超边")

    # ── equity 超图 ──
    eq_ei = edge_index_dict.get(("enterprise", "equity", "enterprise"))
    equity_hyperedges = []
    if eq_ei is not None:
        src = eq_ei[0].numpy()
        dst = eq_ei[1].numpy()
        # 连通分量作为超边
        parent = list(range(n))
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py
        for s, d in zip(src, dst):
            union(int(s), int(d))
        comps = defaultdict(list)
        for i in range(n):
            comps[find(i)].append(i)
        for comp_nodes in comps.values():
            if len(comp_nodes) >= 2:
                equity_hyperedges.append(torch.tensor(comp_nodes))
    hyperedges["equity"] = equity_hyperedges
    print(f"  equity: {len(equity_hyperedges)} 条超边")

    # ── legal_rep 超图 ──
    leg_ei = edge_index_dict.get(("enterprise", "legal_rep", "enterprise"))
    legal_hyperedges = []
    if leg_ei is not None:
        src = leg_ei[0].numpy()
        dst = leg_ei[1].numpy()
        parent2 = list(range(n))
        def find2(x):
            while parent2[x] != x:
                parent2[x] = parent2[parent2[x]]
                x = parent2[x]
            return x
        def union2(x, y):
            px, py = find2(x), find2(y)
            if px != py:
                parent2[px] = py
        for s, d in zip(src, dst):
            union2(int(s), int(d))
        comps2 = defaultdict(list)
        for i in range(n):
            comps2[find2(i)].append(i)
        for comp_nodes in comps2.values():
            if len(comp_nodes) >= 2:
                legal_hyperedges.append(torch.tensor(comp_nodes))
    hyperedges["legal_rep"] = legal_hyperedges
    print(f"  legal_rep: {len(legal_hyperedges)} 条超边")

    # ── industry 超图 ──
    industry_hyperedges = []
    if "industry_codes" in E:
        ind_groups = defaultdict(list)
        for eid, code in E["industry_codes"].items():
            if code and len(str(code)) >= 2:
                # 取前 2 位作为大类行业
                ind_groups[str(code)[:2]].append(int(eid))
        for code, eids in ind_groups.items():
            if len(eids) >= 5:
                industry_hyperedges.append(torch.tensor(eids))
    hyperedges["industry"] = industry_hyperedges
    print(f"  industry: {len(industry_hyperedges)} 条超边")

    return hyperedges


# ═══════════════════════════════════════════════════════════
# Step 5: 标签 + 组装
# ═══════════════════════════════════════════════════════════
def build_labels_and_assemble_v5(T, E, X_ent, M_ent, fin_nodes, ei, hyperedges, fin_records_by_period=None, risk_scf=None):
    print("\n[5/5] Labels + assembly (v5)...")
    n, nl = E["n_total"], E["n_listed"]

    # ── 标签（复用 v4 逻辑） ──
    y_white = np.ones(n, dtype=np.float32)
    y_risk = np.zeros(n, dtype=np.float32)
    y_grade = np.full(n, 2, dtype=int)

    bad = defaultdict(list)
    df_ar = T["scf_ar"]
    eid_ar = df_ar["Symbol"].astype(str).apply(lambda x: ent_id(x, E)).values
    ratio_ar = pd.to_numeric(df_ar.get("EndingBadDebtProRatio", 0), errors="coerce").fillna(0).values
    for i in range(len(df_ar)):
        eid = int(eid_ar[i])
        if eid >= 0:
            bad[eid].append(float(ratio_ar[i]))
    for eid, ratios in bad.items():
        if np.mean(ratios) > 20:
            y_white[eid] = 0
            y_risk[eid] = 1

    exec_cnt = defaultdict(int)
    la = T["lawsuit"]
    eid_la = la["Symbol"].astype(str).apply(lambda x: ent_id(x, E)).values
    exec_st = la["ExecutnStus"].fillna("").astype(str).values
    for i in range(len(la)):
        eid = int(eid_la[i])
        if eid >= 0 and "执行" in exec_st[i]:
            exec_cnt[eid] += 1
    for eid, cnt in exec_cnt.items():
        y_risk[eid] = 1
        if cnt >= 3:
            y_white[eid] = 0

    for i in range(nl):
        if y_white[i] == 1 and y_risk[i] == 0:
            y_grade[i] = 0
        elif y_white[i] == 1 and y_risk[i] == 1:
            y_grade[i] = 1
        elif y_white[i] == 0 and y_risk[i] == 0:
            y_grade[i] = 2
        else:
            y_grade[i] = 3

    # ── 构建 x_dict（4 种节点类型） ──
    x_dict = {
        "enterprise": torch.tensor(X_ent),
    }

    # FinancialState 节点特征: [stat_mean, stat_std, log1p(num_connected), 5 维 padding]
    fin_node_features = np.zeros((len(fin_nodes), 8), dtype=np.float32)
    fin_id_map = {}
    for node_name, node_info in fin_nodes.items():
        fid = node_info["id"]
        fin_id_map[fid] = node_name
        fin_node_features[fid, 0] = node_info["stat_mean"]
        fin_node_features[fid, 1] = node_info["stat_std"]
        fin_node_features[fid, 2] = np.log1p(node_info["num_connected"])
    x_dict["financial_state"] = torch.tensor(fin_node_features)

    # LawsuitType 节点特征: 可学习（初始化随机）
    n_law_types = len(LAWSUIT_CATEGORIES)
    x_dict["lawsuit_type"] = torch.randn(n_law_types, 8) * 0.01

    # SCFType 节点特征: 可学习
    n_scf_types = len(SCF_PRODUCT_TYPES)
    x_dict["scf_type"] = torch.randn(n_scf_types, 8) * 0.01

    # ── 结构特征（保留 FeatureGate 兼容） ──
    #     注意：v5 中结构特征仅用于 Enterprise 的 13 维特征
    X_struct, struct_hint = _compute_structural_features_v5(X_ent, ei, n, nl)
    x_struct = {"enterprise": torch.tensor(X_struct)}
    x_missing = {"enterprise": torch.tensor(M_ent)}
    struct_hint_dict = {"enterprise": torch.tensor(struct_hint)}

    # ── 时序快照（4 个半年） ──
    x_seq = _build_semi_annual_sequences(fin_records_by_period, n, nl)

    # ── 分层随机切分 ──
    from sklearn.model_selection import train_test_split
    all_idx = np.arange(nl)
    y_listed = y_white[:nl]
    train_idx, rest_idx = train_test_split(
        all_idx, test_size=0.3, stratify=y_listed, random_state=SEED)
    y_rest = y_listed[rest_idx]
    val_idx, test_idx = train_test_split(
        rest_idx, test_size=0.5, stratify=y_rest, random_state=SEED)

    train = torch.zeros(n, dtype=bool)
    train[train_idx] = True
    val = torch.zeros(n, dtype=bool)
    val[val_idx] = True
    test = torch.zeros(n, dtype=bool)
    test[test_idx] = True

    # ── 子图过滤：176K 节点 → 只保留有标签节点 + 2 跳邻居 ──
    print("  子图过滤 (保留有标签节点 + 2跳邻居)...")
    _n_before = n
    _nl_orig = nl
    labeled_set = set(range(nl))  # 所有上市公司（有标签）
    keep_ent = set(labeled_set)

    # 构建企业→企业邻接表（仅 trade + equity）
    ent_adj = {i: set() for i in range(n)}
    for (src, rel, dst), edge_idx in ei.items():
        if src == "enterprise" and dst == "enterprise":
            src_arr = edge_idx[0].numpy()
            dst_arr = edge_idx[1].numpy()
            for u, v in zip(src_arr, dst_arr):
                ent_adj[int(u)].add(int(v))
                ent_adj[int(v)].add(int(u))

    # 2 跳 BFS 扩展
    frontier = labeled_set
    for hop in range(2):
        next_frontier = set()
        for node in frontier:
            for nb in ent_adj.get(node, set()):
                if nb not in keep_ent:
                    keep_ent.add(nb)
                    next_frontier.add(nb)
        frontier = next_frontier
        print(f"    {hop+1}跳: +{len(frontier)} 节点, 累计 {len(keep_ent)}")
        if not frontier:
            break

    # 构建旧ID→新ID映射
    keep_list = sorted(keep_ent)
    old2new = {old: new for new, old in enumerate(keep_list)}
    new_n = len(keep_list)

    # 映射 enterprise 特征
    X_ent = X_ent[keep_list]
    M_ent = M_ent[keep_list]
    X_struct = X_struct[keep_list]
    struct_hint = struct_hint[keep_list]
    if x_seq is not None:
        x_seq = x_seq[keep_list]
    if risk_scf is not None:
        risk_scf = risk_scf[keep_list]

    # 映射标签
    y_white = y_white[keep_list]
    y_risk = y_risk[keep_list]
    y_grade = y_grade[keep_list]

    # 映射 mask
    train = train[keep_list]
    val = val[keep_list]
    test = test[keep_list]

    # 更新 nl（新索引下仍是前 nl 个为 labeled）
    new_nl = sum(1 for i in keep_list if i < _nl_orig)

    # 映射边
    new_ei = {}
    for etype, edge_idx in ei.items():
        src_nt, rel, dst_nt = etype
        if src_nt == "enterprise" and dst_nt == "enterprise":
            # 两端都必须在 keep_ent 中
            src_arr = edge_idx[0].numpy()
            dst_arr = edge_idx[1].numpy()
            valid = [(old2new[int(u)], old2new[int(v)])
                     for u, v in zip(src_arr, dst_arr)
                     if int(u) in old2new and int(v) in old2new]
            if valid:
                new_ei[etype] = torch.tensor(valid).T
        elif src_nt == "enterprise":
            # 源端（enterprise）必须保留，目标端不变
            src_arr = edge_idx[0].numpy()
            dst_arr = edge_idx[1].numpy()
            valid = [(old2new[int(u)], int(v))
                     for u, v in zip(src_arr, dst_arr)
                     if int(u) in old2new]
            if valid:
                new_ei[etype] = torch.tensor(valid).T
        else:
            raise NotImplementedError(f"Unexpected edge type: {etype}")

    # 映射超边
    new_hyperedges = {}
    for view_name, he_list in hyperedges.items():
        filtered = []
        for he in he_list:
            he_arr = he.numpy()
            new_he = [old2new[int(v)] for v in he_arr if int(v) in old2new]
            if len(new_he) >= 2:
                filtered.append(torch.tensor(new_he))
        new_hyperedges[view_name] = filtered

    print(f"  过滤完成: {_n_before} → {new_n} enterprise 节点 "
          f"(labeled={new_nl})")
    print(f"  超边: supply {len(new_hyperedges.get('supply',[]))}, "
          f"equity {len(new_hyperedges.get('equity',[]))}, "
          f"legal_rep {len(new_hyperedges.get('legal_rep',[]))}, "
          f"industry {len(new_hyperedges.get('industry',[]))}")

    # 更新变量
    n, nl = new_n, new_nl
    ei = new_ei
    hyperedges = new_hyperedges
    x_dict = {
        "enterprise": torch.tensor(X_ent),
        "financial_state": x_dict["financial_state"],
        "lawsuit_type": x_dict["lawsuit_type"],
        "scf_type": x_dict["scf_type"],
    }
    x_struct = {"enterprise": torch.tensor(X_struct)}
    x_missing = {"enterprise": torch.tensor(M_ent)}
    struct_hint_dict = {"enterprise": torch.tensor(struct_hint)}

    data = HeteroGraphData(
        x_dict=x_dict,
        edge_index_dict=ei,
        hyperedges=hyperedges,
        y_white=torch.tensor(y_white),
        y_risk=torch.tensor(y_risk),
        y_grade=torch.tensor(y_grade),
        train_mask=train, val_mask=val, test_mask=test,
        num_enterprises=n, num_listed=nl, total_nodes=n + len(fin_nodes) + n_law_types + n_scf_types,
        x_struct=x_struct, x_missing=x_missing,
        struct_hint=struct_hint_dict,
        x_seq=torch.tensor(x_seq) if x_seq is not None else None,
        risk_scf=torch.tensor(risk_scf) if risk_scf is not None else None,
    )
    data.summary()
    return data


def _compute_structural_features_v5(X, ei, n, nl):
    """简化版结构特征计算（仅用于 FeatureGate）"""
    D = X.shape[1]
    X_t = torch.tensor(X)
    ei_ent = {}
    for (src, rel, dst), edge_idx in ei.items():
        if src == "enterprise" and dst == "enterprise":
            ei_ent[rel] = edge_idx

    X_struct = np.zeros_like(X)
    for rel, edge_idx in ei_ent.items():
        src, dst_ = edge_idx[0].numpy(), edge_idx[1].numpy()
        agg = np.zeros((n, D), dtype=np.float64)
        cnt = np.zeros(n, dtype=np.int32)
        np.add.at(agg, dst_, X[src])
        np.add.at(cnt, dst_, 1)
        np.add.at(agg, src, X[dst_])
        np.add.at(cnt, src, 1)
        mask = cnt > 0
        agg[mask] /= cnt[mask, np.newaxis]
        X_struct += agg
    n_rel = max(len(ei_ent), 1)
    X_struct /= n_rel
    mask_isolated = (X_struct.sum(axis=1) == 0)
    if nl > 0:
        listed_mean = X[:nl][X[:nl].sum(axis=1) > 0].mean(axis=0)
    else:
        listed_mean = X.mean(axis=0)
    X_struct[mask_isolated] = listed_mean

    S_DIM = 8
    struct_hint = np.zeros((n, S_DIM), dtype=np.float32)
    struct_hint[:nl, 0] = 1.0
    struct_hint[:, 1] = (X.sum(axis=1) != 0).astype(np.float32)
    for rel, edge_idx in ei_ent.items():
        col = {"trade": 2, "equity": 3, "legal_rep": 4}.get(rel, -1)
        if col < 0:
            continue
        deg = np.bincount(edge_idx[0].numpy(), minlength=n) + \
              np.bincount(edge_idx[1].numpy(), minlength=n)
        struct_hint[:, col] += deg.astype(np.float32)
    struct_hint[:, 5] = struct_hint[:, 2:5].sum(axis=1)
    struct_hint[:, 6] = np.log1p(struct_hint[:, 5])
    for i in range(nl):
        struct_hint[i, 7] = (X[i] != 0).mean()
    for c in [2, 3, 4, 5, 6]:
        mx = struct_hint[:, c].max()
        if mx > 0:
            struct_hint[:, c] /= mx

    return X_struct.astype(np.float32), struct_hint.astype(np.float32)


def _build_semi_annual_sequences(fin_records_by_period, n_total, nl):
    """
    ==========================================================================
    构建 4 步半年报时序特征（真实多期数据版）。

    从 fin_records_by_period（多期财务指标）中每家企业取最后 4 个
    报告期，构建 (N, 4, 12+1) 序列：
      - 12 维财务指标（CR, DAR, ICR, ROA, ROE, ART, APT, TAGR, REVGR,
                          CFONI, DFL, DOL）
      - 1  维 has_feature 标记（该期是否有 ≥50% 指标真实存在）

    缺失值处理：
      - ≥3 期有效 → 线性插值
      - 2 期有效 → 公司自身中位数填充
      - <2 期有效 → 全零 + has_feature=0（TemporalGate 走 MLP fallback）

    非上市企业（eid >= nl）：全零序列，TemporalGate 自动走 MLP。
    ==========================================================================
    """
    # 12 个财务指标（固定顺序）
    INDICATOR_ORDER = [
        "CR", "DAR", "ICR",        # 偿债能力
        "ROA", "ROE",               # 盈利能力
        "ART", "APT",               # 经营能力
        "TAGR", "REVGR",            # 发展能力
        "CFONI",                    # 现金流
        "DFL", "DOL",               # 风险水平
    ]
    D = len(INDICATOR_ORDER)  # 12

    # ── 第一遍：收集所有上市公司的原始值 + 初步构建（不含标准化） ──
    x_seq = np.zeros((n_total, 4, D + 1), dtype=np.float32)  # 最后一维 = has_feature
    all_raw_vals = {d_idx: [] for d_idx in range(D)}  # 用于后续按指标标准化

    listed_with_data = 0
    total_periods_collected = 0

    for eid in range(nl):
        periods_dict = fin_records_by_period.get(eid, {})
        if not periods_dict:
            continue

        # 按报告期排序，取最后 4 期
        sorted_periods = sorted(periods_dict.keys())
        recent_periods = sorted_periods[-4:]
        total_periods_collected += len(recent_periods)
        listed_with_data += 1

        # 构建 (num_periods, D) 矩阵
        num_p = len(recent_periods)
        feats_raw = np.full((num_p, D), np.nan, dtype=np.float32)
        for p_idx, accper in enumerate(recent_periods):
            rec = periods_dict[accper]
            for d_idx, ind in enumerate(INDICATOR_ORDER):
                v = rec.get(ind, np.nan)
                if not np.isnan(v):
                    feats_raw[p_idx, d_idx] = v
                    all_raw_vals[d_idx].append(v)

        # ── 缺失值处理 ──
        valid_mask = ~np.isnan(feats_raw)  # (num_p, D)
        n_valid_periods = np.sum(valid_mask.any(axis=1))

        if n_valid_periods >= 3:
            for d_idx in range(D):
                col = feats_raw[:, d_idx].copy()
                valid_idx = np.where(~np.isnan(col))[0]
                if len(valid_idx) >= 2:
                    xp = valid_idx.astype(np.float64)
                    fp = col[valid_idx]
                    all_idx = np.arange(num_p, dtype=np.float64)
                    col = np.interp(all_idx, xp, fp)
                elif len(valid_idx) == 1:
                    col = np.full(num_p, col[valid_idx[0]])
                else:
                    col = np.zeros(num_p)
                feats_raw[:, d_idx] = col
        elif n_valid_periods >= 2:
            col_medians = np.nanmedian(feats_raw, axis=0)
            col_medians = np.where(np.isnan(col_medians), 0.0, col_medians)
            for d_idx in range(D):
                col = feats_raw[:, d_idx].copy()
                col[np.isnan(col)] = col_medians[d_idx]
                feats_raw[:, d_idx] = col
        else:
            feats_raw = np.zeros((num_p, D), dtype=np.float32)

        # 暂存（先不填 has_feature，标准化后再填）
        start_col = 4 - num_p
        for p_idx in range(num_p):
            seq_pos = start_col + p_idx
            x_seq[eid, seq_pos, :D] = feats_raw[p_idx]

    # ── 标准化：按指标计算均值/标准差（只用有数据的上市公司），做 Z-score ──
    indicator_mean = np.zeros(D, dtype=np.float32)
    indicator_std = np.ones(D, dtype=np.float32)   # 防止除零
    for d_idx in range(D):
        vals = all_raw_vals[d_idx]
        if len(vals) >= 30:
            indicator_mean[d_idx] = np.mean(vals)
            indicator_std[d_idx] = np.std(vals) + 1e-8

    # 对所有上市公司 + 非上市公司统一做标准化
    for eid in range(nl):
        for q in range(4):
            if x_seq[eid, q, :D].sum() != 0:  # 有数据的期才标准化
                x_seq[eid, q, :D] = (x_seq[eid, q, :D] - indicator_mean) / indicator_std
                # has_feature: 该期是否有 ≥50% 指标非零（标准化后非零 = 原始有值）
                n_real = (np.abs(x_seq[eid, q, :D]) > 1e-6).sum()
                if n_real >= D * 0.5:
                    x_seq[eid, q, D] = 1.0

    print(f"  时序序列: {listed_with_data}/{nl} 上市公司有时序数据, "
          f"平均 {total_periods_collected / max(listed_with_data, 1):.1f} 期/企")
    if listed_with_data > 0:
        print(f"  标准化: 均值范围 [{indicator_mean.min():.3f}, {indicator_mean.max():.3f}], "
              f"标准差范围 [{indicator_std.min():.3f}, {indicator_std.max():.3f}]")

    return x_seq


# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════
def load_csmar_data_v5():
    import time
    stages = []
    t0 = time.time()
    T = load_all()
    stages.append(("Load", time.time() - t0))
    t0 = time.time()
    E = build_entities(T)
    stages.append(("Entities", time.time() - t0))
    t0 = time.time()
    X_ent, M_ent, fin_nodes, scaler, fin_records_by_period, risk_scf = build_features_v5(T, E)
    stages.append(("Features", time.time() - t0))
    t0 = time.time()
    ei = build_edges_v5(T, E, fin_nodes)
    stages.append(("Edges", time.time() - t0))
    t0 = time.time()
    hyperedges = build_hyperedges(E, ei, E["n_listed"])
    stages.append(("Hyperedges", time.time() - t0))
    t0 = time.time()
    data = build_labels_and_assemble_v5(T, E, X_ent, M_ent, fin_nodes, ei, hyperedges, fin_records_by_period, risk_scf=risk_scf)
    stages.append(("Assembly", time.time() - t0))
    print("\nTiming:")
    for name, sec in stages:
        print(f"  {name}: {sec:.1f}s")
    print(f"\n  数据过滤统计:")
    print(f"    金融业排除: {EXCLUDE_FINANCE}, ST标记: {len(E.get('st_codes', []))} 家")
    return data


if __name__ == "__main__":
    import time
    t0 = time.time()
    data = load_csmar_data_v5()
    print(f"\nTotal: {time.time()-t0:.1f}s")
