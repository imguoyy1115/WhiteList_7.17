"""
================================================================================
v5.3: 特征分工 + 财务时序GRU（三图并联架构 — 图3预留）
================================================================================

  - 图1（超图通道）: 多视图超图学习企业结构位置
  - 图2（异构通道）: 异构图学习产业链资金流向
  - 图3（风险传播网络）: 预留接口，学习风险沿网络传导

  - 特征分工: 同构通道(SCF+诉讼=10维) / 异构通道(营收+资产=2维)
  - 财务时序: 小GRU(2→8→2)在 4 步半年度财务序列上建模
  - 中小企业财务缺失 → fin_missing_emb + MLP fallback
  - 训练结束后自动运行 DebtRank 风险传播 → 生成全企业风险分数

用法:
  cd heterogeneous_hypergraph
  python main_v2.py
================================================================================
"""

import sys
import os
import time
import torch
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
# ── 完整模型: FinGRU ──
config.ABLATION_NO_TEMPORAL = False

from config import (
    SEED, DEVICE, OUTPUT_DIR, EDGE_TYPES,
)
from data_loader.csmar_loader import load_csmar_data_v5
from train import HyperHeteroModel, train, evaluate_model

torch.manual_seed(SEED)


def main():
    print("=" * 60)
    print("  v5.3: 特征分工 + 财务时序GRU（三图并联 — 图3预留）")
    print("=" * 60)

    # ── Step 1: 加载数据 ──
    print("\n[Step 1] 加载 CSMAR 数据 (v5 管线)...")
    t0 = time.time()
    data = load_csmar_data_v5()
    print(f"  数据加载完成，总耗时 {time.time() - t0:.1f}s")

    # ── Step 2: 构建模型 ──
    print("\n[Step 2] 构建模型（特征分工: 同构10维 + 异构2维）...")
    in_dims = {ntype: data.x_dict[ntype].shape[1] for ntype in data.x_dict}

    valid_edge_types = [et for et in EDGE_TYPES if et in data.edge_index_dict]
    if valid_edge_types != EDGE_TYPES:
        missing = set(EDGE_TYPES) - set(valid_edge_types)
        print(f"  注意: 以下边类型在数据中不存在，跳过: {missing}")

    model = HyperHeteroModel(in_dims=in_dims, edge_types=valid_edge_types)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  总参数量: {total_params:,}")
    print(f"  节点类型: {list(in_dims.keys())}")
    print(f"  边类型: {[et[1] for et in valid_edge_types]}")
    print(f"  超图视图: {list(data.hyperedges.keys())}")
    print(f"  架构: 超图4视图 + 异构{len(in_dims)}节点{len(valid_edge_types)}边 + FusionGate + FinGRU (特征分工, 图3预留)")

    # ── Step 3: 训练 ──
    print(f"\n[Step 3] 训练...")
    model = train(model, data)

    # ── Step 4: 测试集评估 ──
    print(f"\n[Step 4] 测试集评估...")
    test_auc, test_acc, test_prec10 = evaluate_model(model, data, data.test_mask)
    print(f"  测试 AUC: {test_auc:.4f}")
    print(f"  测试 Acc: {test_acc:.4f}")
    print(f"  Precision@10: {test_prec10:.4f}")

    # ── Step 5: 保存模型 ──
    print(f"\n[Step 5] 保存到 {OUTPUT_DIR}/ ...")
    torch.save(model.state_dict(), f"{OUTPUT_DIR}/model_v2.pt")
    pd.DataFrame({"test_auc": [test_auc], "test_acc": [test_acc], "precision_at_10": [test_prec10]}).to_csv(
        f"{OUTPUT_DIR}/results_v2.csv", index=False
    )
    print(f"  [OK] model_v2.pt  [OK] results_v2.csv")

    # ── Step 6: 风险传播计算（基于已学好的模型，纯计算，不参与梯度） ──
    from risk_propagation import run_risk_propagation_pipeline
    r_final, r_model, risk_df = run_risk_propagation_pipeline(
        model, data, alpha=0.70, K_max=100, tol=1e-6
    )

    print("\n" + "=" * 60)
    print(f"  训练完成。Test AUC = {test_auc:.4f}, Precision@10 = {test_prec10:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
