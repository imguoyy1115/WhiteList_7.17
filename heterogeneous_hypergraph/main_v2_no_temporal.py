"""
================================================================================
消融实验 v2-A: 特征分工 + FinMLP（去掉财务时序GRU）
================================================================================
对照 main_v2.py（完整模型），将 FinGRU 替换为 FinMLP 静态投影，
测试在特征分工架构下，财务半年度时序GRU是否仍有贡献。

用法:
  cd heterogeneous_hypergraph
  python main_v2_no_temporal.py

对比:
  main_v2.py            (特征分工 + FinGRU):          ???
  main_v2_no_temporal.py (特征分工 + FinMLP):          ???
================================================================================
"""

import sys
import os
import time
import torch
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
# ── 消融: 去掉时序 ──
config.ABLATION_NO_TEMPORAL = True

from config import (
    SEED, DEVICE, OUTPUT_DIR, EDGE_TYPES,
)
from data_loader.csmar_loader import load_csmar_data_v5
from train import HyperHeteroModel, train, evaluate_model

torch.manual_seed(SEED)


def main():
    print("=" * 60)
    print("  消融实验: 特征分工 + FinMLP（无财务时序GRU）")
    print("=" * 60)

    # ── Step 1: 加载数据 ──
    print("\n[Step 1] 加载 CSMAR 数据 (v5 管线)...")
    t0 = time.time()
    data = load_csmar_data_v5()
    print(f"  数据加载完成，总耗时 {time.time() - t0:.1f}s")

    # ── Step 2: 构建模型 ──
    print("\n[Step 2] 构建模型（特征分工 + FinMLP消融）...")
    in_dims = {ntype: data.x_dict[ntype].shape[1] for ntype in data.x_dict}

    valid_edge_types = [et for et in EDGE_TYPES if et in data.edge_index_dict]
    if valid_edge_types != EDGE_TYPES:
        missing = set(EDGE_TYPES) - set(valid_edge_types)
        print(f"  注意: 以下边类型在数据中不存在，跳过: {missing}")

    model = HyperHeteroModel(in_dims=in_dims, edge_types=valid_edge_types)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  总参数量: {total_params:,}")
    print(f"  架构: 超图4视图 + 异构通道 + FusionGate + FinMLP(消融) (特征分工)")

    # ── Step 3: 训练 ──
    print(f"\n[Step 3] 训练...")
    model = train(model, data)

    # ── Step 4: 测试集评估 ──
    print(f"\n[Step 4] 测试集评估...")
    test_auc, test_acc, test_prec10 = evaluate_model(model, data, data.test_mask)
    print(f"  测试 AUC: {test_auc:.4f}")
    print(f"  测试 Acc: {test_acc:.4f}")
    print(f"  Precision@10: {test_prec10:.4f}")

    # ── Step 5: 保存 ──
    print(f"\n[Step 5] 保存到 {OUTPUT_DIR}/ ...")
    torch.save(model.state_dict(), f"{OUTPUT_DIR}/model_v2_no_temporal.pt")
    pd.DataFrame({
        "experiment": ["v2_no_temporal"],
        "test_auc": [test_auc],
        "test_acc": [test_acc],
        "precision_at_10": [test_prec10]
    }).to_csv(f"{OUTPUT_DIR}/results_v2_no_temporal.csv", index=False)
    print(f"  [OK] model_v2_no_temporal.pt  [OK] results_v2_no_temporal.csv")

    print("\n" + "=" * 60)
    print(f"  消融完成。Test AUC = {test_auc:.4f}, Precision@10 = {test_prec10:.4f}")
    print(f"  对照 main_v2.py (FinGRU): Δ = AUC_v2_FIN_GRU - {test_auc:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
