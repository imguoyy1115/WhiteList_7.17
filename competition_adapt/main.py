"""
================================================================================
主入口 — 赛题数据适配版
================================================================================
基于 heterogeneous_hypergraph/main_v2.py 重写。

架构: 多视角超图 + 企业属性编码 → 双通道融合 → 信用风险预测

用法:
  cd competition_adapt
  python main.py
================================================================================
"""

import sys
import os
import time
import torch
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import SEED, DEVICE, OUTPUT_DIR
from data_loader.competition_loader import load_competition_data
from train import CompetitionModel, train, evaluate_model

torch.manual_seed(SEED)


def main():
    print("=" * 60)
    print("  赛题适配: 多视角超图 + 企业属性编码")
    print("  数据: 10万笔数据交付 (工商+招投标+司法涉诉)")
    print("=" * 60)

    # ── Step 1: 加载数据 ──
    print("\n[Step 1] 加载赛题数据...")
    t0 = time.time()
    data = load_competition_data()
    print(f"  数据加载完成，总耗时 {time.time() - t0:.1f}s")

    # ── Step 2: 构建模型 ──
    print("\n[Step 2] 构建模型...")
    ent_dim = data.x_ent.shape[1]
    model = CompetitionModel(ent_dim=ent_dim)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  企业特征维度: {ent_dim}")
    print(f"  总参数量: {total_params:,}")
    print(f"  超图视图: {list(data.hyperedges.keys())}")
    print(f"  边类型: {[et[1] for et in data.edge_index_dict.keys()]}")
    print(f"  架构: 超图5视图 + EnterpriseMLP + FusionGate → 信用风险预测")

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
    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "model_competition.pt"))
    pd.DataFrame({
        "test_auc": [test_auc],
        "test_acc": [test_acc],
        "precision_at_10": [test_prec10]
    }).to_csv(os.path.join(OUTPUT_DIR, "results_competition.csv"), index=False)
    print(f"  [OK] model_competition.pt  [OK] results_competition.csv")

    # ── Step 6: 全量推理 + 风险排序 ──
    print(f"\n[Step 6] 全量推理（生成企业信用排序）...")
    model.eval()
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    data_dev = data.to(device)
    model = model.to(device)

    with torch.no_grad():
        logit_credit, logit_grade = model(
            data_dev.x_ent,
            data_dev.edge_index_dict,
            data_dev.hyperedges,
            struct_hint=data_dev.struct_hint if data_dev.struct_hint is not None else None,
        )
        prob_credit = torch.sigmoid(logit_credit).cpu().squeeze(-1).numpy()
        pred_grade = logit_grade.argmax(dim=1).cpu().numpy()

    # 全量风险排序
    N = len(prob_credit)
    sorted_idx = np.argsort(prob_credit)  # 升序: 低风险在前

    results_df = pd.DataFrame({
        "enterprise_id": np.arange(N),
        "risk_score": prob_credit,          # 0=低风险, 1=高风险
        "pred_grade": pred_grade,
        "credit_rank_raw": data.credit_rank_raw.numpy() if data.credit_rank_raw is not None else np.nan,
        "y_credit": data.y_credit.numpy(),
        "is_train": data.train_mask.numpy().astype(int),
        "is_val": data.val_mask.numpy().astype(int),
        "is_test": data.test_mask.numpy().astype(int),
    })
    results_df = results_df.sort_values("risk_score")
    results_df.to_csv(os.path.join(OUTPUT_DIR, "enterprise_risk_ranking.csv"),
                      index=False, encoding="utf-8-sig")
    print(f"  [OK] enterprise_risk_ranking.csv ({N} 企业)")

    # 打印风险分统计
    print(f"\n  风险分统计:")
    print(f"    mean/median/std: {prob_credit.mean():.4f} / "
          f"{np.median(prob_credit):.4f} / {prob_credit.std():.4f}")
    print(f"    Top-10 最低风险: {prob_credit[sorted_idx[:10]]}")
    print(f"    Top-10 最高风险: {prob_credit[sorted_idx[-10:]]}")

    print("\n" + "=" * 60)
    print(f"  完成。Test AUC = {test_auc:.4f}, Precision@10 = {test_prec10:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    import numpy as np
    main()
