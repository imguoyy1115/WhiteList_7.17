"""
================================================================================
XGBoost Baseline — 使用 v5 数据管线 (csmar_loader)，纯表格二分类
================================================================================

对标 heterogeneous_hypergraph GNN 的纯表格 baseline：
  - 仅使用原始企业特征 X（13 维），不做任何图消息传递
  - 与 GNN 使用完全相同的 train/val/test 切分
  - 不包含图结构信息（无邻域聚合、无超图、无消息传递）

对比逻辑：
  GNN AUC - XGBoost AUC = 图结构 + 异构通道带来的真实增益
  （XGBoost 拿不到图，GNN 如果赢了才是图的价值）

特征说明（11 维，v5.4 财务特征已移入 x_seq）：
  col 0-7:  SCF 贸易信用（8 维）
  col 8-9:  诉讼（金额 + 严重程度）
  col 10:    预留位

不含 x_seq、x_struct、struct_hint 等任何包含图/时序信息的特征，
确保 baseline 是纯表格单企业视角。

用法：
  cd heterogeneous_hypergraph
  python xgboost_baseline.py

对比（在 main_v2.py 等脚本中查看）:
  main_v2_no_temporal.py (特征分工 + FinMLP):      0.8086
  main_v2.py             (特征分工 + FinGRU):        0.8074
  xgboost_baseline.py    (纯表格 XGBoost):          ???
================================================================================
"""

import numpy as np
import pandas as pd
import sys
import os
import time
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_loader.csmar_loader import load_csmar_data_v5

from sklearn.metrics import (
    roc_auc_score, accuracy_score,
    precision_score, recall_score, f1_score,
    confusion_matrix,
)
import xgboost as xgb

# ============================================================================
# 工具函数
# ============================================================================
def precision_at_k(y_true, y_score, k: int = 10):
    """Precision@K: 模型打分最高的 K 个样本中正类的比例"""
    if len(y_score) == 0:
        return 0.0
    k = min(k, len(y_score))
    top_k_idx = np.argsort(y_score)[-k:]
    return float(y_true[top_k_idx].sum()) / k


# ============================================================================
# 全局配置
# ============================================================================
SEED = 42
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "xgboost_outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# XGBoost 超参数
XGB_PARAMS = {
    "n_estimators": 500,
    "max_depth": 6,
    "learning_rate": 0.03,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "min_child_weight": 5,
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "random_state": SEED,
    "early_stopping_rounds": 50,
}


# ============================================================================
# Step 1: 加载 v5 数据，构建表格特征
# ============================================================================
def build_tabular_features():
    """
    调用 v5 数据管线 (csmar_loader.load_csmar_data_v5)，
    提取原始企业特征 X_ent (N, 13)，不做任何图处理。
    """
    print("=" * 60)
    print("  XGBoost Baseline — v5 数据管线 (纯表格，无图)")
    print("=" * 60)

    print("\n[Step 1] 加载 CSMAR 数据 (v5 管线)...")
    t0 = time.time()
    data = load_csmar_data_v5()
    print(f"  加载完成，耗时 {time.time() - t0:.1f}s")

    # ── 仅使用原始企业特征 ──
    #     x_dict["enterprise"] = (N, 13)
    #     SCF(8) + 营收增长率 + 资产周转率 + 诉讼(2) + 预留(1)
    X = data.x_dict["enterprise"].numpy().astype(np.float32)
    n_features = X.shape[1]

    # ── 标签 & Mask（与 GNN 完全一致的切分） ──
    y = data.y_white.numpy().astype(int)
    train_mask = data.train_mask.numpy().astype(bool)
    val_mask = data.val_mask.numpy().astype(bool)
    test_mask = data.test_mask.numpy().astype(bool)

    # ── 特征名（13 维，与 v5 config 对齐） ──
    feature_names = [
        "SCF_AccountPayable",         # 0  应付账款
        "SCF_Prepayment",             # 1  预付账款
        "SCF_AccountReceivable",      # 2  应收账款
        "SCF_TotalAssets",            # 3  总资产
        "SCF_ProvidedTradeCredit",    # 4  提供的贸易信用
        "SCF_ObtainedTradeCredit",    # 5  获得的贸易信用
        "SCF_BankLoanSize",           # 6  银行贷款规模
        "SCF_SupplierPower1",         # 7  供应商议价力
        "Lawsuit_TotalAmount_log1p",  # 8  诉讼累计金额(log1p)
        "Lawsuit_WeightedSeverity",   # 9  诉讼加权严重分(log1p)
        "Reserved",                   # 10 预留位
    ]

    print(f"  特征维度: {n_features} (纯原始企业特征，无图结构信息)")
    print(f"  训练集: {train_mask.sum()} | 验证集: {val_mask.sum()} | 测试集: {test_mask.sum()}")
    print(f"  训练集正样本比例: {y[train_mask].mean():.3f}")
    print(f"  测试集正样本比例: {y[test_mask].mean():.3f}")

    return X, y, train_mask, val_mask, test_mask, feature_names


# ============================================================================
# Step 2: XGBoost 训练
# ============================================================================
def train_xgboost(X_train, y_train, X_val, y_val):
    """二分类 XGBoost，使用验证集做早停。"""
    print(f"\n[Step 2] XGBoost 训练...")

    # 类别不平衡：自动计算权重
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0

    params = XGB_PARAMS.copy()
    params["scale_pos_weight"] = scale_pos_weight
    early_stop = params.pop("early_stopping_rounds", 50)

    print(f"  训练样本: {len(X_train)}, 正样本比例: {y_train.mean():.3f}")
    print(f"  scale_pos_weight: {scale_pos_weight:.2f}")
    print(f"  max_depth: {XGB_PARAMS['max_depth']}, lr: {XGB_PARAMS['learning_rate']}")
    print(f"  n_estimators: {XGB_PARAMS['n_estimators']}, early_stopping: {early_stop}")

    t0 = time.time()
    model = xgb.XGBClassifier(
        early_stopping_rounds=early_stop,
        verbosity=0,
        **params,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    elapsed = time.time() - t0

    # ── 从 evals_result() 中提取真实的验证集 AUC 历史 ──
    #     model.best_score / best_iteration 在某些 xgboost 版本中
    #     报告的是最终 epoch 的值而非历史峰值，因此手动计算
    evals = model.evals_result()
    val_key = list(evals.keys())[0]  # "validation_0"
    val_auc_history = evals[val_key]["auc"]

    best_idx = int(np.argmax(val_auc_history))  # 真正的最优迭代位置
    best_val_auc = float(val_auc_history[best_idx])

    print(f"  训练完成，耗时 {elapsed:.1f}s")
    print(f"  总迭代数: {len(val_auc_history)}, 早停轮次: {early_stop}")
    print(f"  最佳迭代: {best_idx + 1}  (总计 {len(val_auc_history)} 轮)")
    print(f"  最佳验证 AUC: {best_val_auc:.4f}")
    print(f"  最终验证 AUC: {val_auc_history[-1]:.4f} (最后 {early_stop} 轮未提升时触发早停)")

    return model, best_val_auc, best_idx


# ============================================================================
# Step 3: 评估
# ============================================================================
def evaluate(model, X_train, y_train, X_val, y_val, X_test, y_test, feature_names,
             best_val_auc=None, best_idx=None):
    """二分类评估：AUC + Accuracy + F1 + Precision@10 + 特征重要性"""
    print(f"\n[Step 3] 评估...")
    print("=" * 60)

    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)

    # ── 测试集指标 ──
    test_auc = roc_auc_score(y_test, y_pred_proba)
    test_acc = accuracy_score(y_test, y_pred)
    test_precision = precision_score(y_test, y_pred, zero_division=0)
    test_recall = recall_score(y_test, y_pred, zero_division=0)
    test_f1 = f1_score(y_test, y_pred, zero_division=0)
    test_prec10 = precision_at_k(y_test, y_pred_proba, k=10)

    print(f"  ── 测试集 ──")
    print(f"  AUC:          {test_auc:.4f}")
    print(f"  Accuracy:     {test_acc:.4f}")
    print(f"  Precision:    {test_precision:.4f}")
    print(f"  Recall:       {test_recall:.4f}")
    print(f"  F1:           {test_f1:.4f}")
    print(f"  Precision@10: {test_prec10:.4f}")

    # ── 训练集 & 验证集 AUC（辅助诊断过拟合） ──
    train_proba = model.predict_proba(X_train)[:, 1]
    val_proba = model.predict_proba(X_val)[:, 1]
    train_auc = roc_auc_score(y_train, train_proba)
    val_auc_current = roc_auc_score(y_val, val_proba)
    print(f"\n  ── 辅助诊断 ──")
    print(f"  训练 AUC:           {train_auc:.4f}")
    print(f"  验证 AUC (历史峰值): {best_val_auc:.4f}" if best_val_auc else f"  验证 AUC:           {val_auc_current:.4f}")
    print(f"  验证 AUC (最终模型): {val_auc_current:.4f}")
    print(f"  测试 AUC:           {test_auc:.4f}")
    print(f"  Train-Test Gap:     {train_auc - test_auc:.4f}")

    # ── 混淆矩阵 ──
    print(f"\n  ── 混淆矩阵 ──")
    cm = confusion_matrix(y_test, y_pred)
    print(f"             预测负类  预测正类")
    print(f"  实际负类  {cm[0,0]:>10d}  {cm[0,1]:>10d}")
    print(f"  实际正类  {cm[1,0]:>10d}  {cm[1,1]:>10d}")

    # ── 特征重要性 Top-15 ──
    print(f"\n  ── 特征重要性 Top-15 (gain) ──")
    importance = model.get_booster().get_score(importance_type="gain")
    sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:15]
    for rank, (fname, score) in enumerate(sorted_imp, 1):
        idx = int(fname.replace("f", ""))
        real_name = feature_names[idx] if idx < len(feature_names) else fname
        print(f"  {rank:>2}. {real_name:<40s} {score:>10.2f}")

    results = {
        "Test_AUC": test_auc,
        "Test_Accuracy": test_acc,
        "Test_Precision": test_precision,
        "Test_Recall": test_recall,
        "Test_F1": test_f1,
        "Precision_at_10": test_prec10,
        "Train_AUC": train_auc,
        "Val_AUC_Best_History": best_val_auc if best_val_auc else val_auc_current,
        "Val_AUC_Current": val_auc_current,
        "Best_Iteration": best_idx + 1 if best_idx is not None else None,
        "Train_Pos_Ratio": y_train.mean(),
        "Test_Pos_Ratio": y_test.mean(),
    }
    return results, y_pred_proba


# ============================================================================
# Step 4: 保存
# ============================================================================
def save_results(results, model, feature_names, X, y, test_mask, y_pred_proba):
    """保存指标、模型、特征重要性、预测结果"""
    print(f"\n[Step 4] 保存结果到 {OUTPUT_DIR}/ ...")

    pd.DataFrame([results]).to_csv(
        f"{OUTPUT_DIR}/metrics.csv", index=False, encoding="utf-8-sig"
    )
    model.save_model(f"{OUTPUT_DIR}/xgboost_model.json")

    imp = model.get_booster().get_score(importance_type="gain")
    imp_df = pd.DataFrame([
        {"feature": feature_names[int(k.replace("f", ""))], "gain": v}
        for k, v in imp.items()
    ]).sort_values("gain", ascending=False)
    imp_df.to_csv(
        f"{OUTPUT_DIR}/feature_importance.csv", index=False, encoding="utf-8-sig"
    )

    test_idx = np.where(test_mask)[0]
    pred_df = pd.DataFrame({
        "global_id": test_idx,
        "y_true": y[test_idx].astype(int),
        "y_proba": y_pred_proba,
        "y_pred": (y_pred_proba >= 0.5).astype(int),
    })
    pred_df.to_csv(
        f"{OUTPUT_DIR}/test_predictions.csv", index=False, encoding="utf-8-sig"
    )

    print(f"  [OK] metrics.csv  [OK] xgboost_model.json")
    print(f"  [OK] feature_importance.csv  [OK] test_predictions.csv")


# ============================================================================
# 主流程
# ============================================================================
if __name__ == "__main__":
    # ── Step 1: 加载数据 ──
    X, y, train_mask, val_mask, test_mask, feature_names = build_tabular_features()

    # ── Step 2: 按 mask 切分 ──
    X_train = X[train_mask]
    y_train = y[train_mask]
    X_val = X[val_mask]
    y_val = y[val_mask]
    X_test = X[test_mask]
    y_test = y[test_mask]

    # ── Step 3: 训练 ──
    model, best_val_auc, best_idx = train_xgboost(X_train, y_train, X_val, y_val)

    # ── Step 4: 评估 ──
    results, y_pred_proba = evaluate(
        model, X_train, y_train, X_val, y_val, X_test, y_test, feature_names,
        best_val_auc=best_val_auc, best_idx=best_idx,
    )

    # ── Step 5: 保存 ──
    save_results(results, model, feature_names, X, y, test_mask, y_pred_proba)

    # ════════════════════════════════════════════════════════════════
    # GNN vs XGBoost 对比
    # ════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  GNN (v5.2) vs XGBoost 对比")
    print("=" * 60)
    print(f"              Test AUC    Val AUC")
    print(f"  XGBoost     {results['Test_AUC']:.4f}      {results['Val_AUC_Best_History']:.4f}   (纯表格 13维，无图)")
    print(f"  GNN v5.3    0.8086      0.7872   (main_v2_no_temporal, 特征分工 + FinMLP)")
    print(f"  GNN v5.3    0.8074      0.7875   (main_v2, 特征分工 + FinGRU)")
    print(f"  ─────────────────────────────────")
    gap = 0.8086 - results['Test_AUC']
    if gap > 0:
        print(f"  Δ (GNN增益)  {gap:+.4f}  ← GNN 学到了图结构的额外信息")
    else:
        print(f"  Δ (GNN增益)  {gap:+.4f}  ← GNN 未超越纯表格 baseline，需排查")
    print(f"\n  说明:")
    print(f"    XGBoost: 仅 X_ent (11 维纯结构特征)")
    print(f"    GNN:      超图4视图 + 异构双通道 + FusionGate")
    print(f"              + X_ent(11维) → 超图 | x_seq(12维) → FinTemporalEncoder → 异构")
    print(f"    切分:     复用 v5 数据管线的 train/val/test mask（完全对齐）")
    print(f"    含义:     {'GNN 的图结构对白名单识别有正向价值' if gap > 0 else '当前 GNN 架构未充分挖掘图信息'}")
    print("=" * 60)
