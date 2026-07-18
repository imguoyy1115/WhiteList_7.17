"""
================================================================================
风险传播计算 — 基于 DebtRank 框架（不参与梯度，纯确定性计算）
================================================================================
模型训练完成后调用，利用已学好的风险判断 + trade 边拓扑，迭代传播风险。

原理:
  供应链风险不是孤立的——一个企业的财务危机会通过应收/应付账款
  传导给交易对手。本模块采用类 DebtRank / 带重启的 PageRank 传播公式:

    r^(0)[i]   = sigmoid(logit_risk_i)           # 模型对企业 i 的初始风险判断
    r^(k+1)[i] = α * r^(0)[i] + (1-α) * Σ_{j∈N(i)} w[j→i] * r^(k)[j]

    α:     重启概率（模型自判断的信任度），越大越保守，默认 0.7
    w[j→i]: j 对 i 的风险传导权重 = i 对 j 的应收账款占比（欠得越多影响越大）

  保证收敛：α > 0 → 每次迭代都向初始估计回归，残差单调递减。

输出:
  outputs/risk_scores.csv  — 所有企业的风险分数 (0~1)
  outputs/risk_propagation_report.txt  — 传播效果报告

用法:
  python risk_propagation.py                    # 加载已保存的模型
  # 或在 main_v2.py 训练结束后自动调用
================================================================================
"""
import torch
import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import DEVICE, OUTPUT_DIR, SEED
from train import HyperHeteroModel, compute_losses

torch.manual_seed(SEED)


def build_trade_adjacency(edge_index_dict, num_enterprises, X_ent, device):
    """
    ==========================================================================
    从 trade 边构建稀疏邻接矩阵 + 风险传导权重 W。

    权重 w[j→i]:
      j 对 i 的风险影响程度。简化为出度归一化（等权重），
      如果后续有边级交易金额数据，可替换为应收金额占比归一化。

    返回:
      W: (N, N) 稀疏张量, W[i, j] = 风险从 j 传到 i 的权重
      weighted: bool, 是否使用交易金额加权
    ==========================================================================
    """
    trade_ei = edge_index_dict.get(("enterprise", "trade", "enterprise"))
    N = num_enterprises

    if trade_ei is None:
        print("  [警告] 未找到 trade 边，风险传播退化为仅模型自身判断")
        return None, False

    src = trade_ei[0].long()  # 边的源端
    dst = trade_ei[1].long()  # 边的目标端

    # ── 尝试用 SCF 交易金额做加权 ──
    # X_ent[:, 2] = AccountReceivable, X_ent[:, 0] = AccountPayable
    weighted = False
    if X_ent is not None:
        ar = torch.tensor(X_ent[:, 2], device=device).float().abs()  # 应收账款
        weighted = True

    # 构建有向邻接（无向 trade 边拆成两个方向，各自有权重）
    row_indices = []
    col_indices = []
    values = []

    if weighted:
        # 权重 w[j→i] ∝ i 的应收账款（i 作为供应商暴露在客户 j 的风险下）
        # 对每条无向边 (u, v):
        #   u → v: 权重 ∝ ar[v]（v 的应收账款来自 u 多少）
        #   v → u: 权重 ∝ ar[u]（u 的应收账款来自 v 多少）
        ar_np = X_ent[:, 2]
        for s, d in zip(src.cpu().numpy(), dst.cpu().numpy()):
            s, d = int(s), int(d)
            # s → d: 风险从 d 传到 s（s 暴露在 d 的违约风险下）权重 ∝ ar[s]
            w_sd = ar_np[s] if ar_np[s] > 0 else 1.0
            row_indices.append(s)
            col_indices.append(d)
            values.append(float(w_sd))
            # d → s: 风险从 s 传到 d
            w_ds = ar_np[d] if ar_np[d] > 0 else 1.0
            row_indices.append(d)
            col_indices.append(s)
            values.append(float(w_ds))
    else:
        # 等权重退化
        for s, d in zip(src.cpu().numpy(), dst.cpu().numpy()):
            s, d = int(s), int(d)
            row_indices.extend([s, d])
            col_indices.extend([d, s])
            values.extend([1.0, 1.0])

    # 行归一化（每行的权重和为 1）
    W = torch.sparse_coo_tensor(
        torch.tensor([row_indices, col_indices], dtype=torch.long),
        torch.tensor(values, dtype=torch.float32),
        (N, N)
    ).coalesce()

    # 行归一化：每行 / 行和（保证 Σ_j W[i,j] = 1）
    row_sum = torch.sparse.sum(W, dim=1).to_dense().clamp(min=1e-8)
    # 用索引做除法
    W_indices = W.indices()
    W_vals = W.values()
    W_vals = W_vals / row_sum[W_indices[0]]
    W = torch.sparse_coo_tensor(W_indices, W_vals, (N, N)).coalesce()

    print(f"  trade 邻接矩阵: {W._nnz()} 条有向边 (加权={weighted}), "
          f"密度 {W._nnz() / (N * N) * 100:.4f}%")

    return W.to(device), weighted


@torch.no_grad()
def compute_risk_propagation(model, data, alpha=0.7, K_max=100, tol=1e-6,
                             device=None, verbose=True):
    """
    ==========================================================================
    基于 DebtRank 框架的供应链风险传播。

    输入:
      model:  已训练好的 HyperHeteroModel
      data:   HeteroGraphData（包含 x_dict, edge_index_dict 等）
      alpha:  重启概率 (0~1), 越大越保守（信模型自身判断），默认 0.70
      K_max:  最大迭代步数
      tol:    收敛阈值（最大残差）
      device: 计算设备
      verbose: 是否打印收敛过程

    返回:
      r_final:    (N,) 传播后风险分数 ∈ [0, 1]
      r_model:    (N,) 模型原始风险分数（传播前）
      converged:  bool  是否收敛
      n_steps:    int   实际迭代步数
    ==========================================================================
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    data = data.to(device)

    # ── Step 1: 用训练好的模型获取初始风险估计 ──
    logit_w, logit_r, logit_g = model(
        data.x_dict, data.edge_index_dict, data.hyperedges,
        data.num_enterprises,
        x_struct=data.x_struct, x_missing=data.x_missing,
        struct_hint=data.struct_hint,
        x_seq=data.x_seq if hasattr(data, "x_seq") and data.x_seq is not None else None,
    )

    r_0 = torch.sigmoid(logit_r).squeeze(-1).to(device)  # (N,), 模型自身风险判断
    N = r_0.shape[0]

    # ── Step 2: 构建转移矩阵 ──
    X_ent_np = data.x_dict["enterprise"].cpu().numpy()
    W, weighted = build_trade_adjacency(
        data.edge_index_dict, N,
        X_ent_np, device
    )

    if W is None:
        return r_0.cpu(), r_0.cpu(), True, 0

    # ── Step 3: 迭代传播 ──
    r = r_0.clone()
    converged = False
    n_steps = 0

    for k in range(K_max):
        # r_new = α * r_0 + (1-α) * W^T @ r
        neighbor_risk = torch.sparse.mm(W, r.unsqueeze(-1)).squeeze(-1)
        r_new = alpha * r_0 + (1.0 - alpha) * neighbor_risk
        r_new = r_new.clamp(0.0, 1.0)

        diff = (r_new - r).abs().max().item()
        r = r_new
        n_steps = k + 1

        if verbose and (k < 3 or k % 10 == 0 or diff < tol * 100):
            print(f"    step {k+1:4d}:  max Δ = {diff:.8f}")

        if diff < tol:
            converged = True
            break

    if verbose:
        status = "收敛" if converged else f"达到上限 ({K_max} 步)"
        print(f"  风险传播完成: {status}, 共 {n_steps} 步, 最终 max Δ = {diff:.8f}")
        print(f"  风险分统计: mean={r.mean().item():.4f}, "
              f"median={r.median().item():.4f}, "
              f"std={r.std().item():.4f}")

    return r.cpu(), r_0.cpu(), converged, n_steps


def save_risk_scores(r_final, r_model, data, output_dir=None, converged=True, n_steps=0):
    """
    ==========================================================================
    保存所有企业的风险分数到 CSV 文件。

    输出列:
      enterprise_id:  企业全局索引
      risk_score:     传播后的风险分数 (0~1)
      risk_score_raw: 模型原始风险分数（传播前）
      risk_delta:     传播带来的变化 (risk_score - risk_score_raw)
      y_white:        白名单标签 (1=正常)
      y_risk:         风险标签 (1=有风险)
      is_listed:      是否上市公司
    ==========================================================================
    """
    if output_dir is None:
        output_dir = OUTPUT_DIR

    N = r_final.shape[0]
    nl = data.num_listed

    # ── 构建 DataFrame ──
    df = pd.DataFrame({
        "enterprise_id": np.arange(N),
        "risk_score": r_final.numpy(),
        "risk_score_raw": r_model.numpy(),
        "risk_delta": (r_final - r_model).numpy(),
        "y_white": data.y_white.cpu().numpy(),
        "y_risk": data.y_risk.cpu().numpy(),
        "is_listed": np.array([1 if i < nl else 0 for i in range(N)]),
        "is_train": data.train_mask.cpu().numpy().astype(int),
        "is_val": data.val_mask.cpu().numpy().astype(int),
        "is_test": data.test_mask.cpu().numpy().astype(int),
    })

    # 标记传播前后风险等级变化
    df["risk_level_before"] = pd.cut(
        df["risk_score_raw"], bins=[0, 0.2, 0.5, 0.8, 1.0],
        labels=["低风险", "中低风险", "中高风险", "高风险"]
    )
    df["risk_level_after"] = pd.cut(
        df["risk_score"], bins=[0, 0.2, 0.5, 0.8, 1.0],
        labels=["低风险", "中低风险", "中高风险", "高风险"]
    )

    csv_path = os.path.join(output_dir, "risk_scores.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\n  [OK] 风险分数已保存: {csv_path}")

    # ── 报告 ──
    report_path = os.path.join(output_dir, "risk_propagation_report.txt")
    test_mask = data.test_mask.cpu().numpy()
    y_test = data.y_white.cpu().numpy()[test_mask]
    r_test = r_final.numpy()[test_mask]
    r_raw_test = r_model.numpy()[test_mask]

    from sklearn.metrics import roc_auc_score
    auc_after = roc_auc_score(y_test, r_test) if y_test.sum() > 0 else 0.5
    auc_before = roc_auc_score(y_test, r_raw_test) if y_test.sum() > 0 else 0.5

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("  供应链风险传播报告 (DebtRank)\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"  总企业数:           {N}\n")
        f.write(f"  上市公司 (labeled): {nl}\n")
        f.write(f"  非上市企业:         {N - nl}\n\n")
        f.write(f"  传播状态:           {'收敛' if converged else '未收敛'}\n")
        f.write(f"  迭代步数:           {n_steps}\n\n")
        f.write(f"  ── 测试集 AUC 对比 ──\n")
        f.write(f"  传播前 (模型原始):   {auc_before:.4f}\n")
        f.write(f"  传播后 (DebtRank):   {auc_after:.4f}\n")
        f.write(f"  Δ AUC:               {auc_after - auc_before:+.4f}\n\n")
        f.write(f"  ── 全局风险分统计 ──\n")
        f.write(f"  传播前 mean / median / std: "
                f"{r_model.mean().item():.4f} / {r_model.median().item():.4f} / {r_model.std().item():.4f}\n")
        f.write(f"  传播后 mean / median / std: "
                f"{r_final.mean().item():.4f} / {r_final.median().item():.4f} / {r_final.std().item():.4f}\n\n")
        f.write(f"  风险等级变化:\n")
        level_change = df["risk_level_before"] != df["risk_level_after"]
        n_changed = level_change.sum()
        f.write(f"  等级变化企业数:     {n_changed} / {N} ({n_changed/N*100:.2f}%)\n")

    print(f"  [OK] 传播报告已保存: {report_path}")

    return df


def run_risk_propagation_pipeline(model, data, alpha=0.7, K_max=100, tol=1e-6):
    """
    ==========================================================================
    端到端风险传播流程: 模型推理 → 传播计算 → 保存结果。

    供 main_v2.py 训练结束后直接调用。
    ==========================================================================
    """
    device = next(model.parameters()).device
    print("\n" + "=" * 60)
    print("  供应链风险传播 (DebtRank)")
    print("=" * 60)
    print(f"  α = {alpha}, K_max = {K_max}, tol = {tol}")

    r_final, r_model, converged, n_steps = compute_risk_propagation(
        model, data,
        alpha=alpha, K_max=K_max, tol=tol,
        device=device, verbose=True,
    )

    df = save_risk_scores(r_final, r_model, data,
                          converged=converged, n_steps=n_steps)

    return r_final, r_model, df


# ============================================================================
# 独立运行入口
# ============================================================================
if __name__ == "__main__":
    import time
    from data_loader.csmar_loader import load_csmar_data_v5
    from train import evaluate_model

    print("=" * 60)
    print("  独立运行: 风险传播计算")
    print("=" * 60)

    # ── Step 1: 加载数据 ──
    print("\n[Step 1] 加载数据...")
    t0 = time.time()
    data = load_csmar_data_v5()
    print(f"  加载完成，耗时 {time.time() - t0:.1f}s")

    # ── Step 2: 加载训练好的模型 ──
    print("\n[Step 2] 加载模型...")
    from config import EDGE_TYPES

    in_dims = {ntype: data.x_dict[ntype].shape[1] for ntype in data.x_dict}
    valid_edge_types = [et for et in EDGE_TYPES if et in data.edge_index_dict]
    if valid_edge_types != EDGE_TYPES:
        missing = set(EDGE_TYPES) - set(valid_edge_types)
        print(f"  注意: 以下边类型缺失，跳过: {missing}")

    model = HyperHeteroModel(in_dims=in_dims, edge_types=valid_edge_types)

    model_path = os.path.join(OUTPUT_DIR, "model_v2.pt")
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=DEVICE,
                                         weights_only=True))
        print(f"  模型已加载: {model_path}")
    else:
        print(f"  [错误] 模型文件不存在: {model_path}")
        print(f"  请先运行 main_v2.py 训练模型")
        sys.exit(1)

    # ── Step 3: 评估模型原始表现 ──
    print("\n[Step 3] 模型原始评估...")
    model = model.to(DEVICE)
    data = data.to(DEVICE)
    test_auc, test_acc, test_prec10 = evaluate_model(model, data, data.test_mask)
    print(f"  测试 AUC: {test_auc:.4f}, Acc: {test_acc:.4f}, P@10: {test_prec10:.4f}")

    # ── Step 4: 运行风险传播 ──
    print("\n[Step 4] 风险传播...")
    r_final, r_model, df = run_risk_propagation_pipeline(
        model, data, alpha=0.70, K_max=100, tol=1e-6
    )

    print("\n" + "=" * 60)
    print("  完成。")
    print("=" * 60)
