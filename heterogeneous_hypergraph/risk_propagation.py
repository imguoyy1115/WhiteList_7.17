"""
================================================================================
风险传播计算 — 基于 SCF 打分卡 + DebtRank 框架（不参与梯度，纯确定性计算）
================================================================================
模型训练完成后调用，利用 SCF 打分卡风险分数 + trade 边拓扑，迭代传播风险。

原理:
  r^(0)[i]   = risk_scf[i]                  # SCF 打分卡（四维度加权）
                  ┌ 财务健康 (30%): CR + DAR + ICR
                  ├ 供应链地位 (25%): BankLoanSize + SupplierPower1
                  ├ 诉讼合规 (25%): 诉讼金额 + 诉讼严重程度
                  └ 关联质量 (20%): 上市/非上市

  r^(k+1)[i] = α * r^(0)[i] + (1-α) * Σ_{j∈N(i)} w[j→i] * r^(k)[j]

  α:     重启概率，越大越保守（回归 SCF 自身分），默认 0.7
  w[j→i]: j 对 i 的风险传导权重（基于应收账款归一化）

  保证收敛：α > 0 → 每次迭代都向初始估计回归，残差单调递减。

  若 data.risk_scf 不可用，退化为模型 sigmoid(logit_risk) 作为 r^(0)。

输出:
  outputs/risk_scores.csv           — 所有企业的风险分数 (0~1)
  outputs/risk_propagation_report.txt  — 传播效果报告

用法:
  python risk_propagation.py                    # 加载数据 + 已保存模型
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
from train import HyperHeteroModel

torch.manual_seed(SEED)


def build_trade_adjacency(edge_index_dict, num_enterprises, X_ent, device):
    """
    ==========================================================================
    从 trade 边构建稀疏邻接矩阵 + 风险传导权重 W。

    权重 w[j→i]:
      j 对 i 的风险影响程度。用应收账款金额归一化加权，
      非上市企业退化为度归一化（等权重）。

    返回:
      W: (N, N) 稀疏张量, W[i, j] = 风险从 j 传到 i 的权重
      weighted: bool, 是否使用交易金额加权
    ==========================================================================
    """
    trade_ei = edge_index_dict.get(("enterprise", "trade", "enterprise"))
    N = num_enterprises

    if trade_ei is None:
        print("  [警告] 未找到 trade 边，风险传播退化为仅 SCF 打分自身判断")
        return None, False

    src = trade_ei[0].long()
    dst = trade_ei[1].long()

    # ── 用应收账款金额做加权 ──
    weighted = False
    if X_ent is not None and X_ent.shape[1] > 2:
        weighted = True

    row_indices = []
    col_indices = []
    values = []

    if weighted:
        ar = X_ent[:, 2]  # AccountReceivable (归一化后的, 相对大小仍有效)
        ar_abs = np.abs(ar)
        for s, d in zip(src.cpu().numpy(), dst.cpu().numpy()):
            s, d = int(s), int(d)
            w_sd = float(ar_abs[s] + 1e-6)  # s 对 d 的应收 → d 风险传向 s
            w_ds = float(ar_abs[d] + 1e-6)  # d 对 s 的应收 → s 风险传向 d
            row_indices.extend([s, d])
            col_indices.extend([d, s])
            values.extend([w_sd, w_ds])
    else:
        for s, d in zip(src.cpu().numpy(), dst.cpu().numpy()):
            s, d = int(s), int(d)
            row_indices.extend([s, d])
            col_indices.extend([d, s])
            values.extend([1.0, 1.0])

    W = torch.sparse_coo_tensor(
        torch.tensor([row_indices, col_indices], dtype=torch.long),
        torch.tensor(values, dtype=torch.float32),
        (N, N)
    ).coalesce()

    # 行归一化：每行权重和为 1
    row_sum = torch.sparse.sum(W, dim=1).to_dense().clamp(min=1e-8)
    W_indices = W.indices()
    W_vals = W.values()
    W_vals = W_vals / row_sum[W_indices[0]]
    W = torch.sparse_coo_tensor(W_indices, W_vals, (N, N)).coalesce()

    print(f"  trade 邻接矩阵: {W._nnz()} 条有向边 (加权={'是' if weighted else '否'}), "
          f"密度 {W._nnz() / (N * N) * 100:.4f}%")

    return W.to(device), weighted


@torch.no_grad()
def compute_risk_propagation(model=None, data=None, alpha=0.7, K_max=100, tol=1e-6,
                             device=None, verbose=True):
    """
    ==========================================================================
    SCF 打分卡 + DebtRank 传播。

    输入:
      model:  已训练好的 HyperHeteroModel（可选，data.risk_scf 可用时可不传）
      data:   HeteroGraphData（需包含 risk_scf 字段或可用模型推理）
      alpha:  重启概率 (0~1), 默认 0.70
      K_max:  最大迭代步数
      tol:    收敛阈值（最大残差）
      device: 计算设备
      verbose: 是否打印收敛过程

    返回:
      r_final:    (N,) 传播后风险分数 ∈ [0, 1]
      r_init:     (N,) SCF 打分卡原始分数（传播前，即 r^(0)）
      converged:  bool  是否收敛
      n_steps:    int   实际迭代步数
    ==========================================================================
    """
    if data is None:
        raise ValueError("需要传入 data (HeteroGraphData)")

    if device is None and model is not None:
        device = next(model.parameters()).device
    elif device is None:
        device = torch.device("cpu")

    data = data.to(device)

    # ── Step 1: 获取初始风险估计 r^(0) ──
    has_scf = hasattr(data, "risk_scf") and data.risk_scf is not None
    if has_scf:
        r_0 = data.risk_scf.float().to(device)
        if verbose:
            print(f"  初始风险分来源: SCF 打分卡 (四维度加权, n={r_0.shape[0]})")
            print(f"    维度: 财务健康(30%) + 供应链地位(25%) + 诉讼合规(25%) + 关联质量(20%)")
    elif model is not None:
        model.eval()
        logit_w, logit_r, logit_g = model(
            data.x_dict, data.edge_index_dict, data.hyperedges,
            data.num_enterprises,
            x_struct=data.x_struct, x_missing=data.x_missing,
            struct_hint=data.struct_hint,
            x_seq=data.x_seq if hasattr(data, "x_seq") and data.x_seq is not None else None,
        )
        r_0 = torch.sigmoid(logit_r).squeeze(-1).to(device)
        if verbose:
            print(f"  初始风险分来源: 模型 sigmoid(logit_risk) (退化为无 SCF 分数)")
    else:
        raise ValueError("需要 data.risk_scf 或 model 来获取初始风险估计")

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
        # r_new = α * r_0 + (1-α) * W @ r
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
        print(f"  风险分统计 (传播前→后):")
        print(f"    mean:   {r_0.mean().item():.4f} → {r.mean().item():.4f}")
        print(f"    median: {r_0.median().item():.4f} → {r.median().item():.4f}")
        print(f"    std:    {r_0.std().item():.4f} → {r.std().item():.4f}")

    return r.cpu(), r_0.cpu(), converged, n_steps


def save_risk_scores(r_final, r_init, data, output_dir=None, converged=True, n_steps=0):
    """
    ==========================================================================
    保存所有企业的风险分数到 CSV + 生成传播报告。

    输出列:
      enterprise_id:    企业全局索引
      risk_score:       传播后的风险分数 (0~1)
      risk_score_init:  SCF 打分卡原始分数（传播前，r^(0)）
      risk_delta:       传播带来的变化
      y_white / y_risk: 真实标签
      is_listed:        是否上市公司
      is_train/val/test: 数据集切分
    ==========================================================================
    """
    if output_dir is None:
        output_dir = OUTPUT_DIR

    N = r_final.shape[0]
    nl = data.num_listed

    df = pd.DataFrame({
        "enterprise_id": np.arange(N),
        "risk_score": r_final.numpy(),
        "risk_score_init": r_init.numpy(),
        "risk_delta": (r_final - r_init).numpy(),
        "y_white": data.y_white.cpu().numpy(),
        "y_risk": data.y_risk.cpu().numpy(),
        "is_listed": np.array([1 if i < nl else 0 for i in range(N)]),
        "is_train": data.train_mask.cpu().numpy().astype(int),
        "is_val": data.val_mask.cpu().numpy().astype(int),
        "is_test": data.test_mask.cpu().numpy().astype(int),
    })

    # 风险等级标记
    df["risk_level_init"] = pd.cut(
        df["risk_score_init"], bins=[0, 0.2, 0.5, 0.8, 1.0],
        labels=["低风险", "中低风险", "中高风险", "高风险"]
    )
    df["risk_level_final"] = pd.cut(
        df["risk_score"], bins=[0, 0.2, 0.5, 0.8, 1.0],
        labels=["低风险", "中低风险", "中高风险", "高风险"]
    )

    csv_path = os.path.join(output_dir, "risk_scores.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\n  [OK] 风险分数已保存: {csv_path}")

    # ── 传播报告 ──
    report_path = os.path.join(output_dir, "risk_propagation_report.txt")
    test_mask = data.test_mask.cpu().numpy()
    y_test = data.y_white.cpu().numpy()[test_mask]
    r_test = r_final.numpy()[test_mask]
    r_init_test = r_init.numpy()[test_mask]

    from sklearn.metrics import roc_auc_score
    auc_after = roc_auc_score(y_test, r_test) if y_test.sum() > 0 else 0.5
    auc_before = roc_auc_score(y_test, r_init_test) if y_test.sum() > 0 else 0.5

    # 检查来源
    has_scf = hasattr(data, "risk_scf") and data.risk_scf is not None
    source_str = "SCF 打分卡（四维度: 财务健康30% + 供应链地位25% + 诉讼合规25% + 关联质量20%）" \
                 if has_scf else "模型 sigmoid(logit_risk)"

    level_change = df["risk_level_init"] != df["risk_level_final"]
    n_changed = level_change.sum()

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("  供应链风险传播报告 (SCF打分卡 + DebtRank)\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"  初始风险分来源:     {source_str}\n")
        f.write(f"  总企业数:           {N}\n")
        f.write(f"  上市公司 (labeled): {nl}\n")
        f.write(f"  非上市企业:         {N - nl}\n\n")
        f.write(f"  传播状态:           {'收敛' if converged else '未收敛'}\n")
        f.write(f"  迭代步数:           {n_steps}\n\n")
        f.write(f"  ── 测试集 AUC 对比（白名单识别） ──\n")
        f.write(f"  传播前 (SCF原始分):  {auc_before:.4f}\n")
        f.write(f"  传播后 (DebtRank):   {auc_after:.4f}\n")
        f.write(f"  Δ AUC:               {auc_after - auc_before:+.4f}\n\n")
        f.write(f"  ── 全局风险分统计 ──\n")
        f.write(f"  传播前 mean / median / std: "
                f"{r_init.mean().item():.4f} / {r_init.median().item():.4f} / {r_init.std().item():.4f}\n")
        f.write(f"  传播后 mean / median / std: "
                f"{r_final.mean().item():.4f} / {r_final.median().item():.4f} / {r_final.std().item():.4f}\n\n")
        f.write(f"  风险等级变化:\n")
        f.write(f"  等级变化企业数:     {n_changed} / {N} ({n_changed/N*100:.2f}%)\n")

    print(f"  [OK] 传播报告已保存: {report_path}")

    return df


def run_risk_propagation_pipeline(model, data, alpha=0.7, K_max=100, tol=1e-6):
    """
    ==========================================================================
    端到端风险传播流程: 获取初始分 → 传播计算 → 保存结果。

    供 main_v2.py 训练结束后直接调用。
    ==========================================================================
    """
    device = next(model.parameters()).device if model is not None else torch.device("cpu")
    print("\n" + "=" * 60)
    print("  供应链风险传播 (SCF 打分卡 + DebtRank)")
    print("=" * 60)
    print(f"  α = {alpha}, K_max = {K_max}, tol = {tol}")

    r_final, r_init, converged, n_steps = compute_risk_propagation(
        model, data,
        alpha=alpha, K_max=K_max, tol=tol,
        device=device, verbose=True,
    )

    df = save_risk_scores(r_final, r_init, data,
                          converged=converged, n_steps=n_steps)

    return r_final, r_init, df


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

    # ── Step 1: 加载数据（自带 risk_scf 字段） ──
    print("\n[Step 1] 加载数据...")
    t0 = time.time()
    data = load_csmar_data_v5()
    print(f"  加载完成，耗时 {time.time() - t0:.1f}s")

    # ── Step 2: 加载训练好的模型 （用于获取 AUC 基线，传播本身不需要模型） ──
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
        print(f"  [警告] 模型文件不存在: {model_path}")
        print(f"  将仅使用 SCF 打分卡分数进行传播（无需模型）")
        model = None

    # ── Step 3: 评估模型原始表现（如果有模型） ──
    if model is not None:
        print("\n[Step 3] 模型原始评估...")
        model = model.to(DEVICE)
        data = data.to(DEVICE)
        test_auc, test_acc, test_prec10 = evaluate_model(model, data, data.test_mask)
        print(f"  测试 AUC: {test_auc:.4f}, Acc: {test_acc:.4f}, P@10: {test_prec10:.4f}")

    # ── Step 4: 运行风险传播 ──
    print("\n[Step 4] 风险传播 (SCF 打分卡 + DebtRank)...")
    r_final, r_init, df = run_risk_propagation_pipeline(
        model, data, alpha=0.70, K_max=100, tol=1e-6
    )

    print("\n" + "=" * 60)
    print("  完成。")
    print("=" * 60)
