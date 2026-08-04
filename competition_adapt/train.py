"""
================================================================================
训练循环 — 赛题适配版
================================================================================
简化自 heterogeneous_hypergraph/train.py。

与原版的关键区别:
  - 去掉 FinTemporalEncoder（无财务时序）
  - 去掉 FeatureGate（无财务指标门控需求）
  - 去掉 HeteroChannelEncoder（无异构图）
  - 新增 EnterpriseEncoder（企业属性 MLP）
  - 模型: HyperEncoder + EnterpriseEncoder → FusionGate → Heads
  - 损失: L_credit + λ_grade * L_grade + λ_struct * L_struct
================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import roc_auc_score, accuracy_score
import time
import copy
import gc

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    DEVICE, EPOCHS, LR, LR_HYPER, WEIGHT_DECAY, EARLY_STOP_PATIENCE, SEED,
    LAMBDA_RISK, LAMBDA_GRADE, LAMBDA_STRUCT,
    HIDDEN_DIM, HYPER_HIDDEN, FUSION_HIDDEN, USE_AMP, DROPOUT,
    ENT_FEAT_DIM,
)
from hypergraph.hypergraph_conv import MultiViewHyperEncoder
from encoder.enterprise_encoder import EnterpriseEncoder
from fusion.fusion_gate import FusionGate
from classifier import MultiTaskHeads


class CompetitionModel(nn.Module):
    """
    ==========================================================================
    赛题适配模型: 多视角超图 + 企业属性编码 → 融合预测

    数据流:
      x_ent (N, 24)  企业静态属性
        ├─→ MultiViewHyperEncoder → h_struct (N, 128)   结构角色
        └─→ EnterpriseEncoder(MLP) → h_attr (N, 128)    属性编码

      FusionGate(h_struct, h_attr, struct_hint) → h_fusion (N, 64)
        → PostProj → z_v (N, 64)
        → MultiTaskHeads → logit_credit, logit_grade
    ==========================================================================
    """

    def __init__(self, ent_dim: int = ENT_FEAT_DIM):
        super().__init__()

        # ── 超图通道: 企业特征 → 多视图超图卷积 → 结构表示 ──
        self.hyper_encoder = MultiViewHyperEncoder(
            in_dim=ent_dim,
            hidden=HYPER_HIDDEN,
        )

        # ── 属性通道: 企业特征 → MLP → 属性编码 ──
        self.ent_encoder = EnterpriseEncoder(
            in_dim=ent_dim,
            hidden=HIDDEN_DIM,
            out_dim=HIDDEN_DIM,
        )

        # ── 双通道融合 ──
        self.fusion_gate = FusionGate(
            struct_dim=HYPER_HIDDEN,
            attr_dim=HIDDEN_DIM,
            hidden=FUSION_HIDDEN,
            hint_dim=8,
        )

        # ── 融合后 MLP 投影 ──
        self.post_proj = nn.Sequential(
            nn.Linear(FUSION_HIDDEN, 64),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(64, 64),
        )

        # ── 预测头 ──
        self.heads = MultiTaskHeads(in_dim=64)

        # ── 统计 ──
        self._last_gate_mean = 0.0

    def forward(self, x_ent: torch.Tensor,
                edge_index_dict: dict, hyperedges: dict,
                struct_hint: torch.Tensor = None):
        """
        ==========================================================================
        完整前向传播（AMP 友好）。

        输入:
          x_ent:            (N, 24)  企业静态特征
          edge_index_dict:  边表（仅用于可能的后续扩展，当前不直接使用）
          hyperedges:       5 视图超边
          struct_hint:      (N, 8)   结构统计量

        输出:
          logit_credit: (N, 1)
          logit_grade:  (N, 5)
        ==========================================================================
        """
        N = x_ent.shape[0]

        # ── 1. 超图通道: 多视图结构学习 ──
        h_struct = self.hyper_encoder(x_ent, hyperedges)    # (N, 128)

        # ── 2. 属性通道: MLP 编码 ──
        h_attr = self.ent_encoder(x_ent)                     # (N, 128)

        # ── 3. 双通道融合 ──
        h_fusion = self.fusion_gate(h_struct, h_attr,
                                     struct_hint=struct_hint)  # (N, 64)

        # ── 4. MLP 投影 → 预测头 ──
        z_v = self.post_proj(h_fusion)                       # (N, 64)
        logit_credit, logit_grade = self.heads(z_v)

        return logit_credit, logit_grade


# ═══════════════════════════════════════════════════════════
# 损失函数
# ═══════════════════════════════════════════════════════════
def compute_losses(logit_credit, logit_grade,
                   y_credit, y_grade, mask,
                   hyperedges=None, prob_credit=None):
    """
    L_total = L_credit + λ_grade * L_grade + λ_struct * L_struct
    """
    # 主监督: 信用风险二分类
    n_pos = y_credit[mask].sum()
    n_neg = mask.sum() - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)]).clamp(max=100.0).to(logit_credit.device)
    L_credit = F.binary_cross_entropy_with_logits(
        logit_credit[mask].squeeze(-1), y_credit[mask].float(),
        pos_weight=pos_weight,
    )

    # 辅助: 企业分级
    L_grade = F.cross_entropy(logit_grade[mask], y_grade[mask])

    # ── 超图结构一致性（采样评估） ──
    L_struct = torch.tensor(0.0, device=logit_credit.device)
    if hyperedges and LAMBDA_STRUCT > 0 and prob_credit is not None:
        count = 0
        for view_name, he_list in hyperedges.items():
            if len(he_list) == 0:
                continue
            sample_n = min(len(he_list), 300)
            indices = torch.randperm(len(he_list))[:sample_n]
            for i in indices:
                he = he_list[i.item()].to(logit_credit.device)
                if len(he) >= 2:
                    preds = prob_credit[he]
                    L_struct += ((preds - preds.mean()) ** 2).mean()
                    count += 1
        if count > 0:
            L_struct /= count

    L_total = (
        L_credit
        + LAMBDA_GRADE * L_grade
        + LAMBDA_STRUCT * L_struct
    )

    loss_dict = {
        "total": L_total.item(),
        "credit": L_credit.item(),
        "grade": L_grade.item(),
        "struct": L_struct.item(),
    }
    return L_total, loss_dict


# ═══════════════════════════════════════════════════════════
# 评估
# ═══════════════════════════════════════════════════════════
def precision_at_k(y_true, y_score, k: int = 10):
    """Precision@K"""
    if len(y_score) == 0:
        return 0.0
    k = min(k, len(y_score))
    top_k_idx = np.argsort(y_score)[-k:]
    return float(y_true[top_k_idx].sum()) / k


@torch.no_grad()
def evaluate_model(model, data, mask):
    model.eval()
    logit_credit, logit_grade = model(
        data.x_ent,
        data.edge_index_dict,
        data.hyperedges,
        struct_hint=data.struct_hint if hasattr(data, "struct_hint") and data.struct_hint is not None else None,
    )
    prob = torch.sigmoid(logit_credit[mask]).cpu().squeeze(-1)
    y_true = data.y_credit[mask].cpu()

    auc = roc_auc_score(y_true, prob) if y_true.sum() > 0 and (1 - y_true).sum() > 0 else 0.5
    acc = accuracy_score(y_true, (prob >= 0.5).int())
    prec10 = precision_at_k(y_true.numpy(), prob.numpy(), k=10)

    return auc, acc, prec10


# ═══════════════════════════════════════════════════════════
# 训练入口
# ═══════════════════════════════════════════════════════════
def train(model, data, epochs: int = EPOCHS):

    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    print(f"  设备: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(device)}")
        print(f"  显存总量: {torch.cuda.get_device_properties(device).total_memory / 1024**3:.1f} GB")

    model = model.to(device)
    data = data.to(device)

    # ── AMP ──
    use_amp = USE_AMP and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if use_amp else None
    print(f"  AMP: {'启用' if use_amp else '关闭'}")

    # ── 优化器 ──
    param_groups = [
        {"params": model.hyper_encoder.parameters(), "lr": LR_HYPER},
        {"params": model.ent_encoder.parameters(), "lr": LR},
        {"params": model.fusion_gate.parameters(), "lr": LR},
        {"params": model.post_proj.parameters(), "lr": LR},
        {"params": model.heads.parameters(), "lr": LR},
    ]
    optimizer = AdamW(param_groups, lr=LR, weight_decay=WEIGHT_DECAY)

    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5,
                                   patience=15, min_lr=1e-6)
    best_auc = 0
    best_state = None
    patience = EARLY_STOP_PATIENCE

    struct_hint = data.struct_hint if hasattr(data, "struct_hint") and data.struct_hint is not None else None

    t0 = time.time()
    print(f"\n  开始训练 ({epochs} epochs, 早停={patience})...")
    print(f"  {'Epoch':>5s} | {'Loss':>7s} {'Credit':>7s} | "
          f"{'ValAUC':>7s} {'P@10':>7s} {'Best':>7s} | {'Time':>7s}")
    print(f"  {'─'*5:>5s}─┼{'─'*8:>8s}─┼{'─'*7:>7s}─┼{'─'*7:>7s}─┼{'─'*7:>7s}─┼{'─'*7:>7s}")

    t_epoch_start = time.time()

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        with torch.amp.autocast("cuda", enabled=use_amp):
            logit_credit, logit_grade = model(
                data.x_ent,
                data.edge_index_dict,
                data.hyperedges,
                struct_hint=struct_hint,
            )

            prob_credit = torch.sigmoid(logit_credit.squeeze(-1))
            loss, loss_dict = compute_losses(
                logit_credit, logit_grade,
                data.y_credit, data.y_grade,
                data.train_mask,
                hyperedges=data.hyperedges,
                prob_credit=prob_credit,
            )

        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        del logit_credit, logit_grade, loss, prob_credit

        val_auc, val_acc, val_prec10 = evaluate_model(model, data, data.val_mask)
        scheduler.step(val_auc)

        if val_auc > best_auc:
            best_auc = val_auc
            best_state = copy.deepcopy({k: v.cpu() for k, v in model.state_dict().items()})
            patience = EARLY_STOP_PATIENCE
        else:
            patience -= 1

        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

        t_epoch_end = time.time()
        elapsed_epoch = t_epoch_end - t_epoch_start
        t_epoch_start = t_epoch_end

        print(f"  {epoch+1:5d} | {loss_dict['total']:7.4f} {loss_dict['credit']:7.4f} | "
              f"{val_auc:7.4f} {val_prec10:7.4f} {best_auc:7.4f} | {elapsed_epoch:6.1f}s"
              + (" [BEST]" if patience == EARLY_STOP_PATIENCE else ""))

        if patience <= 0:
            print(f"  早停于 epoch {epoch+1}")
            break

    elapsed = time.time() - t0
    print(f"  训练完成, 总耗时 {elapsed:.1f}s, "
          f"最佳验证 AUC={best_auc:.4f}")

    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    return model
