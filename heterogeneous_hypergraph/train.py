"""
================================================================================
训练循环 — 超图异构双通道模型 v5
================================================================================
内存优化版：
  - AMP 混合精度（显存砍 ~40%）
  - 单次全批量前向（去掉假 mini-batch 循环，原来每 epoch 算 6 遍）
  - 每 epoch 强制清理 CUDA cache + GC
  - HeteroConv 层数减为 1（6 边 SAGEConv 的中间张量是最大内存杀手）
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

from config import (
    DEVICE, EPOCHS, LR, LR_HYPER, WEIGHT_DECAY, EARLY_STOP_PATIENCE, SEED,
    LAMBDA_RISK, LAMBDA_GRADE, LAMBDA_STRUCT,
    HIDDEN_DIM, HYPER_HIDDEN, FUSION_HIDDEN, USE_AMP, DROPOUT,
)
import config as _cfg
from hypergraph.hypergraph_conv import MultiViewHyperEncoder
from heterogeneous.hetero_encoder import HeteroChannelEncoder
from fusion.fusion_gate import FusionGate
from layers.layer4_temporal import TemporalEncoder
from classifier import MultiTaskHeads


class FinTemporalEncoder(nn.Module):
    """
    ==========================================================================
    v5.4 财务特征时序编码器

    GRU 在 12 维财务指标上做 4 步半年度序列建模（不再依赖 X_ent 中的财务列）。

    双路径:
      路径A (GRU): x_seq 全量 12 维财务序列 → GRU(fin_dim→gru_hidden→fin_dim) → fin_gru
      路径B (Emb): fin_missing_emb → MLP(fin_dim→gru_hidden→fin_dim) → fin_emb
      Gate:       has_fin_data ? fin_gru : fin_emb
                  (有时序数据的企业走 GRU，中小企业走可学习嵌入)

    消融模式 (ablation=True): 仅 Emb 路径，退化为静态嵌入
    ==========================================================================
    """
    def __init__(self, fin_dim=12, gru_hidden=8, dropout=0.2, ablation=False):
        super().__init__()
        self.fin_dim = fin_dim
        self.gru_hidden = gru_hidden
        self.ablation = ablation

        if not ablation:
            self.gru = nn.GRU(
                input_size=fin_dim,
                hidden_size=gru_hidden,
                num_layers=1,
                batch_first=True,
                dropout=0.0,
            )
            self.gru_proj = nn.Linear(gru_hidden, fin_dim)

        # 用于中小企业 fallback: 把 fin_missing_emb 投影到 fin_dim
        self.emb_proj = nn.Sequential(
            nn.Linear(fin_dim, gru_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(gru_hidden, fin_dim),
        )

    def forward(self, x_fin_seq, fin_all_missing, fin_missing_emb):
        """
        x_fin_seq:       (N, 4, fin_dim)  全量 12 维财务时序
        fin_all_missing: (N,)             完全缺失财务数据的企业
        fin_missing_emb: (fin_dim,)       可学习嵌入（替代缺失企业的财务表示）
        returns:         (N, fin_dim)     时序感知的财务特征
        """
        N = x_fin_seq.shape[0]
        fin_emb = self.emb_proj(fin_missing_emb)                    # (fin_dim,)

        if self.ablation:
            return fin_emb.unsqueeze(0).expand(N, -1)               # (N, fin_dim)

        # 路径A: GRU 时序编码
        gru_out, _ = self.gru(x_fin_seq)                            # (N, 4, gru_hidden)
        fin_gru = self.gru_proj(gru_out[:, -1, :])                  # (N, fin_dim)

        # 路径B: 可学习嵌入 (fin_emb)
        # Gate: 有财务数据 → 信 GRU；无 → 信 embedding
        has_data = (~fin_all_missing).float().unsqueeze(-1)         # (N, 1)
        x_fin = has_data * fin_gru + (1.0 - has_data) * fin_emb.unsqueeze(0)

        return x_fin


class HyperHeteroModel(nn.Module):
    """
    ==========================================================================
    v5.4 完整模型：超图异构双通道 + 财务特征全量 GRU（FeatureGate/特征分流 已删除）

    数据流:
      Enterprise 特征 (N, 11, 纯结构)
        └─→ MultiViewHyperEncoder → h_struct (N, 128)

      x_seq 全量 12 维财务时序 (N, 4, 12)
        └─→ FinTemporalEncoder(GRU + Emb fallback) → x_fin (N, 12)
            └─→ HeteroChannelEncoder → h_feat (N, 128)

      FusionGate(h_struct, h_feat) → h_fusion (N, 64)
        → fusion_proj (64→128) → Concat → (N, 192)
        → PostProj(MLP) → z_v (N, 64)
        → MultiTaskHeads → logit_white, logit_risk, logit_grade
    ==========================================================================
    """

    def __init__(self, in_dims: dict, edge_types: list):
        super().__init__()

        ent_dim = in_dims.get("enterprise", 11)               # X_ent: 纯结构特征
        fin_dim = _cfg.FIN_DIM                                 # 12: x_seq 全量财务指标

        # 财务缺失企业的可学习嵌入（替代填零，送入 FinTemporalEncoder 的 Emb 路径）
        self.fin_missing_emb = nn.Parameter(torch.randn(fin_dim) * 0.01)

        # ── 同构通道：多视图超图（吃 X_ent 全量结构特征） ──
        self.hyper_encoder = MultiViewHyperEncoder(
            in_dim=ent_dim,
            hidden=HYPER_HIDDEN,
        )

        # ── 异构通道：特征图（吃 FinTemporalEncoder 输出的财务特征） ──
        hetero_in_dims = {**in_dims, "enterprise": fin_dim}
        self.hetero_encoder = HeteroChannelEncoder(
            in_dims=hetero_in_dims,
            edge_types=edge_types,
            hidden=HIDDEN_DIM,
        )

        # ── 双通道融合 ──
        self.fusion_gate = FusionGate(
            struct_dim=HYPER_HIDDEN,
            feat_dim=HIDDEN_DIM,
            hidden=FUSION_HIDDEN,
            hint_dim=8,
        )

        self.fusion_dim_val = FUSION_HIDDEN + HIDDEN_DIM  # 64 + 128 = 192

        # ── 财务特征时序编码（v5.4: 12 维全量 x_seq 进 GRU） ──
        self.fin_temporal = FinTemporalEncoder(
            fin_dim=fin_dim,
            gru_hidden=8,
            dropout=DROPOUT,
            ablation=_cfg.ABLATION_NO_TEMPORAL,
        )

        # ── 融合后 MLP 投影 ──
        self.post_proj = nn.Sequential(
            nn.Linear(self.fusion_dim_val, 64),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(64, 64),
        )

        # ── 预测头 ──
        self.heads = MultiTaskHeads(in_dim=64)

        # ── 融合后投影（为图3风险传播网络预留接口） ──
        self.fusion_proj = nn.Linear(FUSION_HIDDEN, HIDDEN_DIM)  # 64 → 128

    def forward(self, x_dict: dict, edge_index_dict: dict,
                hyperedges: dict, num_enterprises: int,
                x_struct: dict = None, x_missing: dict = None,
                struct_hint: dict = None, x_seq: torch.Tensor = None):
        """
        ==========================================================================
        完整前向传播（AMP 友好）。
        ==========================================================================
        """
        N_ent = x_dict["enterprise"].shape[0]
        device = x_dict["enterprise"].device
        x_ent = x_dict["enterprise"]                                  # (N, 11) 纯结构
        hint_ent = struct_hint.get("enterprise") if struct_hint else None

        # ── 1. 超图通道: X_ent 全量进 ──
        h_struct = self.hyper_encoder(x_ent, hyperedges)               # (N, 128)

        # ── 2. 财务时序: x_seq 全量 12 维 → FinTemporalEncoder ──
        if x_seq is not None:
            x_fin_seq = x_seq[:, :, :_cfg.FIN_DIM]                     # (N, 4, 12)
            fin_all_missing = (x_fin_seq.abs().sum(dim=(1, 2)) < 1e-8) # 全零序列 = 无数据
            x_fin = self.fin_temporal(x_fin_seq, fin_all_missing,
                                      self.fin_missing_emb)             # (N, 12)
        else:
            # 无 x_seq: 全部用 embedding
            x_fin = self.fin_missing_emb.unsqueeze(0).expand(N_ent, -1)

        # ── 3. 异构通道: x_fin → HeteroConv ──
        x_dict_gated = {**x_dict, "enterprise": x_fin}
        h_feat = self.hetero_encoder(x_dict_gated, edge_index_dict)    # (N, 128)
        if h_feat is None:
            h_feat = torch.zeros(N_ent, HIDDEN_DIM, device=device)

        # ── 4. 双通道融合 ──
        h_fusion = self.fusion_gate(h_struct, h_feat,
                                    struct_hint=hint_ent)               # (N, 64)

        # ── 5. 融合后投影 ──
        h_proj = self.fusion_proj(h_fusion)                             # (N, 64) → (N, 128)
        h_combined = torch.cat([h_fusion, h_proj], dim=-1)             # (N, 192)

        # ── 6. MLP 投影 → 预测头 ──
        z_v = self.post_proj(h_combined)                                # (N, 64)
        logit_white, logit_risk, logit_grade = self.heads(z_v)

        return logit_white, logit_risk, logit_grade


# ═══════════════════════════════════════════════════════════
# 损失函数
# ═══════════════════════════════════════════════════════════
def compute_losses(logit_white, logit_risk, logit_grade,
                   y_white, y_risk, y_grade, mask,
                   h_fusion=None, hyperedges=None, h_struct=None):
    # 主监督
    n_pos = y_white[mask].sum()
    n_neg = mask.sum() - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)]).clamp(max=100.0).to(logit_white.device)
    L_white = F.binary_cross_entropy_with_logits(
        logit_white[mask].squeeze(-1), y_white[mask].float(),
        pos_weight=pos_weight,
    )
    L_risk = F.binary_cross_entropy_with_logits(
        logit_risk[mask].squeeze(-1), y_risk[mask].float()
    )
    L_grade = F.cross_entropy(logit_grade[mask], y_grade[mask])

    # 超图结构一致性（采样评估，不全量遍历防 OOM）
    L_struct = torch.tensor(0.0, device=logit_white.device)
    if hyperedges and LAMBDA_STRUCT > 0:
        prob_w = torch.sigmoid(logit_white.squeeze(-1))
        count = 0
        # 每个视图只采样前 50 条超边（避免全量遍历撑爆显存）
        for view_name, he_list in hyperedges.items():
            sample_n = min(len(he_list), 50)
            for i in range(sample_n):
                he = he_list[i].to(logit_white.device)
                if len(he) >= 2:
                    preds = prob_w[he]
                    L_struct += ((preds - preds.mean()) ** 2).mean()
                    count += 1
        if count > 0:
            L_struct /= count

    L_total = (L_white
               + LAMBDA_RISK * L_risk
               + LAMBDA_GRADE * L_grade
               + LAMBDA_STRUCT * L_struct)

    loss_dict = {
        "total": L_total.item(), "white": L_white.item(),
        "risk": L_risk.item(), "grade": L_grade.item(),
        "struct": L_struct.item(),
    }
    return L_total, loss_dict


# ═══════════════════════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════════════════════
def precision_at_k(y_true, y_score, k: int = 10):
    """Precision@K: 模型打分最高的 K 个样本中正类的比例"""
    if len(y_score) == 0:
        return 0.0
    k = min(k, len(y_score))
    top_k_idx = np.argsort(y_score)[-k:]  # 升序取最后 k = 最高 k 个
    return float(y_true[top_k_idx].sum()) / k


# ═══════════════════════════════════════════════════════════
# 评估
# ═══════════════════════════════════════════════════════════
@torch.no_grad()
def evaluate_model(model, data, mask):
    model.eval()
    logit_w, logit_r, logit_g = model(
        data.x_dict, data.edge_index_dict, data.hyperedges,
        data.num_enterprises,
        x_struct=data.x_struct, x_missing=data.x_missing,
        struct_hint=data.struct_hint,
        x_seq=data.x_seq if hasattr(data, "x_seq") and data.x_seq is not None else None,
    )
    prob_w = torch.sigmoid(logit_w[mask]).cpu().squeeze(-1)
    y_true = data.y_white[mask].cpu()
    auc = roc_auc_score(y_true, prob_w) if y_true.sum() > 0 and (1 - y_true).sum() > 0 else 0.5
    acc = accuracy_score(y_true, (prob_w >= 0.5).int())
    prec10 = precision_at_k(y_true.numpy(), prob_w.numpy(), k=10)
    return auc, acc, prec10


def _report_memory(device: torch.device):
    """打印当前 GPU 显存使用情况"""
    if device.type == "cuda":
        allocated = torch.cuda.memory_allocated(device) / 1024**3
        reserved = torch.cuda.memory_reserved(device) / 1024**3
        print(f"  GPU 显存: 已分配 {allocated:.2f} GB, 已预留 {reserved:.2f} GB")
    else:
        print(f"  运行在 CPU，未使用 GPU")


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

    _report_memory(device)

    # ── AMP 混合精度 ──
    use_amp = USE_AMP and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if use_amp else None
    print(f"  AMP 混合精度: {'启用' if use_amp else '关闭'}")

    # ── 优化器 ──
    param_groups = [
        {"params": model.hyper_encoder.parameters(), "lr": LR_HYPER},
        {"params": model.hetero_encoder.parameters(), "lr": LR},
        {"params": model.fusion_gate.parameters(), "lr": LR},
        {"params": model.heads.parameters(), "lr": LR},
        {"params": [model.fin_missing_emb], "lr": LR},
    ]
    if model.fin_temporal is not None:
        param_groups.append({"params": model.fin_temporal.parameters(), "lr": LR})
    if model.post_proj is not None:
        param_groups.append({"params": model.post_proj.parameters(), "lr": LR})
    optimizer = AdamW(param_groups, lr=LR, weight_decay=WEIGHT_DECAY)

    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5,
                                   patience=15, min_lr=1e-6)
    best_auc = 0
    best_state = None
    patience = EARLY_STOP_PATIENCE

    # ── 预构建 forward 参数（不变的部分只传一次） ──
    x_seq = data.x_seq if hasattr(data, "x_seq") and data.x_seq is not None else None

    print(f"\n  开始训练 ({epochs} epochs, 早停={patience})...")
    print(f"  {'Epoch':>5s} | {'Loss':>7s} {'W':>7s} | "
          f"{'ValAUC':>7s} {'P@10':>7s} {'Best':>7s} | {'Time':>7s}")
    print(f"  {'─'*5:>5s}─┼{'─'*8:>8s}─┼{'─'*7:>7s}─┼{'─'*7:>7s}─┼{'─'*7:>7s}─┼{'─'*7:>7s}")
    t0 = time.time()
    t_epoch_start = time.time()

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        # ── 全批量前向（AMP） ──
        with torch.amp.autocast("cuda", enabled=use_amp):
            logit_w, logit_r, logit_g = model(
                data.x_dict, data.edge_index_dict, data.hyperedges,
                data.num_enterprises,
                x_struct=data.x_struct, x_missing=data.x_missing,
                struct_hint=data.struct_hint,
                x_seq=x_seq,
            )

            loss, loss_dict = compute_losses(
                logit_w, logit_r, logit_g,
                data.y_white, data.y_risk, data.y_grade,
                data.train_mask,
                hyperedges=data.hyperedges,
            )

        # ── 反向 ──
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

        del logit_w, logit_r, logit_g, loss

        # ── 验证 ──
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

        # ── 日志（每 epoch 输出，CPU 调试不嫌多） ──
        t_epoch_end = time.time()
        elapsed_epoch = t_epoch_end - t_epoch_start
        t_epoch_start = t_epoch_end

        print(f"  {epoch+1:5d} | {loss_dict['total']:7.4f} {loss_dict['white']:7.4f} | "
              f"{val_auc:7.4f} {val_prec10:7.4f} {best_auc:7.4f} | {elapsed_epoch:6.1f}s"
              + (" [BEST]" if patience == EARLY_STOP_PATIENCE else ""))

        if patience <= 0:
            print(f"  早停于 epoch {epoch+1}")
            break

    elapsed = time.time() - t0
    print(f"  训练完成, {elapsed:.1f}s, 最佳验证 AUC={best_auc:.4f}")

    # 恢复最佳模型（从 CPU copy 回 device）
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    return model
