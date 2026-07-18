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
from layers.feature_gate import AdaptiveFeatureGate
from layers.layer4_temporal import TemporalEncoder
from classifier import MultiTaskHeads


class FinTemporalEncoder(nn.Module):
    """
    ==========================================================================
    v5.2 财务特征时序编码器

    小 GRU 在 2 维财务特征上做 4 步半年度序列建模，
    替代原来融合后的大维度全局 GRU（后者被超图静态输出淹没）。

    双路径:
      路径A (GRU): x_seq 原始财务序列 → GRU(2→8→2) → fin_gru
      路径B (MLP): x_fin_raw (FeatureGate 输出) → MLP(2→8→2) → fin_mlp
      Gate:       has_fin_data ? fin_gru : fin_mlp
                  (只有有时序数据的企业才信 GRU，中小企业走 MLP)

    消融模式 (ablation=True): 仅路径B (MLP)，退化为静态财务特征投影
    ==========================================================================
    """
    def __init__(self, fin_dim=2, gru_hidden=8, dropout=0.2, ablation=False):
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

        # MLP（既作为 GRU 模式的 fallback，也作为消融模式的主路径）
        self.mlp = nn.Sequential(
            nn.Linear(fin_dim, gru_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(gru_hidden, fin_dim),
        )

    def forward(self, x_fin_seq, fin_all_missing, x_fin_raw):
        """
        x_fin_seq:       (N, 4, fin_dim)  4 个半年度财务指标原始序列
        fin_all_missing: (N,)             完全缺失财务数据的企业（中小企业）
        x_fin_raw:       (N, fin_dim)     FeatureGate 输出的静态财务特征
        returns:         (N, fin_dim)     时序感知的财务特征
        """
        if self.ablation:
            # 消融模式：仅 MLP 静态投影
            return self.mlp(x_fin_raw)

        N = x_fin_seq.shape[0]

        # 路径A: GRU 时序编码
        gru_out, _ = self.gru(x_fin_seq)          # (N, 4, gru_hidden)
        fin_gru = self.gru_proj(gru_out[:, -1, :])  # (N, fin_dim)

        # 路径B: MLP fallback（静态财务特征投影）
        fin_mlp = self.mlp(x_fin_raw)              # (N, fin_dim)

        # Gate: 有财务数据 → 信 GRU；无 → 信 MLP
        has_data = (~fin_all_missing).float().unsqueeze(-1)  # (N, 1)
        x_fin = has_data * fin_gru + (1.0 - has_data) * fin_mlp

        return x_fin


class HyperHeteroModel(nn.Module):
    """
    ==========================================================================
    v5.2 完整模型：超图异构双通道 + Γ 风险传播 + 财务特征时序GRU

    数据流（新方案 / 特征分工 ON）:
      Enterprise 特征 (N, 13)
        ├─→ FeatureGate → X_gated (N, 13)
        ├─→ 特征分流:
        │     ├─→ col[0-7,10-11] → X_hyper (N, 10) → MultiViewHyperEncoder → h_struct
        │     └─→ col[8,9] → X_fin_raw (N, 2)
        │           └─→ FinTemporalEncoder(x_seq财务序列 + X_fin_raw) → X_fin (N, 2)
        │               (小GRU捕捉营收/资产半年度波动，替代全局GRU)
        ├─→ HeteroChannelEncoder(X_fin) → h_feat
        ├─→ FusionGate(h_struct, h_feat, hint) → h_fusion (N, 64)
        ├─→ Γ 跨关系风险传播 → h_risk (N, 128)
        ├─→ Concat[h_fusion, h_risk] → (N, 192)
        ├─→ PostProj(MLP) → z_v (N, 64)    ← 时序已在财务层面完成，此处仅做维度投影
        └─→ MultiTaskHeads → logit_white, logit_risk, logit_grade

    数据流（旧方案 / 特征分工 OFF，兼容）:
      Enterprise 特征 (N, 13)
        ├─→ FeatureGate → X_gated (N, 13)
        ├─→ 两通道共享 X_gated
        ├─→ FusionGate → h_fusion → Γ → Concat
        └─→ TemporalEncoder(GRU/MLP) → z_v    ← 旧全局时序编码器
    ==========================================================================
    """

    def __init__(self, in_dims: dict, edge_types: list):
        super().__init__()

        # ── 自适应门控阀 ──
        ent_dim = in_dims.get("enterprise", 13)

        # v5.1 特征分工：同构通道看关系，异构通道看财务
        if not _cfg.ABLATION_NO_FEATURE_SPLIT:
            struct_dim = len(_cfg.STRUCT_FEATURE_INDICES)   # 10: SCF(8) + 诉讼(2)
            fin_dim = len(_cfg.FINANCIAL_FEATURE_INDICES)    # 2: 营收增长率 + 资产周转率
        else:
            struct_dim = ent_dim   # 消融：恢复为统一 13 维
            fin_dim = ent_dim

        self.feature_gate = AdaptiveFeatureGate(
            feature_dim=ent_dim,
            struct_hint_dim=8,
            hidden=64,
        )

        # 财务缺失企业的可学习向量（替代填 0，仅特征分流模式下生效）
        self.fin_missing_emb = nn.Parameter(torch.randn(fin_dim) * 0.01)

        # ── 同构通道：多视图超图 ──
        self.hyper_encoder = MultiViewHyperEncoder(
            in_dim=struct_dim,
            hidden=HYPER_HIDDEN,
        )

        # ── 异构通道：特征图（enterprise 只接收财务维度的特征） ──
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

        # ── 时序 / 投影（v5.2：新旧方案双路径） ──
        self.fusion_dim_val = FUSION_HIDDEN + HIDDEN_DIM  # 64 + 128 = 192

        if not _cfg.ABLATION_NO_FEATURE_SPLIT:
            # ── 新方案：财务特征时序编码（v5.2） ──
            self.fin_temporal = FinTemporalEncoder(
                fin_dim=fin_dim,
                gru_hidden=8,
                dropout=DROPOUT,
                ablation=_cfg.ABLATION_NO_TEMPORAL,
            )
            # 融合后仅做简单 MLP 投影（时序已在财务特征层面完成）
            self.post_proj = nn.Sequential(
                nn.Linear(self.fusion_dim_val, 64),
                nn.ReLU(),
                nn.Dropout(DROPOUT),
                nn.Linear(64, 64),
            )
            self.temporal = None  # 旧全局时序编码器不使用
        else:
            # ── 旧方案：全局时序编码器（保持不变） ──
            self.fin_temporal = None
            self.post_proj = None
            if _cfg.ABLATION_NO_TEMPORAL:
                self.temporal = nn.Sequential(
                    nn.Linear(self.fusion_dim_val, 64),
                    nn.ReLU(),
                    nn.Dropout(DROPOUT),
                    nn.Linear(64, 64),
                )
            else:
                self.temporal = TemporalEncoder(input_dim=None)

        # ── 预测头 ──
        self.heads = MultiTaskHeads(in_dim=64)  # 两种路径都输出 64 维

        # ── 融合后投影（替代已删除的 Γ 矩阵，为图3风险传播网络预留接口） ──
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

        # ── 1. FeatureGate ──
        x_ent = x_dict["enterprise"]
        m_ent = x_missing.get("enterprise") if x_missing else None
        xs_ent = x_struct.get("enterprise") if x_struct else None
        hint_ent = struct_hint.get("enterprise") if struct_hint else None

        if m_ent is not None and xs_ent is not None:
            if hint_ent is None:
                hint_ent = torch.zeros(x_ent.shape[0], 8, device=device)
            x_gated_ent = self.feature_gate(x_ent, m_ent, hint_ent, xs_ent)
        else:
            x_gated_ent = x_ent

        # ── 2. 特征分流（v5.2：同构通道看关系，异构通道看财务 + 时序GRU） ──
        if not _cfg.ABLATION_NO_FEATURE_SPLIT:
            # 同构通道：SCF(8) + 诉讼(2) = 10 维，排除财务特征
            x_hyper = x_gated_ent[:, _cfg.STRUCT_FEATURE_INDICES]
            # 异构通道：营收增长率 + 资产周转率 = 2 维，聚焦经营表现
            x_fin_raw = x_gated_ent[:, _cfg.FINANCIAL_FEATURE_INDICES]

            # 中小企业财务缺失 → 用于 Gate 选择 MLP fallback
            if m_ent is not None:
                m_fin = m_ent[:, _cfg.FINANCIAL_FEATURE_INDICES]
                fin_all_missing = m_fin.all(dim=1)  # (N_ent,)
            else:
                fin_all_missing = torch.zeros(N_ent, dtype=torch.bool, device=device)

            # v5.2 财务特征时序编码（小GRU在2维财务序列上，替代全局GRU）
            if self.fin_temporal is not None and x_seq is not None:
                # 从 x_seq 提取对应财务指标的 4 步序列 (N, 4, fin_dim)
                x_fin_seq = x_seq[:, :, _cfg.FIN_SEQ_INDICES]
                x_fin = self.fin_temporal(x_fin_seq, fin_all_missing, x_fin_raw)
            else:
                # Fallback: 无 x_seq 时用 FeatureGate 输出 + 缺失嵌入
                x_fin = x_fin_raw.clone()
                if fin_all_missing.any():
                    x_fin[fin_all_missing] = self.fin_missing_emb.unsqueeze(0)
        else:
            # 消融模式：不拆分特征，两个通道吃相同输入
            x_hyper = x_gated_ent
            x_fin = x_gated_ent

        x_dict_gated = {**x_dict, "enterprise": x_fin}

        # ── 3. 同构通道：超图编码 ──
        h_struct = self.hyper_encoder(x_hyper, hyperedges)  # (N_ent, 128)

        # ── 4. 异构通道：特征图编码 ──
        h_feat = self.hetero_encoder(x_dict_gated, edge_index_dict)  # (N_ent, 128)
        if h_feat is None:
            h_feat = torch.zeros(N_ent, HIDDEN_DIM, device=device)

        # ── 5. 双通道融合 ──
        h_fusion = self.fusion_gate(h_struct, h_feat, struct_hint=hint_ent)  # (N_ent, 64)

        # ── 6. 融合后投影（v5.3: Γ 已删除，直接投影 → 为图3风险传播网络预留接口） ──
        h_proj = self.fusion_proj(h_fusion)  # (N_ent, 64) → (N_ent, 128)
        h_combined = torch.cat([h_fusion, h_proj], dim=-1)  # (N_ent, 192)

        # ── 8. 时序 / 投影（v5.2：新旧方案分叉） ──
        if not _cfg.ABLATION_NO_FEATURE_SPLIT:
            # 新方案：时序已在财务特征层面完成，此处仅 MLP 维度投影
            z_v = self.post_proj(h_combined)            # MLP: (N_ent, 192) → (N_ent, 64)
        else:
            # 旧方案：全局时序编码器
            if _cfg.ABLATION_NO_TEMPORAL:
                z_v = self.temporal(h_combined)         # MLP: (N_ent, 192) → (N_ent, 64)
            else:
                z_v = self.temporal(h_combined, x_seq=x_seq)  # GRU: + x_seq

        # ── 9. 预测头 ──
        logit_white, logit_risk, logit_grade = self.heads(z_v)

        return logit_white, logit_risk, logit_grade


# ═══════════════════════════════════════════════════════════
# 损失函数
# ═══════════════════════════════════════════════════════════
def compute_losses(logit_white, logit_risk, logit_grade,
                   y_white, y_risk, y_grade, mask,
                   h_fusion, hyperedges=None, h_struct=None):
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
        {"params": model.feature_gate.parameters(), "lr": LR},
        {"params": model.heads.parameters(), "lr": LR},
        {"params": [model.fin_missing_emb], "lr": LR},
    ]
    if model.temporal is not None:
        param_groups.append({"params": model.temporal.parameters(), "lr": LR})
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
