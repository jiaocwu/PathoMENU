
import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.common_modules import FeatureProcessor
from models.loss import DisentanglementLoss
from models.sequence_encoder import SequenceEncoder


class LogitClassifier(nn.Module):

    def __init__(self, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class WideLogitClassifier(nn.Module):

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PathoMENUBase(nn.Module):

    def __init__(self, structure_model: nn.Module, config: dict):
        super().__init__()
        self.structure_model = structure_model
        self.config = config.get("params", {})
        hidden_dim = structure_model.output_dim
        self.mlp_dropout = self.config.get("mlp_dropout", 0.45)
        transformer_dropout = self.config.get("transformer_dropout", 0.45)
        self.focal_gamma = self.config.get("focal_loss_gamma", 2.0)
        self.focal_alpha = self.config.get("focal_alpha", 0.25)
        self.lambda_pred_aux = self.config.get("lambda_auxiliary_prediction", 0.01)
        self.lambda_gate_rank = self.config.get("lambda_gate_ranking", 0.005)
        self.lambda_ortho = self.config.get("lambda_orthogonality", 0.0005)
        self.lambda_align = self.config.get("lambda_alignment", 0.002)
        self.rank_margin = self.config.get("rank_margin", 0.005)
        self.aux_warmup_epochs = int(self.config.get("aux_warmup_epochs", 15))
        self.aux_rampup_epochs = int(self.config.get("aux_rampup_epochs", 8))
        self.current_epoch = 0
        self.margin_ranking_loss = nn.MarginRankingLoss(reduction="none", margin=self.rank_margin)
        if self.config.get("sequence_feature", "esm1v") != "esm1v":
            raise ValueError("PathoMENU requires ESM1v features")
        esm_input_dim = self.config.get("esm1v_input_dim", 1280)
        self.feature_processors = nn.ModuleDict({
            "esm_wt": FeatureProcessor(esm_input_dim, 512, hidden_dim),
            "esm_mut": FeatureProcessor(esm_input_dim, 512, hidden_dim),
        })
        self.seq_encoder = SequenceEncoder(
            self.config.get("sequence_encoder_layers", 2),
            hidden_dim,
            self.config.get("sequence_encoder_heads", 8),
            self.config.get("sequence_encoder_hidden_dim", 512),
            transformer_dropout,
        )

    def set_current_epoch(self, epoch: int):
        self.current_epoch = epoch

    def set_focal_loss_alpha(self, alpha: float):
        self.focal_alpha = alpha

    def _extract_structure_features(self, graph) -> torch.Tensor:
        node_features = self.structure_model(graph, graph.res_stru_fea)["node_feature"]
        graph_sizes = graph.num_residue
        offsets = torch.cat([
            torch.tensor([0], device=graph_sizes.device),
            graph_sizes.cumsum(0)[:-1],
        ])
        return node_features[offsets + graph.mut_mpos]

    def _process_sequence_features(self, graph_wt, graph_mut) -> Tuple[torch.Tensor, torch.Tensor]:
        wild = self.feature_processors["esm_wt"](graph_wt.res_esm1v_fea)
        mutant = self.feature_processors["esm_mut"](graph_mut.res_esm1v_fea)
        return self.seq_encoder(wild)[:, 80, :], self.seq_encoder(mutant)[:, 80, :]

    def _focal_loss_with_logits(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = target.view_as(logits)
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        probability = torch.sigmoid(logits)
        p_t = probability * target + (1.0 - probability) * (1.0 - target)
        alpha_t = self.focal_alpha * target + (1.0 - self.focal_alpha) * (1.0 - target)
        return (alpha_t * torch.pow(1.0 - p_t, self.focal_gamma) * bce).squeeze(-1)

    def _aux_weight(self) -> float:
        if self.current_epoch < self.aux_warmup_epochs:
            return 0.0
        if self.current_epoch < self.aux_warmup_epochs + self.aux_rampup_epochs:
            progress = (self.current_epoch - self.aux_warmup_epochs) / max(1, self.aux_rampup_epochs)
            return 0.5 * (1.0 - math.cos(math.pi * progress))
        return 1.0

    def _gate_ranking_loss(self, gate: torch.Tensor, *branch_losses: torch.Tensor) -> torch.Tensor:
        if gate.size(1) != len(branch_losses):
            raise ValueError("The ranking gate size must match the number of branch losses")
        pairwise_losses = []
        for left in range(len(branch_losses)):
            for right in range(left + 1, len(branch_losses)):
                reliability_gap = (branch_losses[right].detach() - branch_losses[left].detach()).view(-1)
                valid = reliability_gap.abs() > 1e-6
                if torch.any(valid):
                    pairwise_losses.append(
                        self.margin_ranking_loss(
                            gate[valid, left],
                            gate[valid, right],
                            torch.sign(reliability_gap[valid]),
                        )
                    )
        if not pairwise_losses:
            return gate.new_tensor(0.0)
        return torch.cat(pairwise_losses).mean()


class StateDisentangler(nn.Module):

    def __init__(self, hidden_dim: int, dropout: float):
        super().__init__()
        self.shared_encoder = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.struct_private_encoder = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.esm_private_encoder = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )

    def forward(self, struct_feat: torch.Tensor, esm_feat: torch.Tensor) -> Dict[str, torch.Tensor]:
        struct_shared = self.shared_encoder(struct_feat)
        esm_shared = self.shared_encoder(esm_feat)
        return {
            "struct_shared": struct_shared,
            "esm_shared": esm_shared,
            "shared": 0.5 * (struct_shared + esm_shared),
            "struct_private": self.struct_private_encoder(struct_feat),
            "esm_private": self.esm_private_encoder(esm_feat),
        }


class PathoMENUFeatureFusion(nn.Module):

    def __init__(self, hidden_dim: int, dropout: float):
        super().__init__()
        self.state_disentangler = StateDisentangler(hidden_dim, dropout)
        self.gate = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )
        self.ranking_gate = nn.Sequential(
            nn.LayerNorm(hidden_dim * 3),
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3),
        )
        self.fusion = nn.Sequential(
            nn.LayerNorm(hidden_dim * 4),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )

    def forward(
        self,
        struct_wt: torch.Tensor,
        struct_mut: torch.Tensor,
        esm_wt: torch.Tensor,
        esm_mut: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        wt = self.state_disentangler(struct_wt, esm_wt)
        mut = self.state_disentangler(struct_mut, esm_mut)

        shared_delta = mut["shared"] - wt["shared"]
        struct_delta = mut["struct_private"] - wt["struct_private"]
        esm_delta = mut["esm_private"] - wt["esm_private"]
        gate = torch.softmax(self.gate(torch.cat([struct_delta, esm_delta], dim=-1)), dim=-1)
        ranking_gate = torch.softmax(
            self.ranking_gate(torch.cat([struct_delta, esm_delta, shared_delta], dim=-1)),
            dim=-1,
        )
        weighted_delta = gate[:, :1] * struct_delta + gate[:, 1:] * esm_delta
        fused = self.fusion(torch.cat([shared_delta, weighted_delta, struct_delta, esm_delta], dim=-1))

        return {
            "feat_struct_delta": struct_delta,
            "feat_esm_delta": esm_delta,
            "feat_shared_delta": shared_delta,
            "feat_weighted_delta": weighted_delta,
            "feat_final": fused,
            "delta_gate": gate,
            "ranking_gate": ranking_gate,
            "wt_struct_shared": wt["struct_shared"],
            "wt_esm_shared": wt["esm_shared"],
            "wt_struct_private": wt["struct_private"],
            "wt_esm_private": wt["esm_private"],
            "mut_struct_shared": mut["struct_shared"],
            "mut_esm_shared": mut["esm_shared"],
            "mut_struct_private": mut["struct_private"],
            "mut_esm_private": mut["esm_private"],
        }


class PathoMENU(PathoMENUBase):

    def __init__(self, structure_model: nn.Module, config: dict):
        super().__init__(structure_model, config)
        hidden_dim = self.structure_model.output_dim
        self.delta_fusion = PathoMENUFeatureFusion(hidden_dim, self.mlp_dropout)
        self.classifier = WideLogitClassifier(hidden_dim * 5 + 2, hidden_dim, self.mlp_dropout)
        self.classifier_struct = LogitClassifier(hidden_dim, self.mlp_dropout)
        self.classifier_esm = LogitClassifier(hidden_dim, self.mlp_dropout)
        self.classifier_shared = LogitClassifier(hidden_dim, self.mlp_dropout)
        self.disentanglement_loss = DisentanglementLoss(
            temperature=self.config.get("disentanglement_temperature", 0.1),
        )

    def _compute_disentanglement_loss(self, outputs: Dict[str, torch.Tensor]):
        wt_ortho, wt_align = self.disentanglement_loss(
            outputs["wt_struct_shared"],
            outputs["wt_struct_private"],
            outputs["wt_esm_shared"],
            outputs["wt_esm_private"],
        )
        mut_ortho, mut_align = self.disentanglement_loss(
            outputs["mut_struct_shared"],
            outputs["mut_struct_private"],
            outputs["mut_esm_shared"],
            outputs["mut_esm_private"],
        )
        ortho_loss = 0.5 * (wt_ortho.mean() + mut_ortho.mean())
        align_loss = 0.5 * (wt_align.mean() + mut_align.mean())
        return ortho_loss, align_loss

    def forward(self, batch_wt, batch_mut) -> Dict[str, torch.Tensor]:
        struct_wt = self._extract_structure_features(batch_wt)
        struct_mut = self._extract_structure_features(batch_mut)
        esm_wt, esm_mut = self._process_sequence_features(batch_wt, batch_mut)

        if hasattr(batch_wt, "y"):
            target = batch_wt.y.view(-1, 1).float()
        elif hasattr(batch_wt, "label"):
            target = batch_wt.label.view(-1, 1).float()
        else:
            raise AttributeError("Cannot find labels in graph data")

        sample_weights = (
            batch_wt.sample_weight.view(-1)
            if hasattr(batch_wt, "sample_weight")
            else torch.ones_like(target).view(-1)
        )

        outputs = self.delta_fusion(struct_wt, struct_mut, esm_wt, esm_mut)
        wide_final = torch.cat(
            [
                outputs["feat_final"],
                outputs["feat_struct_delta"],
                outputs["feat_esm_delta"],
                torch.abs(outputs["feat_struct_delta"] - outputs["feat_esm_delta"]),
                outputs["feat_struct_delta"] * outputs["feat_esm_delta"],
                outputs["delta_gate"],
            ],
            dim=-1,
        )
        logits = self.classifier(wide_final)
        logits_struct = self.classifier_struct(outputs["feat_struct_delta"])
        logits_esm = self.classifier_esm(outputs["feat_esm_delta"])
        logits_shared = self.classifier_shared(outputs["feat_shared_delta"])

        loss_main = self._focal_loss_with_logits(logits, target) * sample_weights
        loss_struct = self._focal_loss_with_logits(logits_struct, target) * sample_weights
        loss_esm = self._focal_loss_with_logits(logits_esm, target) * sample_weights
        loss_shared = self._focal_loss_with_logits(logits_shared, target) * sample_weights
        shared_pred_loss = self.lambda_pred_aux * loss_shared.mean()
        aux_pred_loss = self.lambda_pred_aux * (loss_struct.mean() + loss_esm.mean()) + shared_pred_loss
        gate_rank_loss = self._gate_ranking_loss(
            outputs["ranking_gate"],
            loss_struct,
            loss_esm,
            loss_shared,
        )
        ortho_loss, align_loss = self._compute_disentanglement_loss(outputs)
        scaled_gate_rank_loss = self.lambda_gate_rank * gate_rank_loss
        scaled_ortho_loss = self.lambda_ortho * ortho_loss
        scaled_align_loss = self.lambda_align * align_loss
        aux_weight = self._aux_weight()
        total_loss = loss_main.mean() + aux_weight * (
            aux_pred_loss
            + scaled_gate_rank_loss
            + scaled_ortho_loss
            + scaled_align_loss
        )

        return {
            "total_loss": total_loss,
            "focal_loss": loss_main.mean(),
            "aux_pred_loss": aux_pred_loss,
            "shared_pred_loss": shared_pred_loss,
            "gate_ranking_loss": scaled_gate_rank_loss,
            "ortho_loss": scaled_ortho_loss,
            "align_loss": scaled_align_loss,
            "predictions": torch.sigmoid(logits).squeeze(-1),
            "shared_predictions": torch.sigmoid(logits_shared).squeeze(-1),
            "ranking_gate": outputs["ranking_gate"],
            "labels": target.squeeze(-1),
        }
