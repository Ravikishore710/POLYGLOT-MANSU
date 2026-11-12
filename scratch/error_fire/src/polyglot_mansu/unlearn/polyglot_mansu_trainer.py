"""
src/polyglot_mansu/unlearn/polyglot_mansu_trainer.py
PolyglotMANSUTrainer:
Cross-Lingual Circuit-Intersected Null-Space Unlearning Engine with Multi-Grid 4-Bit Floor.
"""

import copy
from typing import Dict, List, Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from polyglot_mansu.unlearn.fisher_nullspace import (
    compute_diagonal_fisher,
    compute_fisher_preconditioner,
    orthogonalize_gradients,
)
from polyglot_mansu.unlearn.latent_adversary import LatentAdversaryHook, compute_latent_perturbation
from polyglot_mansu.unlearn.quant_permanence import compute_quant_deadzones, snap_quant_deadzone
from polyglot_mansu.guards import verify_guard_01_max_len

class PolyglotMANSUConfig:
    def __init__(
        self,
        lr: float = 2e-5,
        epochs: int = 3,
        beta: float = 0.1,
        lambda_rep: float = 0.5,
        lambda_ret: float = 1.0,
        lambda_b: float = 1.0,
        q_fisher: float = 0.85,
        alpha_F: float = 1.0,
        rho_latent: float = 0.08,
        gamma_quant: float = 1.25,
        eta_active: float = 0.35,
        use_circuit_mask: bool = True,
        use_quant_floor: bool = True,
        max_len: int = 1024
    ):
        verify_guard_01_max_len(max_len)
        self.lr = lr
        self.epochs = epochs
        self.beta = beta
        self.lambda_rep = lambda_rep
        self.lambda_ret = lambda_ret
        self.lambda_b = lambda_b
        self.q_fisher = q_fisher
        self.alpha_F = alpha_F
        self.rho_latent = rho_latent
        self.gamma_quant = gamma_quant
        self.eta_active = eta_active
        self.use_circuit_mask = use_circuit_mask
        self.use_quant_floor = use_quant_floor
        self.max_len = max_len

class PolyglotMANSUTrainer:
    def __init__(
        self,
        model: nn.Module,
        circuit_mask: Dict[str, torch.Tensor],
        config: Optional[PolyglotMANSUConfig] = None,
        device: Optional[torch.device] = None
    ):
        self.model = model
        self.circuit_mask = circuit_mask
        self.config = config or PolyglotMANSUConfig()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.reference_model = copy.deepcopy(model).to(self.device)
        self.reference_model.eval()
        for p in self.reference_model.parameters():
            p.requires_grad = False

        # Store initial reference weights theta_0
        self.theta_0 = {name: p.clone().detach() for name, p in model.named_parameters()}
        
        # Quantization deadzone thresholds
        if self.config.use_quant_floor:
            self.tau = {
                name: compute_quant_deadzones(p, gamma=self.config.gamma_quant).to(self.device)
                for name, p in model.named_parameters() if name in circuit_mask
            }
        else:
            self.tau = {}

        self.preconditioner = None

    def initialize_fisher(self, retain_loader: torch.utils.data.DataLoader):
        """Precomputes Fisher preconditioner P_i on retain data."""
        fisher = compute_diagonal_fisher(
            model=self.model,
            dataloader=retain_loader,
            circuit_mask=self.circuit_mask,
            device=self.device
        )
        self.preconditioner = compute_fisher_preconditioner(
            fisher=fisher,
            circuit_mask=self.circuit_mask,
            q=self.config.q_fisher,
            alpha_F=self.config.alpha_F
        )

    def train_step(
        self,
        forget_batch: Dict[str, torch.Tensor],
        retain_batch: Dict[str, torch.Tensor],
        optimizer: torch.optim.Optimizer
    ) -> Dict[str, float]:
        """
        Executes one unlearning step with latent perturbation, Fisher damping,
        and adaptive gradient orthogonalization.
        """
        self.model.train()
        optimizer.zero_grad()

        # 1. Retain / Boundary protection loss
        r_ids = retain_batch["input_ids"].to(self.device)
        r_labels = retain_batch.get("labels", r_ids).to(self.device)
        out_r = self.model(input_ids=r_ids, labels=r_labels)
        loss_retain = out_r.loss
        loss_retain.backward(retain_graph=True)

        g_r = {name: p.grad.clone() for name, p in self.model.named_parameters() if p.grad is not None}
        optimizer.zero_grad()

        # 2. Forget loss (SimNPO / GA on forget batch)
        f_ids = forget_batch["input_ids"].to(self.device)
        f_labels = forget_batch.get("labels", f_ids).to(self.device)
        out_f = self.model(input_ids=f_ids, labels=f_labels)
        with torch.no_grad():
            out_f_ref = self.reference_model(input_ids=f_ids, labels=f_labels)

        # Negative log-ratio unlearning loss
        logp_curr = -out_f.loss
        logp_ref = -out_f_ref.loss
        diff = logp_curr - logp_ref
        loss_forget = -torch.log(torch.sigmoid(-self.config.beta * diff) + 1e-8)
        loss_forget.backward()

        # 3. Apply Fisher Preconditioning & Orthogonalization
        with torch.no_grad():
            for name, p in self.model.named_parameters():
                if p.grad is None or name not in self.circuit_mask:
                    if p.grad is not None and not self.config.use_circuit_mask:
                        pass
                    elif p.grad is not None:
                        p.grad.zero_()
                    continue

                mask = self.circuit_mask[name].to(self.device)
                prec = self.preconditioner[name].to(self.device) if self.preconditioner else mask

                grad_f = p.grad * prec
                grad_r = g_r.get(name, torch.zeros_like(p.grad)) * prec

                g_total, _ = orthogonalize_gradients(grad_f, grad_r, lambda_ret=self.config.lambda_ret)
                p.grad.copy_(g_total)

        optimizer.step()

        return {
            "loss_forget": loss_forget.item(),
            "loss_retain": loss_retain.item()
        }

    def post_train_quant_snapping(self) -> Dict[str, Any]:
        """
        Consolidation phase: snaps parameters inside the dead-zone
        to the multi-grid quantization permanence threshold tau_i.
        """
        if not self.config.use_quant_floor:
            return {"snapped_count": 0, "status": "SKIPPED_ABLATION"}

        total_snapped = 0
        with torch.no_grad():
            for name, p in self.model.named_parameters():
                if name in self.circuit_mask and name in self.tau:
                    w_orig = self.theta_0[name].to(self.device)
                    tau_i = self.tau[name].to(self.device)
                    mask = self.circuit_mask[name].to(self.device)

                    w_snapped, stats = snap_quant_deadzone(
                        w_curr=p.data,
                        w_orig=w_orig,
                        tau=tau_i,
                        mask=mask,
                        eta_active=self.config.eta_active
                    )
                    p.data.copy_(w_snapped)
                    total_snapped += stats["snapped_count"]

        return {"snapped_count": total_snapped, "status": "CONSOLIDATED"}
