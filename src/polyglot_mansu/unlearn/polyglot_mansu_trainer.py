"""
src/polyglot_mansu/unlearn/polyglot_mansu_trainer.py
PolyglotMANSUTrainer:
Cross-Lingual Circuit-Intersected Null-Space Unlearning Engine with Multi-Grid 4-Bit Floor.
Enforces parameter freezing on non-circuit weights to optimize VRAM and prevent OOM.
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
        device: Optional[torch.device] = None,
        reference_model: Optional[nn.Module] = None
    ):
        self.model = model
        self.device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.circuit_mask = {k: v.to(self.device) for k, v in circuit_mask.items()}
        self.config = config or PolyglotMANSUConfig()

        # Freeze non-circuit parameters so AdamW optimizer only tracks circuit weights (VRAM guard)
        self.trainable_params_count = 0
        self.total_params_count = 0
        for name, p in self.model.named_parameters():
            self.total_params_count += p.numel()
            if self.config.use_circuit_mask and name in self.circuit_mask and (self.circuit_mask[name] > 0).any().item():
                p.requires_grad = True
                self.trainable_params_count += p.numel()
            elif not self.config.use_circuit_mask:
                p.requires_grad = True
                self.trainable_params_count += p.numel()
            else:
                p.requires_grad = False

        # Place reference model on 2nd GPU if available (Kaggle Dual-T4) or CPU to preserve VRAM
        if torch.cuda.is_available() and torch.cuda.device_count() > 1 and str(self.device).startswith("cuda:0"):
            self.ref_device = torch.device("cuda:1")
        elif not torch.cuda.is_available():
            self.ref_device = torch.device("cpu")
        else:
            self.ref_device = self.device

        # Safely setup reference model without duplicating GPU 0 VRAM
        if reference_model is not None:
            self.reference_model = reference_model.to(self.ref_device)
        else:
            model_cpu = copy.deepcopy(model).to("cpu")
            self.reference_model = model_cpu.to(self.ref_device)
        self.reference_model.eval()
        for p in self.reference_model.parameters():
            p.requires_grad = False

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Store initial reference weights theta_0 for deadzone calculation
        self.theta_0 = {name: p.clone().detach().cpu() for name, p in model.named_parameters()}
        
        # Quantization deadzone thresholds stored on CPU to preserve GPU VRAM during training
        if self.config.use_quant_floor:
            self.tau = {
                name: compute_quant_deadzones(p, gamma=self.config.gamma_quant).cpu()
                for name, p in model.named_parameters() if name in circuit_mask
            }
        else:
            self.tau = {}

        self.preconditioner = None

    def get_trainable_parameters(self) -> List[nn.Parameter]:
        """Returns strictly the circuit-active parameters for AdamW optimizer initialization."""
        return [p for p in self.model.parameters() if p.requires_grad]

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
        # Cast preconditioner to model dtype (bf16) to halve memory
        dtype = next(self.model.parameters()).dtype
        for name in self.preconditioner:
            self.preconditioner[name] = self.preconditioner[name].to(dtype=dtype, device=self.device)
        
        # Free raw fisher sensitivities and empty cache
        del fisher
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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
        loss_retain_val = loss_retain.item()
        loss_retain.backward()

        g_r = {name: p.grad.clone() for name, p in self.model.named_parameters() if p.grad is not None}
        optimizer.zero_grad()
        del out_r, loss_retain, r_ids, r_labels
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 2. Forget loss (SimNPO / GA on forget batch)
        f_ids = forget_batch["input_ids"].to(self.device)
        f_labels = forget_batch.get("labels", f_ids).to(self.device)
        
        # Reference pass on ref_device first (no grads, isolated)
        f_ids_ref = f_ids.to(self.ref_device)
        f_labels_ref = f_labels.to(self.ref_device)
        with torch.no_grad():
            out_f_ref = self.reference_model(input_ids=f_ids_ref, labels=f_labels_ref)
            logp_ref = -out_f_ref.loss.to(self.device)
        del out_f_ref, f_ids_ref, f_labels_ref

        # Target forget forward pass on self.device
        out_f = self.model(input_ids=f_ids, labels=f_labels)
        logp_curr = -out_f.loss
        diff = logp_curr - logp_ref
        loss_forget = -torch.log(torch.sigmoid(-self.config.beta * diff) + 1e-8)
        loss_forget_val = loss_forget.item()
        loss_forget.backward()
        del out_f, loss_forget, f_ids, f_labels

        # 3. Apply Fisher Preconditioning & Orthogonalization
        with torch.no_grad():
            for name, p in self.model.named_parameters():
                if p.grad is None or not p.requires_grad:
                    continue

                mask = self.circuit_mask.get(name, torch.ones_like(p)).to(self.device)
                prec = self.preconditioner[name].to(self.device) if (self.preconditioner and name in self.preconditioner) else mask

                grad_f = p.grad * prec
                grad_r = g_r.get(name, torch.zeros_like(p.grad)) * prec

                g_total, _ = orthogonalize_gradients(grad_f, grad_r, lambda_ret=self.config.lambda_ret)
                p.grad.copy_(g_total)

        optimizer.step()

        return {
            "loss_forget": loss_forget_val,
            "loss_retain": loss_retain_val
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
