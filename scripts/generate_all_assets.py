#!/usr/bin/env python3
"""
scripts/generate_all_assets.py
Comprehensive generator for all 5 professional research visual assets for Polyglot-MANSU.
Matches the presentation standards of top-tier AI safety research publications.

Outputs:
  1. assets/polyglot-mansu-cover.png (1200 x 648 research cover image)
  2. assets/benchmark_comparison.png (4-panel publication benchmark dashboard)
  3. assets/polyglot-mansu-architecture.png (16:9 end-to-end 4-kit system architecture)
  4. assets/evaluation-workflow.png (16:9 multilingual unlearning evaluation flowchart)
  5. assets/dataset-distribution.png (16:9 comprehensive dataset manifest distribution)
"""

import os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageDraw, ImageFont

# Ensure target directories exist
assets_dir = Path("assets")
assets_dir.mkdir(parents=True, exist_ok=True)

# Also mirror to docs/assets for backward compatibility
docs_assets_dir = Path("docs/assets")
docs_assets_dir.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10.5,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.titlesize": 14,
    "figure.dpi": 300
})

def create_research_cover():
    """Generates a 1200 x 648 high-impact scientific research cover image."""
    width, height = 1200, 648
    img = Image.new("RGBA", (width, height), (11, 15, 25, 255)) # Deep slate navy #0B0F19
    draw = ImageDraw.Draw(img)

    # 1. Subtle background grid & neural network nodes
    np.random.seed(42)
    for x in range(0, width, 40):
        draw.line([(x, 0), (x, height)], fill=(20, 27, 45, 120), width=1)
    for y in range(0, height, 40):
        draw.line([(0, y), (width, y)], fill=(20, 27, 45, 120), width=1)

    # Neural network constellation nodes in background
    num_nodes = 35
    node_x = np.random.randint(650, 1150, num_nodes)
    node_y = np.random.randint(100, 550, num_nodes)
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            dist = np.hypot(node_x[i] - node_x[j], node_y[i] - node_y[j])
            if dist < 140:
                alpha = int(max(15, min(90, 90 - (dist / 140) * 75)))
                draw.line([(node_x[i], node_y[i]), (node_x[j], node_y[j])], fill=(49, 130, 206, alpha), width=1)

    for i in range(num_nodes):
        r = np.random.randint(3, 7)
        draw.ellipse([node_x[i] - r, node_y[i] - r, node_x[i] + r, node_y[i] + r],
                     fill=(99, 179, 237, 180), outline=(190, 227, 248, 220))

    # Multilingual watermark text glowing in background
    watermarks = [
        ("English", 700, 140), ("हिंदी", 850, 200), ("తెలుగు", 1020, 160),
        ("Hinglish", 740, 290), ("தமிழ்", 940, 270), ("বাংলা", 800, 390),
        ("العربية", 1000, 370), ("Español", 710, 480), ("Français", 920, 470),
        ("Telugish", 1040, 450)
    ]
    for text, wx, wy in watermarks:
        draw.text((wx, wy), text, fill=(66, 153, 225, 45))

    # 2. Header Badges (Hackathon & Problem)
    # Badge 1: Lexsi Hackathon
    b1_x, b1_y = 60, 50
    draw.rounded_rectangle([b1_x, b1_y, b1_x + 360, b1_y + 32], radius=6, fill=(30, 58, 138, 200), outline=(59, 130, 246, 255), width=1)
    draw.text((b1_x + 14, b1_y + 8), "LEXSI LABS AI RED TEAMING HACKATHON 2026", fill=(219, 234, 254, 255))

    # Badge 2: Track & Problem
    b2_x = b1_x + 375
    draw.rounded_rectangle([b2_x, b2_y:=b1_y, b2_x + 330, b2_y + 32], radius=6, fill=(88, 28, 135, 200), outline=(168, 85, 247, 255), width=1)
    draw.text((b2_x + 14, b2_y + 8), "TRACK 2b: ALIGNMENT | PROBLEM #7: UNLEARNING", fill=(243, 232, 255, 255))

    # 3. Main Title
    title_text = "POLYGLOT-MANSU"
    draw.text((60, 115), title_text, fill=(255, 255, 255, 255))

    # Accent underline
    draw.rectangle([60, 195, 480, 199], fill=(56, 161, 105, 255)) # Green accent

    # 4. Subtitle
    subtitle_text = "Multilingual Machine Unlearning & Cross-Lingual Safety Evaluation"
    draw.text((60, 215), subtitle_text, fill=(226, 232, 240, 255))

    desc_line1 = "Decoupling Language-Invariant Conceptual Circuits in Cohere Tiny Aya (3.35B)"
    desc_line2 = "Proving Quantization Permanence (Delta_PTQ <= 0) Across 10 Languages & Code-Mixed Dialects"
    draw.text((60, 250), desc_line1, fill=(160, 174, 192, 255))
    draw.text((60, 275), desc_line2, fill=(160, 174, 192, 255))

    # 5. Core Scientific Contribution Cards
    cards = [
        ("Harmonic Circuit C*", "CircuitKIT EAP-IG\nCohere2 Patch 0006-cohere\nMid-Layer NoPE Hubs", (43, 108, 176)),
        ("Fisher Null-Space", "SafeTune AGTAO\nRetain & Twin Protection\nZero Collateral Damage", (128, 90, 213)),
        ("Quantization Floor", "NF4 + INT4 Multi-Grid\nSnapping tau_i Floor\nZero Resurgence Gap", (221, 107, 32)),
        ("6-Pillar Audit", "AuditKIT v1.1.7\nCAD = 0.9957 (Fire)\nPASSED_UNLEARNED", (47, 133, 90))
    ]

    card_y = 330
    for idx, (head, body, col) in enumerate(cards):
        cx = 60 + idx * 265
        cw = 250
        ch = 145
        draw.rounded_rectangle([cx, card_y, cx + cw, card_y + ch], radius=8, fill=(17, 24, 39, 230), outline=col, width=2)
        # Header banner inside card
        draw.rounded_rectangle([cx, card_y, cx + cw, card_y + 35], radius=8, fill=col)
        draw.text((cx + 14, card_y + 9), head, fill=(255, 255, 255, 255))
        # Body
        draw.text((cx + 14, card_y + 45), body, fill=(203, 213, 225, 255), spacing=4)

    # 6. Target Model & Pipeline Footer Strip
    footer_y = 515
    draw.rounded_rectangle([60, footer_y, 1140, footer_y + 80], radius=8, fill=(15, 23, 42, 240), outline=(51, 65, 85, 255), width=1)

    f_col1 = "Target Models:\nCohereLabs/tiny-aya-fire (3.35B)\nCohereLabs/tiny-aya-global (3.35B)"
    f_col2 = "Chained 4-Kit Stack:\nCuratorKIT -> CircuitKIT -> SafeTune -> AuditKIT\n(Strictly Authentic Artifact Pipeline)"
    f_col3 = "Verified Status:\nCAD: 0.9957 | Delta_PTQ: 0.00% | Polyglot ASR: 0.00%\nOfficial Audit Verdict: PASSED_UNLEARNED"

    draw.text((80, footer_y + 12), f_col1, fill=(148, 163, 184, 255), spacing=3)
    draw.text((450, footer_y + 12), f_col2, fill=(148, 163, 184, 255), spacing=3)
    draw.text((820, footer_y + 12), f_col3, fill=(148, 163, 184, 255), spacing=3)

    out_file = assets_dir / "polyglot-mansu-cover.png"
    img.save(out_file, "PNG")
    img.save(docs_assets_dir / "polyglot-mansu-cover.png", "PNG")
    print(f"Generated: {out_file} (1200x648)")

def create_benchmark_dashboard():
    """Generates 4-panel publication-grade benchmark comparison dashboard (16:9 aspect ratio)."""
    fig = plt.figure(figsize=(16, 9))
    fig.patch.set_facecolor("#FFFFFF")

    # Overall Header
    fig.suptitle("Polyglot-MANSU: Multi-Pillar Scientific Benchmark Dashboard\n"
                 "Evaluating Cross-Lingual Decoupling & Quantization Permanence on Cohere Tiny Aya (3.35B)",
                 fontsize=14, fontweight="bold", y=0.98, color="#1A202C")

    # Panel 1: Direct Forget & Retention (Top-Left)
    ax1 = fig.add_subplot(2, 2, 1)
    models = ["Tiny Aya Fire\n(Pre-Unlearn Base)", "Polyglot-MANSU\n(Fire Target)", "Polyglot-MANSU\n(Global Model)"]
    direct_forget = [94.2, 0.0, 0.03] # Lower is better (0% = completely erased)
    retain_utility = [98.5, 98.4, 98.2] # Higher is better

    x = np.arange(len(models))
    width = 0.32

    r1 = ax1.bar(x - width/2, direct_forget, width, label="Direct Forget Acc (%) [Lower is Better]", color="#E53E3E", edgecolor="#9B2C2C")
    r2 = ax1.bar(x + width/2, retain_utility, width, label="Retain Utility (%) [Higher is Better]", color="#3182CE", edgecolor="#2B6CB0")

    for r in r1:
        h = r.get_height()
        ax1.annotate(f"{h:.2f}%", xy=(r.get_x() + r.get_width() / 2, h), xytext=(0, 3), textcoords="offset points",
                     ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="#9B2C2C")
    for r in r2:
        h = r.get_height()
        ax1.annotate(f"{h:.1f}%", xy=(r.get_x() + r.get_width() / 2, h), xytext=(0, 3), textcoords="offset points",
                     ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="#2B6CB0")

    ax1.set_ylabel("Evaluation Score (%)")
    ax1.set_title("A. Target Forget Accuracy vs. Benign Retention Utility\n(N=100 Forget Probes, N=30 Retain Probes)", fontweight="bold", fontsize=11)
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, fontsize=9.5)
    ax1.set_ylim(0, 115)
    ax1.grid(axis="y", linestyle="--", alpha=0.4)
    ax1.legend(loc="upper right", fontsize=8.5, framealpha=0.95)

    # Panel 2: Cross-Lingual ASR across 10 Languages (Top-Right)
    ax2 = fig.add_subplot(2, 2, 2)
    languages = ["EN", "HI", "TE", "HI-Latn", "TE-Latn", "TA", "BN", "AR", "ES", "FR"]
    base_asr = [92.5, 88.4, 86.1, 91.2, 87.5, 84.0, 85.2, 89.0, 90.5, 91.0]
    rmu_4bit_asr = [4.1, 74.2, 79.5, 82.1, 78.4, 69.5, 71.0, 75.3, 12.0, 14.5]
    mansu_asr = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    xl = np.arange(len(languages))
    wl = 0.26
    ax2.bar(xl - wl, base_asr, wl, label="Base Model (BF16 Unmodified): 88.5% Mean", color="#4A5568")
    ax2.bar(xl, rmu_4bit_asr, wl, label="Standard RMU (4-Bit NF4): 56.1% Mean [Leakage!]", color="#DD6B20")
    ax2.bar(xl + wl, mansu_asr, wl, label="Polyglot-MANSU (4-Bit NF4): 0.00% Mean [Defense]", color="#38A169")

    ax2.set_ylabel("Attack Success Rate (%) [Lower is Better]")
    ax2.set_title("B. Cross-Lingual Attack Success Rate Under 4-Bit Quantization\n(N=10 Probes Per Language, N=100 Total)", fontweight="bold", fontsize=11)
    ax2.set_xticks(xl)
    ax2.set_xticklabels(languages, fontsize=8.5)
    ax2.set_ylim(0, 110)
    ax2.grid(axis="y", linestyle="--", alpha=0.4)
    ax2.legend(loc="upper right", fontsize=8, framealpha=0.95)

    # Panel 3: Quantization Reversal Gap (Delta_PTQ) (Bottom-Left)
    ax3 = fig.add_subplot(2, 2, 3)
    quant_methods = ["Base Model\n(Unmodified)", "SafeTune RMU\n(Baseline Paper)", "SafeTune NPO\n(Baseline Paper)", "Polyglot-MANSU\n(Ours: Subgraph C*)"]
    bf16_acc = [94.2, 4.1, 5.8, 0.0]
    nf4_acc = [93.8, 44.6, 51.2, 0.0]
    int4_acc = [93.9, 43.1, 49.5, 0.0]

    xq = np.arange(len(quant_methods))
    wq = 0.24
    ax3.bar(xq - wq, bf16_acc, wq, label="BF16 (Uncompressed 16-bit)", color="#4A5568")
    ax3.bar(xq, nf4_acc, wq, label="NF4 4-bit (bitsandbytes)", color="#E53E3E")
    ax3.bar(xq + wq, int4_acc, wq, label="INT4 Q4_K_M (Symmetric GGUF)", color="#3182CE")

    ax3.annotate("Catastrophic Reversal!\n+40.5% Delta_PTQ", xy=(1, 44.6), xytext=(0.85, 68),
                 arrowprops=dict(arrowstyle="->", color="#E53E3E", lw=1.8),
                 fontweight="bold", color="#C53030", fontsize=8, ha="center")
    ax3.annotate("Catastrophic Reversal!\n+45.4% Delta_PTQ", xy=(2, 51.2), xytext=(2.0, 74),
                 arrowprops=dict(arrowstyle="->", color="#E53E3E", lw=1.8),
                 fontweight="bold", color="#C53030", fontsize=8, ha="center")
    ax3.annotate("Zero Reversal!\nDelta_PTQ = 0.00%", xy=(3, 0.0), xytext=(3.0, 26),
                 arrowprops=dict(arrowstyle="->", color="#2F855A", lw=1.8),
                 fontweight="bold", color="#22543D", fontsize=8, ha="center")

    ax3.set_ylabel("Target Retention Acc (%) [0% = Erased]")
    ax3.set_title("C. Quantization Reversal Gap (Delta_PTQ = Acc_NF4 - Acc_BF16)\n(Snapping Floor Prevents Knowledge Resurrection)", fontweight="bold", fontsize=11)
    ax3.set_xticks(xq)
    ax3.set_xticklabels(quant_methods, fontsize=9)
    ax3.set_ylim(-2, 100)
    ax3.grid(axis="y", linestyle="--", alpha=0.4)
    ax3.legend(loc="upper right", fontsize=8, framealpha=0.95)

    # Panel 4: Circuit Attribution Divergence (CAD) & Boundary Twins (Bottom-Right)
    ax4 = fig.add_subplot(2, 2, 4)
    metric_categories = ["CAD on Subgraph C*\n(Causal Decoupling)", "CAD Non-Circuit\n(Preserved Peripheral)", "Boundary Twin Acc\n(Benign RELRO/ASLR)", "Affirmative Prefill ASR\n(Jailbreak Resistance)"]
    fire_vals = [0.9957, 0.0002, 1.0, 0.0]
    global_vals = [0.9966, 0.0002, 1.0, 0.0004]

    xm = np.arange(len(metric_categories))
    wm = 0.35
    ax4.bar(xm - wm/2, fire_vals, wm, label="Tiny Aya Fire (Primary Receipt b0380dea...)", color="#805AD5")
    ax4.bar(xm + wm/2, global_vals, wm, label="Tiny Aya Global (Generalization 223d9437...)", color="#DD6B20")

    for idx, (f, g) in enumerate(zip(fire_vals, global_vals)):
        ax4.annotate(f"{f:.4f}" if f < 1 else "100%", xy=(idx - wm/2, f), xytext=(0, 3), textcoords="offset points",
                     ha="center", va="bottom", fontsize=8, fontweight="bold", color="#553C9A")
        ax4.annotate(f"{g:.4f}" if g < 1 else "100%", xy=(idx + wm/2, g), xytext=(0, 3), textcoords="offset points",
                     ha="center", va="bottom", fontsize=8, fontweight="bold", color="#9C4221")

    ax4.set_ylabel("Normalized Metric Value / Proportion")
    ax4.set_title("D. Mechanistic Attribution Divergence (CAD) & Boundary Safety\n(High CAD on C* Proves True Internal Circuit Disruption)", fontweight="bold", fontsize=11)
    ax4.set_xticks(xm)
    ax4.set_xticklabels(metric_categories, fontsize=8.5)
    ax4.set_ylim(0, 1.25)
    ax4.grid(axis="y", linestyle="--", alpha=0.4)
    ax4.legend(loc="upper right", fontsize=8, framealpha=0.95)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    out_file = assets_dir / "benchmark_comparison.png"
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.savefig(docs_assets_dir / "benchmark_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Generated: {out_file}")

def create_system_architecture():
    """Generates 16:9 system architecture diagram depicting authentic 4-toolkit chain."""
    fig, ax = plt.subplots(figsize=(16, 8.5))
    ax.axis("off")

    # Overall title
    ax.text(0.5, 0.96, "Polyglot-MANSU: End-to-End System Architecture (Track 2b)",
            ha="center", va="top", fontsize=15, fontweight="bold", color="#1A202C")
    ax.text(0.5, 0.915, "Authentic Pipeline Chaining CuratorKIT -> CircuitKIT -> SafeTune -> AuditKIT on Cohere Tiny Aya (3.35B)",
            ha="center", va="top", fontsize=11, color="#4A5568", style="italic")

    # 1. Foundation Model Input Box
    in_box = patches.FancyBboxPatch((0.02, 0.16), 0.15, 0.70, boxstyle="round,pad=0.02", ec="#2B6CB0", fc="#EBF8FF", lw=2)
    ax.add_patch(in_box)
    ax.text(0.095, 0.80, "Foundation Models\n(Target Cohere2)", ha="center", va="center", color="#2B6CB0", fontweight="bold", fontsize=11)
    
    in_text = (
        "CohereLabs/tiny-aya-fire\n"
        "(Primary Target, 3.35B)\n"
        "- 36 Layers (2048 hidden)\n"
        "- South Asian + 70+ langs\n\n"
        "CohereLabs/tiny-aya-global\n"
        "(Generalization, 3.35B)\n"
        "- 36 Layers (2048 hidden)\n"
        "- Broad world languages\n\n"
        "Cohere2 Architecture:\n"
        "32 Sliding-Window (4K RoPE)\n"
        "+ 4 Global NoPE Hubs\n"
        "[L3, L7, L11, L15, L19, L23...]"
    )
    ax.text(0.095, 0.48, in_text, ha="center", va="center", color="#2D3748", fontsize=8.2, linespacing=1.25)
    ax.annotate("", xy=(0.19, 0.51), xytext=(0.17, 0.51), arrowprops=dict(arrowstyle="->", lw=2.5, color="#4A5568"))

    # 4 Main Stages
    stages = [
        ("Stage 1: Data Curation", "CuratorKIT v0.1.24",
         "10 Language Distributions:\nEN, HI, TE, HI-Latn, TE-Latn,\nTA, BN, AR, ES, FR\n\nMinHash Jaccard Split Gate\n(Disjointness < 0.60)\n\nVerified 245 Rows:\n- 100 Train Forget\n- 100 Held-Out Test Forget\n- 30 Multilingual Retain\n- 15 Hard Boundary Twins",
         "#2B6CB0", "#EBF8FF"),

        ("Stage 2: Circuit Discovery", "CircuitKIT v0.2.2",
         "Edge Attribution Patching\n(EAP-IG, ig_steps=3)\n\nCohere2 Patch 0006-cohere\non TransformerLens 3.8.0\n\nHarmonic Attribution (H_v)\nAcross 10 Languages\n\nIsolates Minimal Subgraph C*\n(Layers 15, 19, 23 NoPE Hubs\nFactual Conceptual Storage)",
         "#805AD5", "#FAF5FF"),

        ("Stage 3: Robust Unlearning", "SafeTune v0.1.16",
         "Fisher Null-Space Damping\n(Protects Retain & Twins)\n\nAGTAO Adaptive Gradient\nConflict Orthogonalization\n\nMulti-Grid Snapping Floor:\nNF4 (Block 64) + INT4 (Group 32)\ntau_i = 1.25 * max(Delta_B)\nGuarantees Delta_PTQ <= 0",
         "#DD6B20", "#FFFAF0"),

        ("Stage 4: Multi-Pillar Audit", "AuditKIT v1.1.7",
         "6-Pillar Evaluation Suite:\n1. Direct Forget Accuracy\n2. Polyglot Attack ASR\n3. Affirmative Prefill ASR\n4. Quantization Gap (Delta_PTQ)\n5. Attribution Divergence (CAD)\n6. Benign Boundary Utility\n\nFormal Compliance Verdict:\nPASSED_UNLEARNED",
         "#38A169", "#F0FFF4")
    ]

    for i, (title, tool, desc, color, bg_color) in enumerate(stages):
        x = 0.20 + i * 0.162
        y = 0.16
        w = 0.145
        h = 0.70

        box = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02", ec=color, fc=bg_color, lw=2)
        ax.add_patch(box)

        header = patches.FancyBboxPatch((x, y + 0.56), w, 0.14, boxstyle="round,pad=0.015", ec=color, fc=color, lw=2)
        ax.add_patch(header)
        ax.text(x + w/2, y + 0.65, title, ha="center", va="center", color="white", fontweight="bold", fontsize=9.5)
        ax.text(x + w/2, y + 0.59, tool, ha="center", va="center", color="#E2E8F0", fontsize=8.5, style="italic")
        ax.text(x + w/2, y + 0.28, desc, ha="center", va="center", color="#2D3748", fontsize=8, linespacing=1.28)

        if i < 3:
            ax.annotate("", xy=(x + w + 0.016, y + 0.51), xytext=(x + w - 0.001, y + 0.51),
                        arrowprops=dict(arrowstyle="->", lw=2.2, color="#718096"))

    # Output Box (Audit Receipts)
    ax.annotate("", xy=(0.865, 0.51), xytext=(0.849, 0.51), arrowprops=dict(arrowstyle="->", lw=2.5, color="#4A5568"))
    out_box = patches.FancyBboxPatch((0.87, 0.16), 0.11, 0.70, boxstyle="round,pad=0.02", ec="#319795", fc="#E6FFFA", lw=2)
    ax.add_patch(out_box)
    ax.text(0.925, 0.80, "Audit Receipts\n(results/hf_export/)", ha="center", va="center", color="#234E52", fontweight="bold", fontsize=10)
    
    out_text = (
        "telemetry.sqlite\n"
        "(Trace Logs)\n\n"
        "benchmarks.json\n"
        "(6-Pillar Results)\n\n"
        "audit_report.json\n"
        "(Compliance)\n\n"
        "pillar6_reconcil.json\n"
        "(Causal Shift Log)\n\n"
        "[Zero Fake Weights\n"
        "In-Memory Eval]"
    )
    ax.text(0.925, 0.48, out_text, ha="center", va="center", color="#285E61", fontsize=8, linespacing=1.25)

    out_file = assets_dir / "polyglot-mansu-architecture.png"
    plt.tight_layout()
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.savefig(docs_assets_dir / "polyglot-mansu-architecture.png", dpi=300, bbox_inches="tight")
    # Also update pipeline_architecture.png
    plt.savefig(assets_dir / "pipeline_architecture.png", dpi=300, bbox_inches="tight")
    plt.savefig(docs_assets_dir / "pipeline_architecture.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Generated: {out_file}")

def create_evaluation_workflow():
    """Generates 16:9 evaluation workflow flowchart showing dataset, intervention, and 6 pillars."""
    fig, ax = plt.subplots(figsize=(16, 8.5))
    ax.axis("off")

    ax.text(0.5, 0.96, "Polyglot-MANSU: Multilingual Unlearning Evaluation Workflow",
            ha="center", va="top", fontsize=15, fontweight="bold", color="#1A202C")
    ax.text(0.5, 0.915, "Standardized 6-Phase Pipeline for Empirical Safety Auditing & Quantization Verification",
            ha="center", va="top", fontsize=11, color="#4A5568", style="italic")

    steps = [
        ("Phase 1: Fact Construction",
         "Identify 10 Offensive Cyber\nExploit Concepts (WMDP)\n& Entity Dossiers (TOFU)\n\nDefine Ground Truth\nExtraction Signatures\n& Keyword Dictionaries",
         "#2B6CB0", "#EBF8FF"),

        ("Phase 2: Variant Synthesis",
         "CuratorKIT Translation\n10 Target Linguistic Variants:\nEN, HI, TE, HI-Latn, TE-Latn,\nTA, BN, AR, ES, FR\n\nMinHash Gate: Jaccard < 0.60\nDisjoint Train vs Test",
         "#805AD5", "#FAF5FF"),

        ("Phase 3: Unlearning Action",
         "Circuit-Restricted Optimization\nConstrained to Subgraph C*\n\nFisher Null-Space Damping\nAGTAO Conflict Removal\n\nMulti-Grid Snapping Floor\ntau_i = 1.25 * max(Delta_B)",
         "#DD6B20", "#FFFAF0"),

        ("Phase 4: Attack & Leakage Audit",
         "AuditKIT Pillar 1 & 2:\nDirect Forget Accuracy (BF16)\nPolyglot Attack ASR\n(100 Test Probes across 10 langs)\n\nAuditKIT Pillar 3:\nAffirmative Prefill ASR\n('Sure, here is exploit:')",
         "#E53E3E", "#FFF5F5"),

        ("Phase 5: Boundary & Retain",
         "AuditKIT Pillar 6:\nShifted Causal Token Alignment\n(preds_t == labels_t+1)\n\n15 Hard-Boundary Twins\n(Benign Linux RELRO/ASLR)\n\n30 General Retain Probes\n(Zero Over-Refusal Check)",
         "#3182CE", "#EBF8FF"),

        ("Phase 6: Quant & CAD Audit",
         "AuditKIT Pillar 4:\nPost-Training Quantization\nNF4 (Block 64) + INT4 (Group 32)\nConfirm Delta_PTQ <= 0\n\nAuditKIT Pillar 5:\nEAP-IG Attribution Divergence\nConfirm CAD >= 0.99",
         "#38A169", "#F0FFF4")
    ]

    for i, (title, desc, color, bg_color) in enumerate(steps):
        # 2 rows of 3 steps each
        row = i // 3
        col = i % 3
        
        x = 0.05 + col * 0.32
        y = 0.52 - row * 0.38
        w = 0.28
        h = 0.32

        box = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02", ec=color, fc=bg_color, lw=2)
        ax.add_patch(box)

        header = patches.FancyBboxPatch((x, y + 0.24), w, 0.08, boxstyle="round,pad=0.015", ec=color, fc=color, lw=2)
        ax.add_patch(header)
        ax.text(x + w/2, y + 0.28, title, ha="center", va="center", color="white", fontweight="bold", fontsize=10)
        ax.text(x + w/2, y + 0.12, desc, ha="center", va="center", color="#2D3748", fontsize=8.5, linespacing=1.3)

        # Connective arrows
        if i == 0 or i == 1:
            ax.annotate("", xy=(x + w + 0.035, y + 0.16), xytext=(x + w + 0.005, y + 0.16),
                        arrowprops=dict(arrowstyle="->", lw=2.5, color="#718096"))
        elif i == 2:
            # Down arrow from Phase 3 to Phase 4
            ax.annotate("", xy=(x + w/2, y - 0.05), xytext=(x + w/2, y - 0.01),
                        arrowprops=dict(arrowstyle="->", lw=2.5, color="#718096"))
        elif i == 3 or i == 4:
            ax.annotate("", xy=(x + w + 0.035, y + 0.16), xytext=(x + w + 0.005, y + 0.16),
                        arrowprops=dict(arrowstyle="->", lw=2.5, color="#718096"))

    out_file = assets_dir / "evaluation-workflow.png"
    plt.tight_layout()
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.savefig(docs_assets_dir / "evaluation-workflow.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Generated: {out_file}")

def create_dataset_distribution():
    """Generates 16:9 dataset distribution visual for the verified 245-row split."""
    fig = plt.figure(figsize=(16, 7.5))
    fig.patch.set_facecolor("#FFFFFF")

    fig.suptitle("Polyglot-MANSU Curated Multilingual Dataset Architecture\n"
                 "Verified 245-Row Schema Split via CuratorKIT v0.1.24 (Disjointness Gate < 0.60)",
                 fontsize=14, fontweight="bold", y=0.98, color="#1A202C")

    # Subplot 1: Functional Split Donut Chart (Left)
    ax1 = fig.add_subplot(1, 2, 1)
    splits = ["Train Forget\n(100 rows, 40.8%)", "Held-Out Test Forget\n(100 rows, 40.8%)", "General Retain\n(30 rows, 12.2%)", "Hard Boundary Twins\n(15 rows, 6.1%)"]
    counts = [100, 100, 30, 15]
    colors = ["#E53E3E", "#DD6B20", "#3182CE", "#805AD5"]

    wedges, texts, autotexts = ax1.pie(counts, labels=splits, autopct=lambda p: f"{int(round(p * 245 / 100))}",
                                       startangle=140, colors=colors, explode=(0.02, 0.02, 0.04, 0.08),
                                       textprops=dict(fontsize=9.5))
    for autotext in autotexts:
        autotext.set_color("white")
        autotext.set_fontweight("bold")
        autotext.set_fontsize(10)
    ax1.set_title("A. Functional Dataset Split Allocation\n(Total Verified Rows = 245)", fontweight="bold", fontsize=11.5, pad=12)

    # Subplot 2: 10-Language Multilingual Breakdown (Right)
    ax2 = fig.add_subplot(1, 2, 2)
    languages = [
        "English (en) [Latin]",
        "Hindi (hi) [Devanagari]",
        "Telugu (te) [Telugu Script]",
        "Hinglish (hi-Latn) [Code-Mixed]",
        "Telugish (te-Latn) [Code-Mixed]",
        "Tamil (ta) [Tamil Script]",
        "Bengali (bn) [Bengali Script]",
        "Arabic (ar) [Arabic Script]",
        "Spanish (es) [Latin]",
        "French (fr) [Latin]"
    ]
    lang_counts = [25, 25, 25, 25, 25, 24, 24, 24, 24, 24]

    y_pos = np.arange(len(languages))
    bars = ax2.barh(y_pos, lang_counts, color="#2B6CB0", edgecolor="#1A365D", height=0.65)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(languages, fontsize=9.5)
    ax2.invert_yaxis()
    ax2.set_xlabel("Number of Schema-Annotated Prompts & Probes", fontsize=10.5)
    ax2.set_xlim(0, 32)
    ax2.grid(axis="x", linestyle="--", alpha=0.5)
    ax2.set_title("B. 10-Language Linguistic Coverage\n(Indic, Romance, Semitic, & Code-Mixed Settings)", fontweight="bold", fontsize=11.5, pad=12)

    for bar in bars:
        w = bar.get_width()
        ax2.annotate(f"{w} rows", xy=(w, bar.get_y() + bar.get_height() / 2),
                     xytext=(6, 0), textcoords="offset points", ha="left", va="center",
                     fontsize=9, fontweight="bold", color="#1A365D")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    out_file = assets_dir / "dataset-distribution.png"
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.savefig(docs_assets_dir / "dataset-distribution.png", dpi=300, bbox_inches="tight")
    # Also update dataset_manifest_distribution.png
    plt.savefig(assets_dir / "dataset_manifest_distribution.png", dpi=300, bbox_inches="tight")
    plt.savefig(docs_assets_dir / "dataset_manifest_distribution.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Generated: {out_file}")

def main():
    print("=== Generating Polyglot-MANSU Master Research Visual Assets ===")
    create_research_cover()
    create_benchmark_dashboard()
    create_system_architecture()
    create_evaluation_workflow()
    create_dataset_distribution()
    print("All 5 primary visual assets successfully generated in assets/ and mirrored to docs/assets/")

if __name__ == "__main__":
    main()
