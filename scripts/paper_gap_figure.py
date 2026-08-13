"""
Five findings in the codebase that are not in the paper.
Unified figure showing the gap between submitted and available evidence.
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# Project palette
TEAL = "#0C6583"
AMBER = "#FFBA08"
GREEN = "#2CA02C"
GRAY = "#AAAAAA"
NAVY = "#2B2D42"
ORANGE = "#EE7733"
BG = "#f5f5f5"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})

fig = plt.figure(figsize=(16, 11))
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.50, wspace=0.35,
                       left=0.06, right=0.97, top=0.88, bottom=0.06)

fig.text(0.5, 0.97, "Five findings not in the paper", fontsize=16,
         ha="center", va="top", color=NAVY)
fig.text(0.5, 0.935, "Each strengthens the contribution from descriptive to explanatory",
         fontsize=10, ha="center", va="top", color=GRAY)

# Panel A: FIM separates difficulty 4.6x.
ax1 = fig.add_subplot(gs[0, 0])
methods = [
    "Issue text\n(k-means, k=10)",
    "Predicted fix\n(k-means, k=10)",
    "Fix from traces\n(k-means, k=10)",
    "AST cert\n(decision tree, 10)",
    "FIM closed\n(itemsets, 15)",
]
variances = [0.0073, 0.0087, 0.0083, 0.0257, 0.0333]
colors = [GRAY, GRAY, GRAY, AMBER, TEAL]
bars = ax1.barh(range(len(methods)), variances, color=colors, height=0.65, edgecolor="white")
ax1.set_yticks(range(len(methods)))
ax1.set_yticklabels(methods, fontsize=8.5)
ax1.set_xlabel("Variance of per-group mean agent ease", fontsize=9)
ax1.set_title("A. FIM separates difficulty 4.6x better", fontsize=11, color=NAVY, pad=10)
for i, (v, c) in enumerate(zip(variances, colors)):
    ax1.text(v + 0.0005, i, f"{v:.4f}", va="center", fontsize=8,
             color=NAVY if c != GRAY else "#666666")
ax1.set_xlim(0, 0.042)
ax1.invert_yaxis()

# Panel B: Composition failures.
ax2 = fig.add_subplot(gs[0, 1])
categories = ["Novel\nprimitive", "Novel\ncomposition", "Familiar"]
fractions = [21.3, 45.8, 32.9]
bar_colors = [TEAL, AMBER, GREEN]
bars2 = ax2.bar(categories, fractions, color=bar_colors, width=0.6, edgecolor="white")
for bar, val in zip(bars2, fractions):
    ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.2,
             f"{val}%", ha="center", va="bottom", fontsize=9, color=NAVY)
ax2.set_ylabel("Mean fraction of failures", fontsize=9)
ax2.set_ylim(0, 100)
ax2.set_title("B. 45.8% of failures are composition failures", fontsize=11,
              color=NAVY, pad=10)
ax2.tick_params(axis="x", labelsize=9)

# Panel C: Grounding failure F1=0.20.
ax3 = fig.add_subplot(gs[0, 2])
rep_labels = [
    "FIM pattern\noverlap",
    "Edit cert\n(Jaccard)",
    "Motif\ndistance",
    "Staged\nnarrative",
    "CoT free-form\n(embedding)",
    "Fix type\n(13 classes)",
]
f1_scores = [0.265, 0.205, 0.135, 0.125, 0.075, 0.0]
rep_colors = [GREEN, TEAL, TEAL, AMBER, AMBER, GRAY]
bars3 = ax3.bar(range(len(rep_labels)), f1_scores, color=rep_colors,
                width=0.65, edgecolor="white")
ax3.set_xticks(range(len(rep_labels)))
ax3.set_xticklabels(rep_labels, fontsize=7.5, rotation=0)
ax3.set_ylabel("F1 at k=5", fontsize=9)
ax3.set_title("C. Structural representations outperform self-report", fontsize=11,
              color=NAVY, pad=10)
ax3.set_ylim(0, 1.0)
# Annotate the grounding finding
ax3.annotate("Agent self-report\nF1 = 0.20",
             xy=(4, 0.075), xytext=(4.5, 0.24),
             fontsize=8, color=ORANGE, ha="center",
             arrowprops=dict(arrowstyle="->", color=ORANGE, lw=0.8))

# Panel D: Scoped certificates (localization bottleneck).
ax4 = fig.add_subplot(gs[1, 0])
pairs = ["C3.5/G4", "C3.5/G4o", "C3O/C3.5", "C3O/G4", "C3O/G4o", "G4/G4o"]
file_agree = [0.76, 0.76, 0.65, 0.61, 0.69, 0.68]
edit_agree = [0.45, 0.44, 0.46, 0.54, 0.51, 0.51]
scope_agree = [0.26, 0.28, 0.25, 0.26, 0.28, 0.31]

x = np.arange(len(pairs))
w = 0.25
ax4.bar(x - w, file_agree, w, color=TEAL, label="File agreement", edgecolor="white")
ax4.bar(x, edit_agree, w, color=AMBER, label="Edit Jaccard", edgecolor="white")
ax4.bar(x + w, scope_agree, w, color=GREEN, label="Scope Jaccard", edgecolor="white")
ax4.set_xticks(x)
ax4.set_xticklabels(pairs, fontsize=8)
ax4.set_ylabel("Mean agreement", fontsize=9)
ax4.set_ylim(0, 1.0)
ax4.set_title("D. Bottleneck is localization, not edit strategy", fontsize=11,
              color=NAVY, pad=10)
ax4.legend(fontsize=8, loc="upper right", frameon=False, ncol=1)

# Draw the "gap" annotation
mid = len(pairs) / 2 - 0.5
ax4.annotate("", xy=(mid, 0.28), xytext=(mid, 0.68),
             arrowprops=dict(arrowstyle="<->", color=ORANGE, lw=1.5))
ax4.text(mid + 0.15, 0.48, "gap", fontsize=9, color=ORANGE, va="center")

# Panel E: Semantic independence (ARI and variance).
ax5 = fig.add_subplot(gs[1, 1])
# Show ARI near zero + variance comparison as a combined view
groupings = [
    "Semantic topic\nvs structural form",
]
ari_vals = [0.010]
ax5_left = ax5

# Make this a simpler, more impactful display
ax5.set_xlim(0, 1)
ax5.set_ylim(0, 1)
ax5.axis("off")
ax5.set_title("E. Structure and semantics are independent", fontsize=11,
              color=NAVY, pad=10)

# Big ARI number
ax5.text(0.5, 0.72, "ARI = 0.010", fontsize=28, ha="center", va="center",
         color=TEAL, fontweight="regular")
ax5.text(0.5, 0.55, "Adjusted Rand Index between structural form\nand semantic topic groupings",
         fontsize=9, ha="center", va="center", color="#666666")
ax5.text(0.5, 0.42, "(0 = random chance, 1 = identical groupings)",
         fontsize=8, ha="center", va="center", color=GRAY)

# Comparison line
ax5.plot([0.15, 0.85], [0.28, 0.28], color="#dddddd", lw=0.5)

# Below: the variance punch line
ax5.text(0.5, 0.17, "No semantic method separates difficulty",
         fontsize=10, ha="center", va="center", color=NAVY)
ax5.text(0.5, 0.07,
         "Issue text 0.0073  |  Predicted fix 0.0087  |  Fix from traces 0.0083",
         fontsize=8, ha="center", va="center", color=GRAY)

# Panel F: Scale (3 models -> 84 agents).
ax6 = fig.add_subplot(gs[1, 2])
ax6.set_xlim(0, 1)
ax6.set_ylim(0, 1)
ax6.axis("off")
ax6.set_title("F. Scale changes the claim", fontsize=11, color=NAVY, pad=10)

# Paper vs codebase comparison
paper_stats = [
    ("Models compared", "3", "24"),
    ("Agent ease data", "n/a", "84 agents"),
    ("Benchmarks validated", "1", "3 + BugsInPy"),
    ("Structural forms", "ad hoc", "15 FIM patterns"),
    ("Failure taxonomy", "none", "3 categories, 84 agents"),
]

y_pos = 0.82
for label, paper_val, code_val in paper_stats:
    ax6.text(0.02, y_pos, label, fontsize=9, va="center", color="#666666")
    ax6.text(0.58, y_pos, paper_val, fontsize=9, va="center", color=GRAY,
             ha="center")
    ax6.text(0.85, y_pos, code_val, fontsize=9, va="center", color=TEAL,
             ha="center")
    y_pos -= 0.14

# Column headers
ax6.text(0.58, 0.96, "Paper", fontsize=9, va="center", ha="center", color=GRAY)
ax6.text(0.85, 0.96, "Codebase", fontsize=9, va="center", ha="center", color=TEAL)

# Divider
ax6.plot([0.0, 1.0], [0.92, 0.92], color="#dddddd", lw=0.5)

out = "/Users/hamidaho/learning-from-dev/bidirect-align-dev-traces/figures/paper_gap_summary.png"
fig.savefig(out, dpi=200, facecolor="white")
print(f"Saved to {out}")
plt.close()
