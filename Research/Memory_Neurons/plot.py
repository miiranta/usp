import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ── Config ────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
PLOTS_DIR  = os.path.join(BASE_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)


def pretty(name):
    """Human-readable label: 'gelu2_k4_attn' → 'K=4 +attn'."""
    if name == "control":
        return "Control"
    base   = name.replace("_attn", "")
    suffix = " +attn" if name.endswith("_attn") else ""
    if base.startswith("gelu2_k"):
        return f"K={base[7:]}{suffix}"
    return name


# ── Load all experiment metrics ───────────────────────────────────────
def exp_sort_key(name):
    if name == "control":
        return (0, 0, 0, "")
    base      = name.replace("_attn", "")
    attn_flag = 1 if name.endswith("_attn") else 0
    if base.startswith("gelu2_k"):
        try:
            return (1, int(base[7:]), attn_flag, name)
        except ValueError:
            pass
    return (2, 0, 0, name)


exp_names_sorted = sorted(
    [e for e in os.listdir(OUTPUT_DIR)
     if os.path.isfile(os.path.join(OUTPUT_DIR, e, "metrics.csv"))],
    key=exp_sort_key
)

records = []
for exp_name in exp_names_sorted:
    df = pd.read_csv(os.path.join(OUTPUT_DIR, exp_name, "metrics.csv"))
    df["experiment"] = exp_name
    df["label"]      = pretty(exp_name)
    records.append(df)

if not records:
    print("No metrics.csv files found in output/. Run train.py first.")
    exit(1)

data = pd.concat(records, ignore_index=True)

# ── Save training curves data ─────────────────────────────────────────
data.to_csv(os.path.join(PLOTS_DIR, "training_curves.csv"), index=False)
print(f"Saved training data: {os.path.join(PLOTS_DIR, 'training_curves.csv')}")

# ── Training curves (2×2) ─────────────────────────────────────────────
sns.set_theme(style="darkgrid")
label_order = [pretty(e) for e in exp_names_sorted]

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle("Training Curves — WikiText-2", fontsize=14, fontweight="bold")

for ax, col, ylabel, title in [
    (axes[0, 0], "train_loss", "Loss",       "Train Loss"),
    (axes[0, 1], "val_loss",   "Loss",       "Validation Loss"),
    (axes[1, 0], "train_ppl",  "Perplexity", "Train Perplexity"),
    (axes[1, 1], "val_ppl",    "Perplexity", "Validation Perplexity"),
]:
    sns.lineplot(data=data, x="epoch", y=col, hue="label",
                 hue_order=label_order, markers=False, ax=ax)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.legend(title="", fontsize=7, ncol=2)

plt.tight_layout()
path = os.path.join(PLOTS_DIR, "training_curves.png")
fig.savefig(path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {path}")

# ── Load test metrics ─────────────────────────────────────────────────
test_records = []
for exp_name in exp_names_sorted:
    test_path = os.path.join(OUTPUT_DIR, exp_name, "test_metrics.csv")
    if not os.path.isfile(test_path):
        continue
    df = pd.read_csv(test_path)
    df["experiment"] = exp_name
    df["label"]      = pretty(exp_name)
    test_records.append(df)

if not test_records:
    print("No test_metrics.csv files found.")
    exit(0)

test_data = pd.concat(test_records, ignore_index=True)[
    ["experiment", "label", "test_loss", "test_ppl"]
].copy()

control_ppl  = test_data.loc[test_data["experiment"] == "control", "test_ppl"].values[0]
control_loss = test_data.loc[test_data["experiment"] == "control", "test_loss"].values[0]
test_data["ppl_delta"]   = test_data["test_ppl"]  - control_ppl
test_data["loss_delta"]  = test_data["test_loss"] - control_loss
test_data["ppl_improv%"] = -100 * test_data["ppl_delta"] / control_ppl

# ── Save test results ─────────────────────────────────────────────────
test_csv = os.path.join(PLOTS_DIR, "test_results.csv")
test_data.to_csv(test_csv, index=False)
print(f"Saved test data:     {test_csv}")

# ── Print analysis ────────────────────────────────────────────────────
print("\n" + "="*65)
print("  TEST RESULTS ANALYSIS")
print("="*65)
print(f"  {'Experiment':<18} {'Loss':>7} {'PPL':>8} {'ΔPPL':>8} {'Improv%':>9}")
print("  " + "-"*53)
for _, row in test_data.sort_values("test_ppl").iterrows():
    marker = " ◀ best" if row["test_ppl"] == test_data["test_ppl"].min() else ""
    print(f"  {row['label']:<18} {row['test_loss']:>7.4f} {row['test_ppl']:>8.2f}"
          f" {row['ppl_delta']:>+8.2f} {row['ppl_improv%']:>8.1f}%{marker}")
print("="*65)

best    = test_data.loc[test_data["test_ppl"].idxmin()]
gelu2   = test_data[test_data["experiment"] != "control"]
attn    = test_data[test_data["experiment"].str.endswith("_attn")]
no_attn = test_data[test_data["experiment"].str.contains("gelu2") &
                    ~test_data["experiment"].str.endswith("_attn")]

print(f"\n  Best overall : {best['label']}  "
      f"(PPL {best['test_ppl']:.2f}, {best['ppl_improv%']:.1f}% better than control)")
print(f"  Avg GELU2    : PPL {gelu2['test_ppl'].mean():.2f}  "
      f"(Δ {gelu2['test_ppl'].mean() - control_ppl:+.2f} vs control {control_ppl:.2f})")
if len(attn) and len(no_attn):
    # Pair by base name (strip trailing _attn) so mismatched set sizes don't matter
    attn_base    = attn.copy()
    attn_base["_base"] = attn_base["experiment"].str.replace(r"_attn$", "", regex=True)
    no_attn_base = no_attn.copy()
    no_attn_base["_base"] = no_attn_base["experiment"]
    paired = attn_base.merge(no_attn_base, on="_base", suffixes=("_a", "_n"))
    attn_wins = int((paired["test_ppl_a"] < paired["test_ppl_n"]).sum())
    print(f"  +attn avg    : PPL {attn['test_ppl'].mean():.2f}")
    print(f"  no-attn avg  : PPL {no_attn['test_ppl'].mean():.2f}")
    print(f"  +attn beats no-attn: {attn_wins}/{len(paired)} paired variants")
print()

# ── Horizontal bar chart ──────────────────────────────────────────────
ordered = [pretty(e) for e in exp_names_sorted
           if e in test_data["experiment"].values]
plot_df = test_data.set_index("label").loc[ordered].reset_index()
n = len(plot_df)

blue_shades   = plt.cm.Blues(np.linspace(0.4, 0.85, 5))
orange_shades = plt.cm.Oranges(np.linspace(0.4, 0.85, 5))
colors, bi, oi = [], 0, 0
for lbl in plot_df["label"]:
    if lbl == "Control":
        colors.append("#888888")
    elif "+attn" in lbl:
        colors.append(orange_shades[oi % 5]);  oi += 1
    else:
        colors.append(blue_shades[bi % 5]);    bi += 1

fig, axes = plt.subplots(1, 2, figsize=(14, max(5, 0.45 * n + 2)))
fig.suptitle("Test Results", fontsize=13, fontweight="bold")

for ax, col, xlabel, title in [
    (axes[0], "test_loss", "Loss",       "Test Loss"),
    (axes[1], "test_ppl",  "Perplexity", "Test Perplexity"),
]:
    bars = ax.barh(plot_df["label"], plot_df[col], color=colors, edgecolor="white")
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.invert_yaxis()
    span = plot_df[col].max() - plot_df[col].min() + 1e-9
    for bar, val in zip(bars, plot_df[col]):
        ax.text(val + 0.015 * span, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", ha="left", fontsize=8)
    ctrl_val = plot_df.loc[plot_df["label"] == "Control", col].values
    if len(ctrl_val):
        ax.axvline(ctrl_val[0], color="red", linewidth=1.2, linestyle="--",
                   alpha=0.7, label="Control")
        ax.legend(fontsize=8)
    ax.set_xlim(plot_df[col].min() - 0.05 * span,
                plot_df[col].max() + 0.20 * span)

legend_handles = [
    Patch(color="#888888",       label="Control"),
    Patch(color=blue_shades[2],  label="GELU2 (no attn)"),
    Patch(color=orange_shades[2],label="GELU2 (+attn)"),
]
fig.legend(handles=legend_handles, loc="lower center", ncol=3,
           fontsize=9, bbox_to_anchor=(0.5, -0.02))
plt.tight_layout(rect=[0, 0.04, 1, 1])
path = os.path.join(PLOTS_DIR, "test_results.png")
fig.savefig(path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {path}")
