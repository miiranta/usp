import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ── Config ────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
PLOTS_DIR  = os.path.join(BASE_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# ── Load all experiment metrics ───────────────────────────────────────
records = []
for exp_name in sorted(os.listdir(OUTPUT_DIR)):
    metrics_path = os.path.join(OUTPUT_DIR, exp_name, "metrics.csv")
    if not os.path.isfile(metrics_path):
        continue
    df = pd.read_csv(metrics_path)
    df["experiment"] = exp_name
    records.append(df)

if not records:
    print("No metrics.csv files found in output/. Run train.py first.")
    exit(1)

data = pd.concat(records, ignore_index=True)

# Melt to long format for seaborn
loss_df = data.melt(
    id_vars=["epoch", "experiment"],
    value_vars=["train_loss", "val_loss"],
    var_name="split", value_name="loss"
)
ppl_df = data.melt(
    id_vars=["epoch", "experiment"],
    value_vars=["train_ppl", "val_ppl"],
    var_name="split", value_name="ppl"
)

# Clean up split labels
loss_df["split"] = loss_df["split"].str.replace("_loss", "")
ppl_df["split"]  = ppl_df["split"].str.replace("_ppl", "")

# ── Plot ──────────────────────────────────────────────────────────────
sns.set_theme(style="darkgrid", palette="tab10")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Training Curves — WikiText-2", fontsize=14, fontweight="bold")

# Loss
sns.lineplot(
    data=loss_df, x="epoch", y="loss",
    hue="experiment", style="split",
    markers=True, dashes={"train": (4, 2), "val": (1, 0)},
    ax=axes[0]
)
axes[0].set_title("Cross-Entropy Loss")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")

# Perplexity
sns.lineplot(
    data=ppl_df, x="epoch", y="ppl",
    hue="experiment", style="split",
    markers=True, dashes={"train": (4, 2), "val": (1, 0)},
    ax=axes[1]
)
axes[1].set_title("Perplexity")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("PPL")

plt.tight_layout()
out_path = os.path.join(PLOTS_DIR, "training_curves.png")
plt.savefig(out_path, dpi=150)
print(f"Saved → {out_path}")
plt.show()

# ── Test results summary ──────────────────────────────────────────────
test_records = []
for exp_name in sorted(os.listdir(OUTPUT_DIR)):
    test_path = os.path.join(OUTPUT_DIR, exp_name, "test_metrics.csv")
    if not os.path.isfile(test_path):
        continue
    df = pd.read_csv(test_path)
    df["experiment"] = exp_name
    test_records.append(df)

if test_records:
    test_data = pd.concat(test_records, ignore_index=True)[["experiment", "test_loss", "test_ppl"]]
    print("\nTest results:")
    print(test_data.to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle("Test Results", fontsize=13, fontweight="bold")

    sns.barplot(data=test_data, x="experiment", y="test_loss", ax=axes[0], palette="tab10")
    axes[0].set_title("Test Loss")
    axes[0].set_ylabel("Loss")

    sns.barplot(data=test_data, x="experiment", y="test_ppl", ax=axes[1], palette="tab10")
    axes[1].set_title("Test Perplexity")
    axes[1].set_ylabel("PPL")

    plt.tight_layout()
    out_path = os.path.join(PLOTS_DIR, "test_results.png")
    plt.savefig(out_path, dpi=150)
    print(f"Saved → {out_path}")
    plt.show()
