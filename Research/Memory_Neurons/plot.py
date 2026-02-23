import os
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt

# ── Config ────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
PLOTS_DIR  = os.path.join(BASE_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# ── Load all experiment metrics ───────────────────────────────────────
def exp_sort_key(name):
    if name == "control":
        return (0, 0.0)
    if name.startswith("gelu2_ema"):
        try:
            return (1, float(name[9:]))
        except ValueError:
            pass
    if name.startswith("gelu2_k"):
        try:
            return (1, int(name[7:]))
        except ValueError:
            pass
    return (2, name)

records = []
for exp_name in sorted(os.listdir(OUTPUT_DIR), key=exp_sort_key):
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

sns.set_theme(style="darkgrid", palette="tab10")


def save_curve(ax, y_col, ylabel, title):
    sns.lineplot(data=data, x="epoch", y=y_col, hue="experiment", markers=True, ax=ax)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)


fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Training Curves — WikiText-2", fontsize=14, fontweight="bold")

save_curve(axes[0, 0], "train_loss", "Loss",       "Train Loss")
save_curve(axes[0, 1], "val_loss",   "Loss",       "Validation Loss")
save_curve(axes[1, 0], "train_ppl",  "Perplexity", "Train Perplexity")
save_curve(axes[1, 1], "val_ppl",    "Perplexity", "Validation Perplexity")

plt.tight_layout()
path = os.path.join(PLOTS_DIR, "training_curves.png")
fig.savefig(path, dpi=150)
plt.close(fig)
print(f"Saved: {path}")

# ── Test results summary ──────────────────────────────────────────────
test_records = []
for exp_name in sorted(os.listdir(OUTPUT_DIR), key=exp_sort_key):
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

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Test Results", fontsize=13, fontweight="bold")
    for ax, col, ylabel, title in [
        (axes[0], "test_loss", "Loss",       "Test Loss"),
        (axes[1], "test_ppl",  "Perplexity", "Test Perplexity"),
    ]:
        sns.barplot(data=test_data, x="experiment", y=col, ax=ax, palette="tab10")
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("")
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "test_results.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")
