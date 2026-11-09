import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Load results.csv
csv_path = Path("G:/My Drive/ALDR_Part2/modification_unified_11class-9/results.csv")
if not csv_path.exists():
    csv_path = Path("G:/My Drive/ALDR_Part2/modification_unified_11class-2/results.csv")

df = pd.read_csv(csv_path)
df.columns = [c.strip() for c in df.columns]

# Create figure with 2 subplots
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=150)
fig.suptitle("Training curves: loss and validation metrics", fontsize=18, fontweight='bold', color='#16325c', y=0.98)

epochs = df['epoch']
ticks_x = [e for e in epochs if e % 4 == 1 or e == 1]

# -------------------------------------------------------------
# Left Subplot: Training loss (11-class model)
# -------------------------------------------------------------
ax1.set_title("Training loss (11-class model)", fontsize=13, pad=12, color='#2c3e50')
ax1.plot(epochs, df['train/cls_loss'], label='Classification Loss (cls_loss)', color='#d9822b', linewidth=2.2)
ax1.plot(epochs, df['train/box_loss'], label='Bounding Box Loss (box_loss)', color='#1f3d5c', linewidth=2.2)
ax1.plot(epochs, df['train/dfl_loss'], label='Distribution Focal Loss (dfl_loss)', color='#8c9ba5', linewidth=2.2)

ax1.set_xlim(1, 80)
ax1.set_ylim(0, 7)
ax1.set_xticks(ticks_x)
ax1.set_xticklabels(ticks_x, rotation=45, fontsize=9)
ax1.set_yticks(range(0, 8))
ax1.set_xlabel("Epoch", fontsize=11, labelpad=8)
ax1.grid(True, linestyle='-', alpha=0.3, color='#cccccc', axis='y')
ax1.grid(False, axis='x')
ax1.legend(loc='upper right', frameon=True, facecolor='white', edgecolor='#e2e8f0', fontsize=9)

# -------------------------------------------------------------
# Right Subplot: Validation metrics per epoch
# -------------------------------------------------------------
ax2.set_title("Validation metrics per epoch", fontsize=13, pad=12, color='#2c3e50')
ax2.plot(epochs, df['metrics/precision(B)'], label='Precision', color='#1f3d5c', linewidth=2.0)
ax2.plot(epochs, df['metrics/recall(B)'], label='Recall', color='#d9822b', linewidth=2.0)
ax2.plot(epochs, df['metrics/mAP50(B)'], label='mAP@0.5', color='#2b7a4b', linewidth=2.0)
ax2.plot(epochs, df['metrics/mAP50-95(B)'], label='mAP@0.5:0.95', color='#8c9ba5', linewidth=2.0)

ax2.set_xlim(1, 80)
ax2.set_ylim(0, 1.0)
ax2.set_xticks(ticks_x)
ax2.set_xticklabels(ticks_x, rotation=45, fontsize=9)
ax2.set_yticks([i / 10.0 for i in range(11)])
ax2.set_xlabel("Epoch", fontsize=11, labelpad=8)
ax2.grid(True, linestyle='-', alpha=0.3, color='#cccccc', axis='y')
ax2.grid(False, axis='x')
ax2.legend(loc='lower center', bbox_to_anchor=(0.5, -0.28), ncol=4, frameon=False, fontsize=10)

plt.tight_layout()

out_plot_path = Path("c:/Users/Suyash/Saved Games/Math.ai/aldr_output/training_curves_11class.png")
out_plot_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(str(out_plot_path), bbox_inches='tight', dpi=200)
print(f"Plot saved successfully to: {out_plot_path}")
