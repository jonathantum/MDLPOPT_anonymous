import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# --- Data ---
classes = ['C0\nLow-Stable', 'C1\nImproving', 'C2\nWorsening', 'C3\nHigh-Persistent']
counts  = [5696, 442, 1199, 165]
total   = sum(counts)
original  = [c / total * 100 for c in counts]
sampled   = [50.0, 15.0, 20.0, 15.0]

x     = np.arange(len(classes))
width = 0.35

# --- Figure ---
fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor('white')
ax.set_facecolor('#F8F8F8')

COLOR_ORIG    = '#2E75B6'
COLOR_SAMPLED = '#C55A11'

bars1 = ax.bar(x - width/2, original, width,
               label='Original Distribution',
               color=COLOR_ORIG, edgecolor='white',
               linewidth=0.8, zorder=3)
bars2 = ax.bar(x + width/2, sampled, width,
               label='After Weighted Sampling',
               color=COLOR_SAMPLED, edgecolor='white',
               linewidth=0.8, zorder=3)

# --- Value labels: percentage on top + absolute count inside bar ---
for bar, count, pct in zip(bars1, counts, original):
    h = bar.get_height()
    # percentage + absolute above bar
    ax.annotate(f'{pct:.0f}%\n(N={count:,})',
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 5),
                textcoords='offset points',
                ha='center', va='bottom',
                fontsize=9, color='#1a1a1a',
                fontweight='bold',
                linespacing=1.5)

for bar, pct in zip(bars2, sampled):
    h = bar.get_height()
    ax.annotate(f'{pct:.0f}%',
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 5),
                textcoords='offset points',
                ha='center', va='bottom',
                fontsize=9, color='#1a1a1a',
                fontweight='bold')

# --- Reference line at 25% (random baseline for 4 classes) ---
ax.axhline(y=25, color='grey', linestyle=':', linewidth=1.2,
           zorder=2, label='Random Baseline (25%)')

# --- Axes ---
ax.set_xlabel('Trajectory Phenotype', fontsize=10, labelpad=10)#, fontweight='bold')
ax.set_ylabel('Proportion of Training Samples (%)', fontsize=10, labelpad=10)#, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(classes, fontsize=11)
ax.set_ylim(0, 92)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda val, _: f'{val:.0f}%'))
ax.tick_params(axis='both', labelsize=8, length=0)

# --- Grid ---
ax.yaxis.grid(True, linestyle='--', alpha=0.4, color='grey', zorder=0)
ax.xaxis.grid(False)
ax.set_axisbelow(True)

# --- Spines ---
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('#cccccc')
ax.spines['bottom'].set_color('#cccccc')

# --- Subtle background bands per class group ---
for i in range(len(classes)):
    if i % 2 == 0:
        ax.axvspan(i - 0.5, i + 0.5,
                   color='#eeeeee', alpha=0.5, zorder=1)

# --- Legend ---
ax.legend(fontsize=10, frameon=True,
          framealpha=0.9, edgecolor='#cccccc',
          loc='upper right', borderpad=0.8)

# --- Title ---
ax.set_title('Class Distribution: Original vs. Weighted Sampling',
             fontsize=13, pad=14, color='#1a1a1a')

plt.tight_layout(pad=1.5)

# --- Save ---
plt.savefig('/home/anonymous/MDLPOPT/data/visuals/class_distribution.pdf', dpi=300, bbox_inches='tight')
plt.savefig('/home/anonymous/MDLPOPT/data/visuals/class_distribution.png', dpi=300, bbox_inches='tight')

plt.show()