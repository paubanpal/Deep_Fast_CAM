import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
import numpy as np

fig, ax = plt.subplots(figsize=(14, 4.5), dpi=300)

def draw_3d_block(ax, x, y, w, h, d, facecolor, edgecolor="black", alpha=0.95):
    # Cara frontal
    front = Rectangle((x, y), w, h, facecolor=facecolor, edgecolor=edgecolor, lw=0.9, alpha=alpha)
    ax.add_patch(front)
    # Cara superior
    top = Polygon([[x, y+h], [x+d, y+h+d], [x+w+d, y+h+d], [x+w, y+h]], 
                  facecolor=facecolor, edgecolor=edgecolor, lw=0.8, alpha=0.85)
    ax.add_patch(top)
    # Cara lateral derecha
    side = Polygon([[x+w, y], [x+w+d, y+d], [x+w+d, y+h+d], [x+w, y+h]], 
                   facecolor=facecolor, edgecolor=edgecolor, lw=0.8, alpha=0.7)
    ax.add_patch(side)

# Paleta de colores acorde a la referencia
c_input = "#c95d63"   # Rojo desaturado
c_stem  = "#b0b5b3"   # Gris claro
c_down  = "#3b82f6"   # Azul (stride=2)
c_conv  = "#d1d5db"   # Gris intermedio
c_pool  = "#06b6d4"   # Cian
c_lstm  = "#8b5cf6"   # Púrpura
c_mlp   = "#f59e0b"   # Naranja/Ámbar
c_opt   = "#10b981"   # Verde óptico

# 0. Entrada (I)
draw_3d_block(ax, 0.5, 0.5, 0.4, 3.2, 0.35, c_input)
ax.text(0.7, 0.0, r"$I$", fontsize=15, ha='center', va='center', style='italic')

# 1. Stem A01
draw_3d_block(ax, 1.6, 0.5, 0.4, 3.2, 0.35, c_stem)
ax.text(1.8, 4.3, "0", fontsize=11, fontweight='bold', ha='center')

# 2. Stage 0 (H/2)
xs0 = [2.6, 3.1, 3.6, 4.1]
draw_3d_block(ax, xs0[0], 0.5, 0.35, 2.6, 0.3, c_down)
for x in xs0[1:]:
    draw_3d_block(ax, x, 0.5, 0.35, 2.6, 0.3, c_conv)
ax.text(3.5, 3.6, "1", fontsize=11, fontweight='bold', ha='center')

# Skip Stage 0
ax.annotate('', xy=(4.3, 0.3), xytext=(2.8, 0.3),
            arrowprops=dict(arrowstyle="->", lw=1.0, connectionstyle="arc,rad=-0.4"))
ax.text(3.55, -0.05, "+", fontsize=12, ha='center', va='center')

# 3. Stage 1 (H/4)
xs1 = [5.1, 5.6, 6.1, 6.6]
draw_3d_block(ax, xs1[0], 0.5, 0.35, 2.0, 0.25, c_down)
for x in xs1[1:]:
    draw_3d_block(ax, x, 0.5, 0.35, 2.0, 0.25, c_conv)
ax.text(6.0, 3.0, "2", fontsize=11, fontweight='bold', ha='center')

# Skip Stage 1
ax.annotate('', xy=(6.8, 0.3), xytext=(5.3, 0.3),
            arrowprops=dict(arrowstyle="->", lw=1.0, connectionstyle="arc,rad=-0.4"))
ax.text(6.05, -0.05, "+", fontsize=12, ha='center', va='center')

# 4. Stage 2 (H/8)
xs2 = [7.6, 8.1, 8.6, 9.1]
draw_3d_block(ax, xs2[0], 0.5, 0.35, 1.5, 0.2, c_down)
for x in xs2[1:]:
    draw_3d_block(ax, x, 0.5, 0.35, 1.5, 0.2, c_conv)
ax.text(8.5, 2.4, "3", fontsize=11, fontweight='bold', ha='center')

# Skip Stage 2
ax.annotate('', xy=(9.3, 0.3), xytext=(7.8, 0.3),
            arrowprops=dict(arrowstyle="->", lw=1.0, connectionstyle="arc,rad=-0.4"))
ax.text(8.55, -0.05, "+", fontsize=12, ha='center', va='center')

# 5. Bottleneck (GAP + C41) & Temporal (LSTM)
draw_3d_block(ax, 10.1, 0.8, 0.35, 0.9, 0.18, c_pool)
ax.text(10.3, 2.0, "4", fontsize=11, fontweight='bold', ha='center')

draw_3d_block(ax, 10.9, 0.7, 0.5, 1.2, 0.2, c_lstm)
ax.text(11.2, 2.2, "5", fontsize=11, fontweight='bold', ha='center')

# 6. Dense Head (C42, C43)
draw_3d_block(ax, 11.9, 0.6, 0.35, 1.5, 0.2, c_mlp)
draw_3d_block(ax, 12.4, 0.5, 0.35, 1.8, 0.22, c_mlp)
ax.text(12.4, 2.7, "6", fontsize=11, fontweight='bold', ha='center')
ax.text(12.6, 0.0, r"$\mathbf{a}_k$", fontsize=13, ha='center', va='center')

# 7. Physical Layer (Pupil, PSF, Wiener)
draw_3d_block(ax, 13.4, 0.5, 0.45, 3.2, 0.35, c_opt)
ax.text(13.6, 4.3, "7", fontsize=11, fontweight='bold', ha='center')
ax.text(13.6, 0.0, r"$\hat{O}, \mathcal{L}$", fontsize=14, ha='center', va='center')

# Skip global superior (Entrada -> Pérdida multiframe)
ax.annotate('', xy=(13.6, 4.1), xytext=(0.7, 4.1),
            arrowprops=dict(arrowstyle="->", lw=1.2, connectionstyle="arc,rad=0.18"))
ax.text(13.3, 4.5, "+", fontsize=12, fontweight='bold')

ax.set_xlim(-0.2, 14.5)
ax.set_ylim(-0.8, 5.0)
ax.axis("off")
plt.tight_layout()
plt.savefig("network_architecture.png", dpi=300, bbox_inches='tight')
plt.show()