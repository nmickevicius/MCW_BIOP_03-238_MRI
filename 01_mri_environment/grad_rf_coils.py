# 3D plot: Birdcage RF coil with superimposed gradient coils
# - Z gradient: anti-Helmholtz pair (two loops with opposite current) at radius r_z
# - Y gradient: Golay (saddle) pair at radius r_y
# - X gradient: Golay (saddle) pair at radius r_x
# - Offsets in radius so each layer is visible
#
# NOTE: This is a didactic geometry illustration (not a manufacturable winding layout).

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# --- Base coil and shield parameters (as before) ---
a = 0.30      # birdcage radius [m]
L = 0.60      # birdcage length [m]
N = 16        # rungs
z_top = +L/2
z_bot = -L/2

# --- Gradient coil radii (offset outward from the birdcage, but inside the shield) ---
r_z = 0.33    # Z gradient radius [m]
r_y = 0.35    # Y (Golay) radius [m]
r_x = 0.37    # X (Golay) radius [m]
r_shield = 0.40  # outer context cylinder

# --- Helper for equal aspect ---
def set_axes_equal(ax):
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()
    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)
    plot_radius = 0.5 * max([x_range, y_range, z_range])
    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

fig = plt.figure(figsize=(10, 7.5))
ax = fig.add_subplot(111, projection='3d')

# --- Draw birdcage minimal context (rungs + rings, lightly) ---
theta = np.linspace(0, 2*np.pi, 200)
xring = a*np.cos(theta); yring = a*np.sin(theta)
ax.plot(xring, yring, np.full_like(theta, z_top), linewidth=1.5, alpha=0.8)
ax.plot(xring, yring, np.full_like(theta, z_bot), linewidth=1.5, alpha=0.8)

rung_thetas = np.linspace(0, 2*np.pi, N, endpoint=False)
for t in rung_thetas:
    x = a*np.cos(t); y = a*np.sin(t)
    ax.plot([x, x], [y, y], [z_bot, z_top], linewidth=1.2, alpha=0.8)

# --- Z gradient (anti-Helmholtz): two loops at z=±dz/2 with opposite current ---
dz = 0.30  # axial separation between the centers of the two loops [m]
z1 = +dz/2; z2 = -dz/2
theta_fine = np.linspace(0, 2*np.pi, 600)
xz1 = r_z*np.cos(theta_fine); yz1 = r_z*np.sin(theta_fine)
xz2 = r_z*np.cos(theta_fine); yz2 = r_z*np.sin(theta_fine)
ax.plot(xz1, yz1, np.full_like(theta_fine, z1), linewidth=2)
ax.plot(xz2, yz2, np.full_like(theta_fine, z2), linewidth=2)
# Indicate opposite current directions with small tangential arrows
for t in np.linspace(0, 2*np.pi, 16, endpoint=False):
    # top loop current +phi
    x0, y0 = r_z*np.cos(t), r_z*np.sin(t)
    phi_hat = np.array([-np.sin(t), np.cos(t), 0.0])
    A = 0.06*r_z
    ax.quiver(x0, y0, z1, A*phi_hat[0], A*phi_hat[1], 0, arrow_length_ratio=0.35, linewidth=1.2)
    # bottom loop current -phi
    ax.quiver(x0, y0, z2, -A*phi_hat[0], -A*phi_hat[1], 0, arrow_length_ratio=0.35, linewidth=1.2)

ax.text(r_z*0.05, 0, z1+0.05, "Z gradient (anti-Helmholtz)", fontsize=10)

# --- Golay (saddle) helper: draw a single saddle on radius r ---
def draw_saddle(ax, r, z_len, th1, th2, current_sign=+1, npts=200, linewidth=2):
    """Draw one saddle: two longitudinal legs at th1, th2 from -z_len/2..+z_len/2
       and two end arcs at z=±z_len/2 joining th1↔th2. current_sign determines arrow directions.
    """
    zA, zB = -z_len/2, +z_len/2
    # Legs
    z_leg = np.linspace(zA, zB, npts)
    x1, y1 = r*np.cos(th1), r*np.sin(th1)
    x2, y2 = r*np.cos(th2), r*np.sin(th2)
    ax.plot([x1]*npts, [y1]*npts, z_leg, linewidth=linewidth)
    ax.plot([x2]*npts, [y2]*npts, z_leg, linewidth=linewidth)
    # End arcs (shortest arc between th1 and th2)
    th_arc1 = np.linspace(th1, th2, npts)
    th_arc2 = np.linspace(th2, th1, npts)
    ax.plot(r*np.cos(th_arc1), r*np.sin(th_arc1), np.full(npts, zB), linewidth=linewidth)
    ax.plot(r*np.cos(th_arc2), r*np.sin(th_arc2), np.full(npts, zA), linewidth=linewidth)
    # Arrows for current direction (legs)
    # Up along th1, down along th2 (or vice versa) depends on current_sign
    A = 0.10*z_len
    if current_sign > 0:
        ax.quiver(x1, y1, 0, 0, 0, +A, arrow_length_ratio=0.15, linewidth=1.2)
        ax.quiver(x2, y2, 0, 0, 0, -A, arrow_length_ratio=0.15, linewidth=1.2)
    else:
        ax.quiver(x1, y1, 0, 0, 0, -A, arrow_length_ratio=0.15, linewidth=1.2)
        ax.quiver(x2, y2, 0, 0, 0, +A, arrow_length_ratio=0.15, linewidth=1.2)

# --- Y gradient (Golay): saddles centered around ±x (θ≈0 and π) ---
z_len_g = 0.42          # axial extent of each saddle
dth = np.deg2rad(35)    # angular half-width (separation between legs ≈ 2*dth)
# Front saddle near +x axis
draw_saddle(ax, r_y, z_len_g, -dth, +dth, current_sign=+1, linewidth=2)
# Back saddle (opposite side) with opposite current, centered near -x (θ≈π)
draw_saddle(ax, r_y, z_len_g, np.pi - dth, np.pi + dth, current_sign=-1, linewidth=2)
ax.text(r_y*0.7, 0, -0.26, "Y gradient (Golay saddles)", fontsize=10)

# --- X gradient (Golay): saddles centered around ±y (θ≈π/2 and 3π/2) ---
# Left/right saddles near y-axis
draw_saddle(ax, r_x, z_len_g, np.pi/2 - dth, np.pi/2 + dth, current_sign=+1, linewidth=2)
draw_saddle(ax, r_x, z_len_g, 3*np.pi/2 - dth, 3*np.pi/2 + dth, current_sign=-1, linewidth=2)
ax.text(0, r_x*0.7, 0.26, "X gradient (Golay saddles)", fontsize=10, rotation=0)

# --- Context: faint outer shield cylinder ---
theta_s = np.linspace(0, 2*np.pi, 80)
z_s = np.linspace(z_bot, z_top, 40)
ThetaS, ZS = np.meshgrid(theta_s, z_s)
XS = r_shield * np.cos(ThetaS); YS = r_shield * np.sin(ThetaS)
ax.plot_surface(XS, YS, ZS, rstride=2, cstride=10, linewidth=0.2, alpha=0.05, shade=True)

# Labels and view
ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]'); ax.set_zlabel('z [m]')
ax.set_title('Birdcage with superimposed gradient coils:\nZ (anti-Helmholtz) + Y/X (Golay saddles), each at a different radius')

set_axes_equal(ax)
ax.view_init(elev=18, azim=35)

png_path = "birdcage_with_gradients.png"
plt.tight_layout()
plt.savefig(png_path, dpi=200)
plt.show()

