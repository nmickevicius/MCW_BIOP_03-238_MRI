# 3D plot: Birdcage + Gradients + Outer Solenoid
# Same as previous figure, but now enforce CONSISTENT COLORS per gradient axis:
#   X gradient (Golay saddles) -> red
#   Y gradient (Golay saddles) -> green
#   Z gradient (anti-Helmholtz loops) -> blue

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# --- Base coil parameters ---
a = 0.30      # birdcage radius [m]
L = 0.60      # length [m]
N = 16        # rungs
z_top = +L/2
z_bot = -L/2

# --- Gradient coil radii ---
r_z = 0.33    # Z gradient
r_y = 0.35    # Y (Golay)
r_x = 0.37    # X (Golay)

# --- Solenoid parameters (outer, encompassing others) ---
r_sol = 0.43          # solenoid radius [m]
n_turns = 32          # tightly wound turns along length
n_helices = 4         # draw multiple start positions to suggest a winding pack
alpha_sol = 0.05      # transparency for solenoid lines

# --- Colors per axis ---
col_x = 'tab:red'
col_y = 'tab:green'
col_z = 'tab:blue'
col_bird = '0.3'      # gray for birdcage
col_sol = 'k' # solenoid color

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

# --- Birdcage context (neutral gray) ---
theta = np.linspace(0, 2*np.pi, 200)
xring = a*np.cos(theta); yring = a*np.sin(theta)
ax.plot(xring, yring, np.full_like(theta, z_top), linewidth=1.5, alpha=0.8, color=col_bird)
ax.plot(xring, yring, np.full_like(theta, z_bot), linewidth=1.5, alpha=0.8, color=col_bird)

rung_thetas = np.linspace(0, 2*np.pi, N, endpoint=False)
for t in rung_thetas:
    x = a*np.cos(t); y = a*np.sin(t)
    ax.plot([x, x], [y, y], [z_bot, z_top], linewidth=1.2, alpha=0.8, color=col_bird)

# --- Z gradient (anti-Helmholtz) in BLUE ---
dz = 0.30
z1 = +dz/2; z2 = -dz/2
theta_fine = np.linspace(0, 2*np.pi, 600)
xz1 = r_z*np.cos(theta_fine); yz1 = r_z*np.sin(theta_fine)
xz2 = r_z*np.cos(theta_fine); yz2 = r_z*np.sin(theta_fine)
ax.plot(xz1, yz1, np.full_like(theta_fine, z1), linewidth=2, color=col_z)
ax.plot(xz2, yz2, np.full_like(theta_fine, z2), linewidth=2, color=col_z)
# Direction arrows along loops (also blue)
for t in np.linspace(0, 2*np.pi, 16, endpoint=False):
    phi_hat = np.array([-np.sin(t), np.cos(t), 0.0])
    A = 0.06*r_z
    x0, y0 = r_z*np.cos(t), r_z*np.sin(t)
    ax.quiver(x0, y0, z1, A*phi_hat[0], A*phi_hat[1], 0, arrow_length_ratio=0.35, linewidth=1.1, color=col_z)
    ax.quiver(x0, y0, z2, -A*phi_hat[0], -A*phi_hat[1], 0, arrow_length_ratio=0.35, linewidth=1.1, color=col_z)
ax.text(r_z*0.05, 0, z1+0.05, "Z gradient", fontsize=10, color=col_z)

# --- Golay saddle helper ---
def draw_saddle(ax, r, z_len, th1, th2, color='k', linewidth=2):
    zA, zB = -z_len/2, +z_len/2
    z_leg = np.linspace(zA, zB, 200)
    x1, y1 = r*np.cos(th1), r*np.sin(th1)
    x2, y2 = r*np.cos(th2), r*np.sin(th2)
    ax.plot([x1]*200, [y1]*200, z_leg, linewidth=linewidth, color=color)
    ax.plot([x2]*200, [y2]*200, z_leg, linewidth=linewidth, color=color)
    th_arc1 = np.linspace(th1, th2, 200)
    th_arc2 = np.linspace(th2, th1, 200)
    ax.plot(r*np.cos(th_arc1), r*np.sin(th_arc1), np.full(200, zB), linewidth=linewidth, color=color)
    ax.plot(r*np.cos(th_arc2), r*np.sin(th_arc2), np.full(200, zA), linewidth=linewidth, color=color)

# --- Y gradient (Golay) in GREEN ---
z_len_g = 0.42
dth = np.deg2rad(35)
draw_saddle(ax, r_y, z_len_g, -dth, +dth, color=col_y, linewidth=2.2)
draw_saddle(ax, r_y, z_len_g, np.pi - dth, np.pi + dth, color=col_y, linewidth=2.2)
ax.text(r_y*0.7, 0, -0.26, "Y gradient", fontsize=10, color=col_y)

# --- X gradient (Golay) in RED ---
draw_saddle(ax, r_x, z_len_g, np.pi/2 - dth, np.pi/2 + dth, color=col_x, linewidth=2.2)
draw_saddle(ax, r_x, z_len_g, 3*np.pi/2 - dth, 3*np.pi/2 + dth, color=col_x, linewidth=2.2)
ax.text(0, r_x*0.7, 0.26, "X gradient", fontsize=10, color=col_x, rotation=0)

# --- Outer solenoid (nearly transparent) ---
t = np.linspace(0, 2*np.pi*n_turns, 4000)
z_sol = np.linspace(z_bot*2, z_top*2, t.size)
for k in range(n_helices):
    t_shift = t + 2*np.pi * k / n_helices
    xs = r_sol * np.cos(t_shift)
    ys = r_sol * np.sin(t_shift)
    ax.plot(xs, ys, z_sol, linewidth=1.0, alpha=alpha_sol, color=col_sol)

# Labels and view
ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]'); ax.set_zlabel('z [m]')
# ax.set_title('Birdcage + Gradients + Solenoid\nConsistent colors per gradient axis (X=red, Y=green, Z=blue)')

set_axes_equal(ax)
ax.view_init(elev=18, azim=35)

png_path = "scanner.png"
plt.tight_layout()
plt.savefig(png_path, dpi=200)
plt.show()

