# 3D birdcage with explicit RF "ground" (room shield) and feed depiction
# - Adds an outer cylindrical RF shield at larger radius, annotated as ground
# - Shows a coax feedthrough at Port A: inner conductor to the coil, outer conductor bonded to shield
# - Indicates a balun converting unbalanced (coax) to balanced coil port (schematic stub)
#
# Uses only matplotlib.

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# Coil parameters (illustrative)
a = 0.30      # coil radius [m]
L = 0.60      # coil length [m]
N = 16        # number of rungs
phi0 = 0.0

# Shield parameters
a_shield = 0.38  # RF room/gantry shield radius [m]
z_top = +L/2
z_bot = -L/2

fig = plt.figure(figsize=(9, 7))
ax = fig.add_subplot(111, projection='3d')

# Helper for equal aspect
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

# Coil: cylinder surface (light), end rings, rungs, capacitors, and current arrows
theta = np.linspace(0, 2*np.pi, 80)
z = np.linspace(z_bot, z_top, 40)
Theta, Z = np.meshgrid(theta, z)
X = a * np.cos(Theta)
Y = a * np.sin(Theta)
ax.plot_surface(X, Y, Z, rstride=2, cstride=10, linewidth=0.2, alpha=0.10, shade=True)

th = np.linspace(0, 2*np.pi, 400)
xring = a*np.cos(th); yring = a*np.sin(th)
ax.plot(xring, yring, np.full_like(th, z_top), linewidth=2)
ax.plot(xring, yring, np.full_like(th, z_bot), linewidth=2)

rung_thetas = np.linspace(0, 2*np.pi, N, endpoint=False)
for t in rung_thetas:
    x = a*np.cos(t); y = a*np.sin(t)
    ax.plot([x, x], [y, y], [z_bot, z_top], linewidth=1.5)

# Capacitor "plates" at z≈0 on each rung
cap_h = 0.05 * L; cap_gap = 0.02 * a
for t in rung_thetas:
    rho_hat = np.array([np.cos(t), np.sin(t), 0.0])
    center = np.array([a*np.cos(t), a*np.sin(t), 0.0])
    for sgn in (-1, +1):
        p0 = center + sgn * 0.5 * cap_gap * rho_hat + np.array([0,0,-cap_h/2])
        p1 = center + sgn * 0.5 * cap_gap * rho_hat + np.array([0,0,+cap_h/2])
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]], linewidth=2)

# Current arrows (rungs) ~ cos(theta - phi0)
Iscale = 0.18 * L
for t in rung_thetas:
    Irel = np.cos(t - phi0)
    x = a*np.cos(t); y = a*np.sin(t)
    if Irel >= 0:
        z0, z1 = 0.0 - Iscale*abs(Irel), 0.0 + Iscale*abs(Irel)
    else:
        z0, z1 = 0.0 + Iscale*abs(Irel), 0.0 - Iscale*abs(Irel)
    ax.quiver(x, y, z0, 0, 0, (z1 - z0), arrow_length_ratio=0.1, linewidth=1.4)

# End-ring tangential current arrows
nar = 28
th_ar = np.linspace(0, 2*np.pi, nar, endpoint=False)
for t in th_ar:
    phi_hat = np.array([-np.sin(t), np.cos(t), 0.0])
    A = 0.12 * a * abs(np.sin(t - phi0))
    sgn_top = 1 if np.sin(t - phi0) >= 0 else -1
    sgn_bot = -sgn_top
    pt = np.array([a*np.cos(t), a*np.sin(t), z_top])
    pb = np.array([a*np.cos(t), a*np.sin(t), z_bot])
    ax.quiver(pt[0], pt[1], pt[2], sgn_top*A*phi_hat[0], sgn_top*A*phi_hat[1], 0, arrow_length_ratio=0.2, linewidth=1.2)
    ax.quiver(pb[0], pb[1], pb[2], sgn_bot*A*phi_hat[0], sgn_bot*A*phi_hat[1], 0, arrow_length_ratio=0.2, linewidth=1.2)

# Outer RF shield cylinder (ground)
ThetaS, ZS = np.meshgrid(theta, z)
XS = a_shield * np.cos(ThetaS); YS = a_shield * np.sin(ThetaS)
ax.plot_surface(XS, YS, ZS, rstride=2, cstride=10, linewidth=0.2, alpha=0.08, shade=True)
ax.text(a_shield*0.2, 0, z_top+0.06, "RF shield (ground)", fontsize=10)

# Show explicit "ground" symbol and feed at Port A (θ=0 on top ring)
tA = 0.0
portA_coil = np.array([a*np.cos(tA), a*np.sin(tA), z_top])
portA_shield = np.array([a_shield*np.cos(tA), a_shield*np.sin(tA), z_top])

# Coax outer (ground) bonded to shield
ax.plot([portA_shield[0]], [portA_shield[1]], [portA_shield[2]], marker='o')
ax.text(portA_shield[0]+0.03, portA_shield[1], portA_shield[2]+0.02, "Coax outer → ground", fontsize=9)

# Draw a small ground symbol on the shield near port A (three lines of decreasing width)
g_w = 0.06; g_dz = 0.01
for k, frac in enumerate([1.0, 0.65, 0.35]):
    dz = (k+1)*g_dz
    # Tangential direction at θ=0 is +y
    x0 = portA_shield[0]; y0 = portA_shield[1]
    ax.plot([x0, x0], [y0 - 0.5*frac*g_w, y0 + 0.5*frac*g_w], [z_top - dz, z_top - dz], linewidth=1.5)

# Coax inner conductor from shield to coil port (schematic straight line)
ax.plot([portA_shield[0], portA_coil[0]], [portA_shield[1], portA_coil[1]], [portA_shield[2], portA_coil[2]], linewidth=2)
ax.text((portA_shield[0]+portA_coil[0])/2, (portA_shield[1]+portA_coil[1])/2, z_top+0.02, "Coax inner", fontsize=9)

# "Balun / match" schematic stub near the port (short little box)
# Represented by a small rectangle bridging two adjacent rung nodes (balanced) at top ring
t_bal = 0.0
x_bal1 = a*np.cos(t_bal); y_bal1 = a*np.sin(t_bal)
x_bal2 = a*np.cos(t_bal + 2*np.pi/N); y_bal2 = a*np.sin(t_bal + 2*np.pi/N)
zb = z_top + 0.015
# ax.plot([x_bal1, x_bal2], [y_bal1, y_bal2], [zb, zb], linewidth=3)
ax.text((x_bal1+x_bal2)/2, (y_bal1+y_bal2)/2, zb+0.02, "Port A", fontsize=9, ha='center')

# Also indicate Port B 90° away on the top ring
tB = np.pi/2
ax.scatter([a*np.cos(tB)], [a*np.sin(tB)], [z_top], s=40)
# ax.text(a*np.cos(tB)+0.03, a*np.sin(tB), z_top+0.03, "Port B (90°)", fontsize=10)

ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]'); ax.set_zlabel('z [m]')
# ax.set_title('Birdcage coil with RF shield (ground) and feed showing where "ground" lives')

set_axes_equal(ax)
ax.view_init(elev=18, azim=30)

png_path = "birdcage_ground_wrapped3d.png"
plt.tight_layout()
plt.savefig(png_path, dpi=200)
plt.show()
