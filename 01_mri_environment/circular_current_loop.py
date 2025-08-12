# Re-run with mathtext-friendly labels (avoid \boldsymbol)
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# Parameters
a = 1.0          # loop radius
z0 = 1.2         # observation point z
phi0 = np.deg2rad(40)  # source point angle on the loop
dl_scale = 0.4   # length scaling for dℓ'
R_scale = 1.0    # scaling for R arrow for visibility

# Geometry
phi = np.linspace(0, 2*np.pi, 400)
x_loop = a * np.cos(phi)
y_loop = a * np.sin(phi)
z_loop = np.zeros_like(phi)

rprime = np.array([a*np.cos(phi0), a*np.sin(phi0), 0.0])
phihat = np.array([-np.sin(phi0), np.cos(phi0), 0.0])
dl_vec = dl_scale * phihat

robsv = np.array([0.0, 0.0, z0])
Rvec = robsv - rprime

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

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# Loop and z-axis
ax.plot(x_loop, y_loop, z_loop, linewidth=2, label='Current loop')
ax.plot([0, 0], [0, 0], [-1.5*a, 1.8*a], linestyle='--', linewidth=1)

# Points and labels
ax.scatter(*rprime, s=40)
ax.text(*(rprime + np.array([0.1, 0.1, 0.05])), r"$\mathbf{r}'$", fontsize=11)
ax.scatter(*robsv, s=40)
ax.text(*(robsv + np.array([0.1, 0.1, 0.05])), r"$\mathbf{r}$", fontsize=11)

# Vectors
ax.quiver(0, 0, 0, rprime[0], rprime[1], rprime[2], arrow_length_ratio=0.08, linewidth=2, label=r"$\mathbf{r}'$")
ax.quiver(0, 0, 0, robsv[0], robsv[1], robsv[2], arrow_length_ratio=0.08, linewidth=2, label=r"$\mathbf{r}$")
ax.quiver(rprime[0], rprime[1], rprime[2], Rvec[0], Rvec[1], Rvec[2], arrow_length_ratio=0.08, linewidth=2, label=r"$\mathbf{R}=\mathbf{r}-\mathbf{r}'$")
ax.quiver(rprime[0], rprime[1], rprime[2], dl_vec[0], dl_vec[1], dl_vec[2], arrow_length_ratio=0.3, linewidth=2, label=r"$d\vec{\ell}'$")

# Labels, view, limits
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
# ax.set_title('Biot–Savart setup: field on the axis of a circular current loop')
ax.view_init(elev=22, azim=40)

lim = 1.9*a
ax.set_xlim(-lim, lim)
ax.set_ylim(-lim, lim)
ax.set_zlim(-lim, lim)
set_axes_equal(ax)

ax.legend(loc='upper left', fontsize=10)

# png_path = '/mnt/data/mri_loop_axis_setup.png'
# pdf_path = '/mnt/data/mri_loop_axis_setup.pdf'
plt.tight_layout()
# plt.savefig(png_path, dpi=200)
# plt.savefig(pdf_path)
plt.show()

