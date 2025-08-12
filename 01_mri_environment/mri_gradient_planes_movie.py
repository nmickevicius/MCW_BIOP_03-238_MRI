#!/usr/bin/env python3
"""
MRI Gradient Coils — 3 Subplots + 360° Rotation Movie
=====================================================
- Three 3D subplots: Bz planes (x=0, y=0, z=0) per coil (X, Y, Z)
- Independent colormap scaling per subplot, no colorbars
- All coil windings drawn; the active coil in each subplot is opaque and others semi-transparent
- Saves a 20-second, 30-fps MP4 rotating the azimuth from -60° through a full 360°

Requirements:
    numpy, matplotlib, and ffmpeg on your PATH (for MP4). If ffmpeg is missing,
    the script falls back to saving an animated GIF via Pillow (if available).

Run:
    python mri_gradient_planes_movie.py
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, animation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# ------------------------------ CONFIG ------------------------------
# Cylinder / coil geometry
R_bore = 0.28          # cylinder radius [m]
z0 = 0.25              # axial offset of saddle / Maxwell centers [m]
arc_span_deg = 70      # half angular span of saddle arcs [deg]

# Discretization (balance smoothness vs speed)
n_arc = 900            # points per saddle arc
n_leg = 50             # points per short axial return leg
leg_len = 0.035        # length of return legs [m]
n_loop = 700           # points per full circle for Z (Maxwell) loops

# Coil visual separation (just for clarity; has no effect on fields)
offset_X = 0.00
offset_Y = 0.030
offset_Z = 0.060

# Colors
color_X, color_Y, color_Z = 'tab:blue', 'tab:orange', 'tab:green'
cmap_name = 'viridis'  # colormap for Bz planes

# Line widths and arrow settings
line_w = 2.0
arrow_len = 0.07
arrow_every_XY = 140
arrow_every_Z  = 100

# Transparency for non-active coils in each subplot
alpha_active = 1.0
alpha_inactive = 0.22

# Plane sampling
plane_halfwidth = 0.12
plane_N = 91
plane_alpha = 0.8

# Animation
fps = 30
duration_s = 20
frames = fps * duration_s
start_azim_deg = -60.0   # starting azimuth
elev_deg = 20.0          # fixed elevation

outfile_mp4 = "gradient_planes_rotation.mp4"
outfile_gif = "gradient_planes_rotation.gif"  # fallback if ffmpeg unavailable

# --------------------------------------------------------------------

mu0 = 4e-7*np.pi
arc_span = np.deg2rad(arc_span_deg)

# --------------------------- Geometry helpers ---------------------------
def arc_points(z_center, phi_center, s, R=R_bore, span=arc_span, n=n_arc):
    """Arc on cylinder centered at (z_center, phi_center)."""
    t = np.linspace(-span, span, n, endpoint=True)
    if s < 0:
        t = t[::-1]
    phi = phi_center + t
    x = R*np.cos(phi); y = R*np.sin(phi); z = np.full_like(phi, z_center)
    return np.stack([x,y,z], axis=1)

def axial_leg(end_point, s_dir, n=n_leg, L=leg_len):
    """Short axial segment attached at an arc end to mimic return paths."""
    x,y,z = end_point
    z2 = z + s_dir*L
    zline = np.linspace(z, z2, n, endpoint=True)
    return np.stack([np.full(n,x), np.full(n,y), zline], axis=1)

def full_circle(z_center, s=+1, R=R_bore, n=n_loop):
    """Full loop at given z. s sets current direction."""
    phi = np.linspace(0, 2*np.pi, n, endpoint=True)
    if s < 0:
        phi = phi[::-1]
    x = R*np.cos(phi); y = R*np.sin(phi); z = np.full_like(phi, z_center)
    return np.stack([x,y,z], axis=1)

def rotate_paths_around_z(paths, angle_rad):
    """Rotate polylines around z by angle_rad."""
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return [P @ Rz.T for P in paths]

def scale_radius_in_paths(paths, delta_r):
    """Push paths outward by delta_r in xy-plane (z stays the same)."""
    out = []
    for P in paths:
        xy = P[:, :2]
        r = np.linalg.norm(xy, axis=1)
        scale = (r + delta_r) / np.where(r == 0, 1, r)
        new_xy = xy * scale[:, None]
        out.append(np.column_stack([new_xy, P[:,2]]))
    return out

# ------------------------- Build & optimize X coil -------------------------
def build_X_paths(signs, R=R_bore):
    # signs: (s1,s2,s3,s4) for arcs at (+z0,0), (+z0,π), (-z0,0), (-z0,π)
    s1,s2,s3,s4 = signs
    P = []
    a = arc_points(+z0, 0.0, s=s1, R=R); P += [a, axial_leg(a[0], -1), axial_leg(a[-1], +1)]
    a = arc_points(+z0, np.pi, s=s2, R=R); P += [a, axial_leg(a[0], -1), axial_leg(a[-1], +1)]
    a = arc_points(-z0, 0.0, s=s3, R=R); P += [a, axial_leg(a[0], +1), axial_leg(a[-1], -1)]
    a = arc_points(-z0, np.pi, s=s4, R=R); P += [a, axial_leg(a[0], +1), axial_leg(a[-1], -1)]
    return P

def segments_from_paths(paths):
    segs = []
    for P in paths:
        d = np.diff(P, axis=0); P0 = P[:-1]
        for i in range(len(d)):
            segs.append((P0[i], d[i]))
    return segs

def B_from_segments(points, segments, I=100.0):
    """Biot–Savart for piecewise-linear conductors; returns B for each point."""
    B = np.zeros_like(points)
    for P0, dl in segments:
        r = points - P0
        r2 = np.einsum('ij,ij->i', r, r)
        mask = r2 > 1e-12
        if not np.any(mask):
            continue
        r_valid = r[mask]; r2_valid = r2[mask]
        dl_cross_r = np.cross(np.tile(dl, (r_valid.shape[0],1)), r_valid)
        dB = mu0*I/(4*np.pi) * (dl_cross_r.T / (r2_valid**1.5)).T
        B[mask] += dB
    return B

def find_best_X_signs():
    import itertools
    x = np.linspace(-0.12, 0.12, 121)
    pts_line = np.stack([x, np.zeros_like(x), np.zeros_like(x)], axis=1)
    best = None
    for signs in itertools.product([+1,-1], repeat=4):
        P = build_X_paths(signs)
        segs = segments_from_paths(P)
        B = B_from_segments(pts_line, segs, I=100.0)
        Bz = B[:,2]
        mask = np.abs(x) <= 0.06
        G = np.polyfit(x[mask], Bz[mask], 1)[0]
        score = abs(G)
        if best is None or score > best[0]:
            best = (score, signs, P)
    return best[1], best[2]

# Build coils
signs_best, paths_X = find_best_X_signs()
paths_Y = rotate_paths_around_z(paths_X, np.pi/2)
paths_Z = [full_circle(+z0, s=+1, R=R_bore, n=n_loop),
           full_circle(-z0, s=-1, R=R_bore, n=n_loop)]

# ----------------------------- Visualization helpers -----------------------------
def plot_paths_with_arrows(ax, paths, color, alpha=1.0, arrow_every=120, arrow_len=0.07):
    for P in paths:
        ax.plot(P[:,0], P[:,1], P[:,2], linewidth=line_w, color=color, alpha=alpha)
        idx = np.arange(0, len(P)-1, arrow_every)
        if len(idx) == 0: 
            continue
        P0 = P[idx]; P1 = P[idx+1]
        d = P1 - P0
        dn = d / (np.linalg.norm(d, axis=1, keepdims=True) + 1e-12) * arrow_len
        ax.quiver(P0[:,0], P0[:,1], P0[:,2], dn[:,0], dn[:,1], dn[:,2],
                  length=1.0, normalize=False, color=color, linewidth=1.0, alpha=alpha)

def select_segments(which='X'):
    segs_X = segments_from_paths(paths_X)
    segs_Y = segments_from_paths(paths_Y)
    segs_Z = segments_from_paths(paths_Z)
    if which == 'X':
        return segs_X
    elif which == 'Y':
        return segs_Y
    elif which == 'Z':
        return segs_Z
    else:
        return segs_X + segs_Y + segs_Z

def compute_plane_Bz(segs_for_planes, u):
    # z=0: (x,y,0)
    Xz, Yz = np.meshgrid(u, u); Zz = np.zeros_like(Xz)
    pts_z0 = np.stack([Xz.ravel(), Yz.ravel(), Zz.ravel()], axis=1)
    Bz_z0 = B_from_segments(pts_z0, segs_for_planes, I=100.0)[:,2].reshape(len(u), len(u))

    # y=0: (x,0,z)
    Xy0, Zy0 = np.meshgrid(u, u); Yy0 = np.zeros_like(Xy0)
    pts_y0 = np.stack([Xy0.ravel(), Yy0.ravel(), Zy0.ravel()], axis=1)
    Bz_y0 = B_from_segments(pts_y0, segs_for_planes, I=100.0)[:,2].reshape(len(u), len(u))

    # x=0: (0,y,z)
    Yx0, Zx0 = np.meshgrid(u, u); Xx0 = np.zeros_like(Yx0)
    pts_x0 = np.stack([Xx0.ravel(), Yx0.ravel(), Zx0.ravel()], axis=1)
    Bz_x0 = B_from_segments(pts_x0, segs_for_planes, I=100.0)[:,2].reshape(len(u), len(u))

    return (Xz, Yz, Zz, Bz_z0), (Xy0, Yy0, Zy0, Bz_y0), (Xx0, Yx0, Zx0, Bz_x0)

def draw_bore(ax):
    phi = np.linspace(0, 2*np.pi, 140)
    z_cyl = np.linspace(-(abs(z0)+0.35), (abs(z0)+0.35), 160)
    Phi, Zc = np.meshgrid(phi, z_cyl)
    Xc = R_bore*np.cos(Phi); Yc = R_bore*np.sin(Phi)
    ax.plot_wireframe(Xc, Yc, Zc, rcount=30, ccount=30, linewidth=0.3, alpha=0.2)

def draw_planes(ax, Bz_x0, Bz_y0, Bz_z0, u, cmap, norm):
    # Surfaces for three planes
    surf_kwargs = dict(rstride=1, cstride=1, shade=False, antialiased=False, alpha=plane_alpha)
    # z=0
    Xz, Yz = np.meshgrid(u, u); Zz = np.zeros_like(Xz)
    ax.plot_surface(Xz, Yz, Zz, facecolors=cmap(norm(Bz_z0)), **surf_kwargs)
    # y=0
    Xy0, Zy0 = np.meshgrid(u, u); Yy0 = np.zeros_like(Xy0)
    ax.plot_surface(Xy0, Yy0, Zy0, facecolors=cmap(norm(Bz_y0)), **surf_kwargs)
    # x=0
    Yx0, Zx0 = np.meshgrid(u, u); Xx0 = np.zeros_like(Yx0)
    ax.plot_surface(Xx0, Yx0, Zx0, facecolors=cmap(norm(Bz_x0)), **surf_kwargs)

# ----------------------------- Build figure once -----------------------------
def build_figure():
    # Precompute offset/visual paths
    X_off = scale_radius_in_paths(paths_X, offset_X)
    Y_off = scale_radius_in_paths(paths_Y, offset_Y)
    Z_off = scale_radius_in_paths(paths_Z, offset_Z)

    # Prepare grids
    u = np.linspace(-plane_halfwidth, plane_halfwidth, plane_N)
    cmap = cm.get_cmap(cmap_name)

    # Figure with 3 subplots
    fig = plt.figure(figsize=(16,6))
    axs = [fig.add_subplot(1,3,i+1, projection='3d') for i in range(3)]
    titles = ['Bz from X coil', 'Bz from Y coil', 'Bz from Z coil']
    which_list = ['X','Y','Z']

    # For each subplot, compute its own normalization (independent color scale)
    for ax, title, which in zip(axs, titles, which_list):
        draw_bore(ax)
        segs = select_segments(which)
        (Xz, Yz, Zz, Bz_z0), (Xy0, Yy0, Zy0, Bz_y0), (Xx0, Yx0, Zx0, Bz_x0) = compute_plane_Bz(segs, u)

        # independent normalization for this axis
        m = np.max(np.abs([Bz_z0, Bz_y0, Bz_x0]))
        norm = plt.Normalize(vmin=-m, vmax=+m)

        draw_planes(ax, Bz_x0, Bz_y0, Bz_z0, u, cmap, norm)

        # coils with alpha highlighting
        if which == 'X':
            plot_paths_with_arrows(ax, X_off, color_X, alpha_active,  arrow_every_XY, arrow_len)
            plot_paths_with_arrows(ax, Y_off, color_Y, alpha_inactive, arrow_every_XY, arrow_len)
            plot_paths_with_arrows(ax, Z_off, color_Z, alpha_inactive, arrow_every_Z,  arrow_len)
        elif which == 'Y':
            plot_paths_with_arrows(ax, X_off, color_X, alpha_inactive, arrow_every_XY, arrow_len)
            plot_paths_with_arrows(ax, Y_off, color_Y, alpha_active,  arrow_every_XY, arrow_len)
            plot_paths_with_arrows(ax, Z_off, color_Z, alpha_inactive, arrow_every_Z,  arrow_len)
        else: # 'Z'
            plot_paths_with_arrows(ax, X_off, color_X, alpha_inactive, arrow_every_XY, arrow_len)
            plot_paths_with_arrows(ax, Y_off, color_Y, alpha_inactive, arrow_every_XY, arrow_len)
            plot_paths_with_arrows(ax, Z_off, color_Z, alpha_active,  arrow_every_Z,  arrow_len)

        ax.set_title(title)
        ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)'); ax.set_zlabel('z (m)')
        ax.set_zlim(-0.3, 0.3)
        ax.set_box_aspect([1,1,1])
        ax.view_init(elev=elev_deg, azim=start_azim_deg)

    plt.tight_layout()
    return fig, axs

# ----------------------------- Animation -----------------------------
def main():
    fig, axs = build_figure()

    def update(frame):
        print(frame)
        # Rotate azimuth uniformly across all 3 axes
        az = start_azim_deg + 360.0 * frame / (frames - 1)
        for ax in axs:
            ax.view_init(elev=elev_deg, azim=az)
        return []  # no blitting

    print(frames) 
    
    anim = animation.FuncAnimation(fig, update, frames=frames, interval=1000/fps, blit=False)

    # Try MP4 via ffmpeg; fall back to GIF if unavailable
    try:
        Writer = animation.FFMpegWriter
        writer = Writer(fps=fps, bitrate=2400)
        anim.save(outfile_mp4, writer=writer, dpi=120)
        print(f"Saved MP4: {outfile_mp4}")
    except Exception as e:
        print("FFmpeg not available or failed:", e)
        try:
            from matplotlib.animation import PillowWriter
            anim.save(outfile_gif, writer=PillowWriter(fps=fps), dpi=120)
            print(f"Saved GIF: {outfile_gif}")
        except Exception as e2:
            print("GIF fallback failed as well:", e2)
            print("Please install ffmpeg (or Pillow) and try again.")

    plt.show()

if __name__ == '__main__':
    print("Best X winding signs (arcs at +z0,0; +z0,pi; -z0,0; -z0,pi):", find_best_X_signs()[0])
    main()
