#!/usr/bin/env python3
"""
Anti-Helmholtz coil pair (two circular loops with opposite current):
- 3D geometry plot
- On-axis Bz(z) plot with linear approximation at z=0
- Current per loop needed to achieve a target gradient G at z=0

Usage (example):
  python3 gz_coil.py --a 0.30 --d 0.30 --N 10 --G 0.030 --outdir .

Dependencies: numpy, matplotlib
  pip install --upgrade pip
  pip install numpy matplotlib
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import os
import csv

mu0 = 4*np.pi*1e-7  # H/m

def Bz_loop_on_axis(z, a, z0, I, N=1):
    """On-axis Bz for a single circular loop (N turns) centered at z0."""
    return mu0 * N * I * a**2 / (2.0 * (a**2 + (z - z0)**2)**1.5)

def Bz_antihelmholtz(z, a, d, I, N=1):
    """On-axis Bz for two equal loops at z=±d/2 with opposite currents ±I."""
    z0 = d/2.0
    return Bz_loop_on_axis(z, a, +z0, +I, N) - Bz_loop_on_axis(z, a, -z0, +I, N)

def gradient_at_center(a, d, I, N=1):
    """Analytic dBz/dz at z=0 for anti-Helmholtz pair."""
    denom = (a**2 + (d/2.0)**2)**2.5
    return (3*mu0*N*I*a**2*d) / (2*denom)

def current_for_gradient(G, a, d, N=1):
    """Solve for I to achieve target gradient G = dBz/dz at z=0."""
    denom = (a**2 + (d/2.0)**2)**2.5
    return (2*G*denom) / (3*mu0*N*a**2*d)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--a", type=float, default=0.30, help="Loop radius [m]")
    p.add_argument("--d", type=float, default=0.30, help="Axial separation between loop centers [m]")
    p.add_argument("--N", type=int,   default=10,   help="Turns per loop")
    p.add_argument("--G", type=float, default=0.030,help="Target gradient at center [T/m] (e.g., 0.030=30 mT/m)")
    p.add_argument("--zmin", type=float, default=-0.20, help="Plot range min z [m]")
    p.add_argument("--zmax", type=float, default=+0.20, help="Plot range max z [m]")
    p.add_argument("--nz",   type=int,   default=801,   help="Number of z samples")
    p.add_argument("--outdir", type=str, default=".",   help="Output directory for figures/CSV")
    p.add_argument("--csv", action="store_true", help="Also write CSV of z, Bz, and linear approx")
    args = p.parse_args()

    a, d, N, G = args.a, args.d, args.N, args.G

    # Compute required current
    I_needed = current_for_gradient(G, a, d, N=N)

    print("\nAnti-Helmholtz parameters")
    print("-------------------------")
    print(f"a (radius)      : {a:.4f} m")
    print(f"d (separation)  : {d:.4f} m")
    print(f"N (turns/loop)  : {N:d}")
    print(f"G target        : {G:.6f} T/m  ({G*1e3:.1f} mT/m)")
    print(f"Required current: I ≈ {I_needed:.2f} A per loop\n")

    # --- Plot 1: 3D geometry of coil pair ---
    phi = np.linspace(0, 2*np.pi, 401)
    x = a*np.cos(phi)
    y = a*np.sin(phi)
    z_upper = np.full_like(phi, d/2.0)
    z_lower = np.full_like(phi, -d/2.0)

    fig1 = plt.figure(figsize=(8, 6))
    ax1 = fig1.add_subplot(111, projection='3d')
    ax1.plot(x, y, z_upper, linewidth=2, label="Loop at +d/2 (current +I)")
    ax1.plot(x, y, z_lower, linewidth=2, label="Loop at -d/2 (current -I)")

    # Indicate current directions at one point
    phi_arrow = np.deg2rad(60)
    xA, yA = a*np.cos(phi_arrow), a*np.sin(phi_arrow)
    ax1.quiver(xA, yA, d/2.0, -yA*0.2, xA*0.2, 0, arrow_length_ratio=0.2)
    ax1.quiver(xA, yA, -d/2.0, yA*0.2, -xA*0.2, 0, arrow_length_ratio=0.2)

    # z-axis and center
    lim = max(0.6, 1.2*max(a, abs(d)/2))
    ax1.plot([0,0],[0,0],[-lim, lim], linestyle='--', linewidth=1)
    ax1.scatter(0,0,0,s=30)
    ax1.text(0.05,0.05,0.05,"center", fontsize=10)

    ax1.set_xlabel('x [m]')
    ax1.set_ylabel('y [m]')
    ax1.set_zlabel('z [m]')
    ax1.set_xlim(-lim, lim); ax1.set_ylim(-lim, lim); ax1.set_zlim(-lim, lim)
    ax1.set_title('Anti-Helmholtz pair: two loops with opposite current')
    ax1.legend(loc='upper left', fontsize=9)
    fig1.tight_layout()

    os.makedirs(args.outdir, exist_ok=True)
    fig1_png = os.path.join(args.outdir, "antihelmholtz_geometry.png")
    fig1_pdf = os.path.join(args.outdir, "antihelmholtz_geometry.pdf")
    fig1.savefig(fig1_png, dpi=200)
    fig1.savefig(fig1_pdf)

    # --- Plot 2: On-axis Bz(z) and linear approximation ---
    z = np.linspace(args.zmin, args.zmax, args.nz)
    Bz = Bz_antihelmholtz(z, a, d, I_needed, N=N)
    G0 = gradient_at_center(a, d, I_needed, N=N)
    Bz_lin = G0 * z

    fig2 = plt.figure(figsize=(8, 5.5))
    ax2 = fig2.add_subplot(111)
    ax2.plot(z, Bz, linewidth=2, label=r"$B_z(z)$ (exact)")
    ax2.plot(z, Bz_lin, linestyle='--', linewidth=1.5, label=r"Linear approx $Gz$ at $z=0$")
    ax2.set_xlabel("z [m]")
    ax2.set_ylabel(r"$B_z$ [T]")
    ax2.set_title(f"a={a:.2f} m, d={d:.2f} m, N={N}, I={I_needed:.1f} A  →  G={G0*1e3:.1f} mT/m")
    ax2.legend(loc="upper left", fontsize=9)
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()

    fig2_png = os.path.join(args.outdir, "antihelmholtz_Bz_vs_z.png")
    fig2_pdf = os.path.join(args.outdir, "antihelmholtz_Bz_vs_z.pdf")
    fig2.savefig(fig2_png, dpi=200)
    fig2.savefig(fig2_pdf)

    if args.csv:
        csv_path = os.path.join(args.outdir, "antihelmholtz_Bz.csv")
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["z_m", "Bz_T", "Bz_linear_T"])
            for zi, Bi, Bil in zip(z, Bz, Bz_lin):
                w.writerow([f"{zi:.8e}", f"{Bi:.8e}", f"{Bil:.8e}"])
        print(f"Wrote CSV: {csv_path}")

    print(f"Saved: {fig1_png}\n       {fig1_pdf}\n       {fig2_png}\n       {fig2_pdf}")

if __name__ == "__main__":
    main()
