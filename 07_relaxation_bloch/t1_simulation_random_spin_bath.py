# Scenario 3: Spin–lattice (T1) via Ornstein–Uhlenbeck transverse bath fields
# Outputs: t1_ou_spins.gif, t1_ou_magnetization.png

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation

SEED = 3
np.random.seed(SEED)

# ---------------- Parameters ----------------
N_SPINS = 401
BOX_SIZE = 1.0
MIN_DIST = 0.18
STEPS = 1000
DT = 0.012
SCALE = 0.25

GAMMA = 1.0
B0 = 30.0

TAU_C = 0.10        # OU correlation time
NOISE_SIGMA = 2.0   # OU RMS amplitude
T1_TARGET = 3.0     # weak macroscopic bias toward +z (None to disable)

# --------------- Helpers --------------------
def random_positions(n, box, min_dist):
    pts = []
    while len(pts) < n:
        p = np.random.uniform(-box, box, size=3)
        if not pts or all(np.linalg.norm(p - q) >= min_dist for q in pts):
            pts.append(p)
    return np.array(pts)

def unit(v):
    n = np.linalg.norm(v, axis=-1, keepdims=True) + 1e-12
    return v / n

def ou_update(x, dt, tau_c, sigma):
    # dx = -(x/tau_c) dt + sqrt(2 sigma^2 / tau_c) dW
    if tau_c <= 0:
        return sigma * np.random.randn(*x.shape)
    drift = -x / tau_c
    diff = np.sqrt(2 * (sigma**2) / tau_c)
    return x + drift * dt + diff * np.sqrt(dt) * np.random.randn(*x.shape)

def rotate_spins(spins, fields, gamma, dt):
    omega = gamma * fields
    angle = np.linalg.norm(omega, axis=1) * dt
    axis = unit(omega)
    ca = np.cos(angle)[:, None]
    sa = np.sin(angle)[:, None]
    axis_dot_S = np.sum(axis * spins, axis=1)[:, None]
    cross = np.cross(axis, spins)
    new_spins = spins * ca + cross * sa + axis * axis_dot_S * (1 - ca)
    return unit(new_spins)

# --------------- Init -----------------------
pos = random_positions(N_SPINS, BOX_SIZE, MIN_DIST)
spins = np.zeros((N_SPINS, 3)); spins[:, 0] = 1.0  # 90° pulse -> +x
B_noise = np.zeros((N_SPINS, 2))  # track OU states for Bx, By

M_record = np.zeros((STEPS, 3))
lines_data = []

# --------------- Simulate -------------------
for t_idx in range(STEPS):
    B_noise = ou_update(B_noise, DT, TAU_C, NOISE_SIGMA)
    Bx, By = B_noise[:,0], B_noise[:,1]
    B_tot = np.column_stack([Bx, By, np.full(N_SPINS, B0)])

    # weak macroscopic T1 bias (thermal reservoir)
    if T1_TARGET is not None and T1_TARGET > 0:
        alpha = DT / T1_TARGET
        spins = unit((1 - alpha) * spins + alpha * np.array([0,0,1.0]))

    spins = rotate_spins(spins, B_tot, GAMMA, DT)

    M_record[t_idx] = spins.mean(axis=0)
    seg = np.zeros((N_SPINS, 6))
    seg[:, 0:3] = pos
    seg[:, 3:6] = pos + SCALE * spins
    lines_data.append(seg)

# --------------- Animate --------------------
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection="3d")
ax.set_xlim(-BOX_SIZE, BOX_SIZE); ax.set_ylim(-BOX_SIZE, BOX_SIZE); ax.set_zlim(-BOX_SIZE, BOX_SIZE)
ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
ax.set_title("Spin–lattice (T1) Animation")

lines = []
seg0 = lines_data[0]
for i in range(N_SPINS):
    x0,y0,z0,x1,y1,z1 = seg0[i]
    ln, = ax.plot([x0,x1],[y0,y1],[z0,z1], linewidth=1.5)
    lines.append(ln)

def animate(k):
    seg = lines_data[k]
    for i, ln in enumerate(lines):
        x0,y0,z0,x1,y1,z1 = seg[i]
        ln.set_data([x0,x1],[y0,y1])
        ln.set_3d_properties([z0,z1])
    return lines

ani = animation.FuncAnimation(fig, animate, frames=STEPS, interval=30, blit=True)
ani.save("t1_ou_spins.gif", writer=animation.PillowWriter(fps=30))
plt.close(fig)

# --------------- Magnetization plot ---------
t = np.arange(STEPS) * DT
plt.figure(figsize=(6,4))
plt.plot(t, M_record[:,0], label="Mx")
plt.plot(t, M_record[:,1], label="My")
plt.plot(t, M_record[:,2], label="Mz")
plt.xlabel("time (arb. units)"); plt.ylabel("ensemble magnetization"); plt.legend()
plt.title("Spin–lattice (T1) Magnetization")
plt.tight_layout(); plt.savefig("t1_ou_magnetization.png", dpi=140); plt.close()