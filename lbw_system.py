import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.animation import FuncAnimation

# --------------------------
# Custom trapezoidal integral
# --------------------------
def cumtrapz_manual(y, x):
    out = np.zeros_like(y)
    for i in range(1, len(y)):
        out[i] = out[i - 1] + 0.5 * (y[i] + y[i - 1]) * (x[i] - x[i - 1])
    return out

# --------------------------
# Constants
# --------------------------
pitch_len = 20.12
pitch_width = 4.0
wicket_w = 0.2286  # 9 inches between outer stumps
stump_r = 0.019  # realistic radius
stump_h = 0.9  # ← TALLER STUMPS

# Y positions for stumps
mid_y = 0.0
off_y = wicket_w / 2
leg_y = -wicket_w / 2
stump_centres = [leg_y, mid_y, off_y]

# --------------------------
# Ball parameters – Changeable
# --------------------------
x0 = pitch_len
y0 = 1.4
z0 = -0.5
v0 = 48.0
angle_deg = 2
max_swing_deg = 1.2
swing_dir = "inswing"  # or "outswing"
swing_decay = 0.8
g = 9.81
coef_rest = 0.6
total_time = 2.0
points = 400

# --------------------------
# Compute trajectory
# --------------------------
angle = np.radians(angle_deg)
swing_mag = np.radians(max_swing_deg)
sign = -1 if swing_dir == "inswing" else 1
t = np.linspace(0, total_time, points)

vz_const = sign * v0 * np.sin(swing_mag)
v_xy = v0 * np.cos(angle)
vx_const = np.sqrt(v_xy ** 2 - vz_const ** 2)

x_pre = x0 - vx_const * t
z_pre = z0 + vz_const * t
y_pre = y0 + v0 * np.sin(angle) * t - 0.5 * g * t ** 2

bounce_idx = np.argmax(y_pre <= 0)
t_b = t[bounce_idx]
x_b, y_b, z_b = x_pre[bounce_idx], y_pre[bounce_idx], z_pre[bounce_idx]
vy_b = v0 * np.sin(angle) - g * t_b

# After bounce
t_after = t[bounce_idx:] - t_b
vy_after0 = -vy_b * coef_rest
vz_after = vz_const * np.maximum(0, 1 - t_after / swing_decay)
vx_after = np.sqrt(np.maximum(0.0, v_xy ** 2 - vz_after ** 2))

y_after = vy_after0 * t_after - 0.5 * g * t_after ** 2
valid = y_after >= 0
x_after = x_b - cumtrapz_manual(vx_after, t_after)[valid]
z_after = z_b + cumtrapz_manual(vz_after, t_after)[valid]
y_after = y_after[valid]

# Combine
x_full = np.concatenate((x_pre[:bounce_idx], x_after))
y_full = np.concatenate((y_pre[:bounce_idx], y_after))
z_full = np.concatenate((z_pre[:bounce_idx], z_after))

# Animation data
step = 5
x_anim, y_anim, z_anim = x_full[::step], y_full[::step], z_full[::step]

# --------------------------
# Plot setup
# --------------------------
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

# Pitch
XX, YY = np.meshgrid([0, pitch_len], [-pitch_width / 2, pitch_width / 2])
ZZ = np.zeros_like(XX)
ax.plot_surface(XX, YY, ZZ, color='lightgreen', alpha=0.6)

# Yellow pitching corridor
verts = [list(zip([0, pitch_len, pitch_len, 0],
                  [leg_y, leg_y, off_y, off_y],
                  [0, 0, 0, 0]))]
ax.add_collection3d(Poly3DCollection(verts, facecolors='yellow', alpha=0.15))

# Stumps (taller)
theta = np.linspace(0, 2 * np.pi, 20)
for yc in stump_centres:
    Xc = np.zeros_like(theta)
    Yc = yc + stump_r * np.cos(theta)
    Zc = stump_r * np.sin(theta)
    ax.plot_surface(np.vstack([Xc, Xc]),
                    np.vstack([Yc, Yc]),
                    np.vstack([Zc * 0, Zc * 0 + stump_h]),
                    color='brown')

# Trajectory
ax.plot(x_full, z_full, y_full, '--', color='gray', label='Trajectory')

# Ball and trail
ball, = ax.plot([], [], [], 'ro', ms=8, label='Ball')
trail, = ax.plot([], [], [], 'o', color='black', alpha=0.3, ms=6)
trail_len = 15

# Axes & view
ax.set_xlim(-1, pitch_len + 1)
ax.set_ylim(-pitch_width / 2, pitch_width / 2)
ax.set_zlim(0, 3)
ax.set_xlabel('Distance (m)')
ax.set_ylabel('Side offset (m)')
ax.set_zlabel('Height (m)')
ax.set_title('Cricket Ball Trajectory – Realistic Tall Stumps')
ax.legend()

# --------------------------
# Animation update
# --------------------------
def update(i):
    ball.set_data([x_anim[i]], [z_anim[i]])
    ball.set_3d_properties([y_anim[i]])
    start = max(0, i - trail_len)
    trail.set_data(x_anim[start:i], z_anim[start:i])
    trail.set_3d_properties(y_anim[start:i])
    return ball, trail

ani = FuncAnimation(fig, update, frames=len(x_anim), interval=30, blit=False)
plt.tight_layout()
plt.show()
