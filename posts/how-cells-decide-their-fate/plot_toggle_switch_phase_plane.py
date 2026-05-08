import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from scipy.integrate import solve_ivp
from scipy.optimize import root


def hill(x, k, n):
    return x**n / (k**n + x**n)


def dhill(x, k, n):
    return n * (k**n) * (x ** (n - 1)) / (k**n + x**n) ** 2


def rhs(xa, xb, p):
    dxa = p["c1"] * hill(xa, p["K1"], p["n1"]) + p["c2"] * (1.0 - hill(xb, p["K2"], p["n2"])) - p["gamma"] * xa
    dxb = p["c3"] * hill(xb, p["K3"], p["n3"]) + p["c4"] * (1.0 - hill(xa, p["K4"], p["n4"])) - p["gamma"] * xb
    return dxa, dxb


def jacobian(xa, xb, p):
    j11 = p["c1"] * dhill(xa, p["K1"], p["n1"]) - p["gamma"]
    j12 = -p["c2"] * dhill(xb, p["K2"], p["n2"])
    j21 = -p["c4"] * dhill(xa, p["K4"], p["n4"])
    j22 = p["c3"] * dhill(xb, p["K3"], p["n3"]) - p["gamma"]
    return np.array([[j11, j12], [j21, j22]], dtype=float)


def classify_fixed_points(eq, p):
    stable = []
    unstable = []

    for xa, xb in eq:
        eigvals = np.linalg.eigvals(jacobian(xa, xb, p))
        if np.max(np.real(eigvals)) < 0:
            stable.append([xa, xb])
        else:
            unstable.append([xa, xb])

    stable = np.array(stable) if stable else np.empty((0, 2))
    unstable = np.array(unstable) if unstable else np.empty((0, 2))
    return stable, unstable


def find_fixed_points(p, x_min, x_max, y_min, y_max):
    seeds_x = np.linspace(x_min, x_max, 8)
    seeds_y = np.linspace(y_min, y_max, 8)
    pts = []

    for sx in seeds_x:
        for sy in seeds_y:
            sol = root(lambda z: rhs(z[0], z[1], p), x0=[sx, sy])
            if not sol.success:
                continue
            xa, xb = sol.x
            if xa < x_min - 1e-6 or xa > x_max + 1e-6 or xb < y_min - 1e-6 or xb > y_max + 1e-6:
                continue
            if np.linalg.norm(rhs(xa, xb, p)) > 1e-6:
                continue
            candidate = np.array([xa, xb])
            if not any(np.linalg.norm(candidate - q) < 1e-3 for q in pts):
                pts.append(candidate)

    return np.array(pts)


def rhs_state(_t, z, p):
    xa, xb = z
    dxa, dxb = rhs(xa, xb, p)
    return [dxa, dxb]


def draw_phase_plane(ax, p, x_min, x_max, y_min, y_max, show_legend=True):
    xa = np.linspace(x_min, x_max, 140)
    xb = np.linspace(y_min, y_max, 140)
    XA, XB = np.meshgrid(xa, xb)

    dXA, dXB = rhs(XA, XB, p)
    speed = np.sqrt(dXA**2 + dXB**2)
    dXA_n = dXA / (speed + 1e-12)
    dXB_n = dXB / (speed + 1e-12)

    step = 6
    ax.quiver(
        XA[::step, ::step],
        XB[::step, ::step],
        dXA_n[::step, ::step],
        dXB_n[::step, ::step],
        color="0.35",
        alpha=0.85,
        scale=26,
        width=0.003,
    )

    nc_a = ax.contour(XA, XB, dXA, levels=[0], colors=["tab:red"], linewidths=2.2)
    nc_b = ax.contour(XA, XB, dXB, levels=[0], colors=["tab:blue"], linewidths=2.2)

    eq = find_fixed_points(p, x_min, x_max, y_min, y_max)
    stable_pts, unstable_pts = classify_fixed_points(eq, p)

    stable_scatter = None
    unstable_scatter = None

    if stable_pts.size > 0:
        stable_scatter = ax.scatter(
            stable_pts[:, 0],
            stable_pts[:, 1],
            s=60,
            facecolors="k",
            edgecolors="k",
            marker="o",
            zorder=4,
            label="stable fixed point",
        )

    if unstable_pts.size > 0:
        unstable_scatter = ax.scatter(
            unstable_pts[:, 0],
            unstable_pts[:, 1],
            s=60,
            facecolors="none",
            edgecolors="k",
            linewidths=1.6,
            marker="o",
            zorder=4,
            label="unstable fixed point",
        )

    if show_legend:
        h1, _ = nc_a.legend_elements()
        h2, _ = nc_b.legend_elements()
        handles = [h1[0], h2[0]]
        labels = [r"$\dot{x}_a=0$", r"$\dot{x}_b=0$"]
        if stable_scatter is not None:
            handles.append(stable_scatter)
            labels.append("stable fixed point")
        if unstable_scatter is not None:
            handles.append(unstable_scatter)
            labels.append("unstable fixed point")
        ax.legend(handles, labels, loc="upper right")

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel(r"$x_a$")
    ax.set_ylabel(r"$x_b$")
    ax.grid(alpha=0.2)


def make_two_cell_animation(p, x_min, x_max, y_min, y_max):
    initials = [np.array([0.5, 0.4]), np.array([0.4, 0.5])]
    t_span = (0.0, 10.0)
    t_eval = np.linspace(t_span[0], t_span[1], 220)

    sol1 = solve_ivp(rhs_state, t_span, initials[0], t_eval=t_eval, args=(p,), rtol=1e-8, atol=1e-10)
    sol2 = solve_ivp(rhs_state, t_span, initials[1], t_eval=t_eval, args=(p,), rtol=1e-8, atol=1e-10)

    fig, ax = plt.subplots(figsize=(6.8, 5.8))
    draw_phase_plane(ax, p, x_min, x_max, y_min, y_max, show_legend=False)
    ax.set_title("Two-cell trajectories on the toggle-switch phase plane")

    trail1 = ax.scatter([], [], c=[], cmap="viridis", vmin=t_span[0], vmax=t_span[1], s=18, marker="o", alpha=0.95)
    trail2 = ax.scatter([], [], c=[], cmap="viridis", vmin=t_span[0], vmax=t_span[1], s=20, marker="s", alpha=0.95)
    point1, = ax.plot([], [], "o", color="tab:green", mec="k", mew=0.5, ms=8, label="cell 1")
    point2, = ax.plot([], [], "o", color="tab:orange", mec="k", mew=0.5, ms=8, label="cell 2")
    time_text = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top")

    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(vmin=t_span[0], vmax=t_span[1]), cmap="viridis"), ax=ax)
    cbar.set_label("time")

    ax.legend(loc="upper right")

    def init():
        trail1.set_offsets(np.empty((0, 2)))
        trail2.set_offsets(np.empty((0, 2)))
        trail1.set_array(np.array([]))
        trail2.set_array(np.array([]))
        point1.set_data([], [])
        point2.set_data([], [])
        time_text.set_text("")
        return trail1, trail2, point1, point2, time_text

    def update(i):
        x1, y1 = sol1.y[0, : i + 1], sol1.y[1, : i + 1]
        x2, y2 = sol2.y[0, : i + 1], sol2.y[1, : i + 1]

        trail1.set_offsets(np.column_stack([x1, y1]))
        trail2.set_offsets(np.column_stack([x2, y2]))
        trail1.set_array(t_eval[: i + 1])
        trail2.set_array(t_eval[: i + 1])
        point1.set_data([x1[-1]], [y1[-1]])
        point2.set_data([x2[-1]], [y2[-1]])
        time_text.set_text(f"t = {t_eval[i]:.1f}")
        return trail1, trail2, point1, point2, time_text

    ani = animation.FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=len(t_eval),
        interval=45,
        blit=True,
    )

    ani.save("toggle_switch_two_cells.gif", writer=animation.PillowWriter(fps=20), dpi=120)
    plt.close(fig)


def main():
    # Toggle-switch model
    p = {
        "c1": 1.3,
        "c2": 2.0,
        "c3": 1.3,
        "c4": 2.0,
        "K1": 1.0,
        "K2": 1.0,
        "K3": 1.0,
        "K4": 1.0,
        "n1": 4,
        "n2": 4,
        "n3": 4,
        "n4": 4,
        "gamma": 1.0,
    }

    x_min, x_max = 0.0, 4.0
    y_min, y_max = 0.0, 4.0

    fig, ax = plt.subplots(figsize=(6.5, 5.6))
    draw_phase_plane(ax, p, x_min, x_max, y_min, y_max, show_legend=True)
    ax.set_title("Toggle switch: vector field and nullclines")

    plt.tight_layout()
    plt.savefig("toggle_switch_phase_plane.png", dpi=180)
    make_two_cell_animation(p, x_min, x_max, y_min, y_max)
    plt.show()


if __name__ == "__main__":
    main()
