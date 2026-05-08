import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 4, 500)

fig, axes = plt.subplots(1, 2, figsize=(8, 3))

# Left: vary n, fix K=1
K = 1
for n in [1, 2, 4, 8]:
    y = x**n / (K**n + x**n)
    axes[0].plot(x, y, label=f"n = {n}")
axes[0].set_title("Hill function: varying $n$ ($K = 1$)")
axes[0].set_xlabel("$x$")
axes[0].set_ylabel("$f(x)$")
axes[0].axvline(K, color="gray", linestyle="--", linewidth=0.8, label="$x = K$")
axes[0].legend()

# Right: vary K, fix n=5
n = 5
for K in [0.5, 1, 2, 3]:
    y = x**n / (K**n + x**n)
    axes[1].plot(x, y, label=f"K = {K}")
axes[1].set_title("Hill function: varying $K$ ($n = 5$)")
axes[1].set_xlabel("$x$")
axes[1].set_ylabel("$f(x)$")
axes[1].legend()

plt.tight_layout()
plt.savefig("hill_function.png", dpi=150)
plt.show()
