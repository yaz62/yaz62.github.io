import numpy as np
import matplotlib.pyplot as plt


# Parameters for h_{K,n}(x_b) and x_a^{ss} = (c/gamma) * (...) 
C = 1.0
GAMMA = 1.0
K = 1.0
N = 4.0
X_B_MIN = 0.0
X_B_MAX = 4.0
NUM_POINTS = 400
OUTPUT_FIG = "steady_states_vs_xb.png"


def hill(x: np.ndarray, k: float, n: float) -> np.ndarray:
    return x**n / (k**n + x**n)


def main() -> None:
    x_b = np.linspace(X_B_MIN, X_B_MAX, NUM_POINTS)
    h = hill(x_b, K, N)

    # Steady states from the post:
    # activation: x_a^ss = (c/gamma) h_{K,n}(x_b)
    # inhibition: x_a^ss = (c/gamma) (1 - h_{K,n}(x_b))
    x_a_ss_activation = (C / GAMMA) * h
    x_a_ss_inhibition = (C / GAMMA) * (1.0 - h)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(x_b, x_a_ss_activation, lw=2.5, color="#d62728", label=r"activation: $x_a^{ss}=\frac{c}{\gamma}h_{K,n}(x_b)$")
    ax.plot(x_b, x_a_ss_inhibition, lw=2.5, color="#1f77b4", label=r"inhibition: $x_a^{ss}=\frac{c}{\gamma}(1-h_{K,n}(x_b))$")

    ax.axhline(C / GAMMA, color="gray", lw=1.0, ls="--", alpha=0.6, label=r"$c/\gamma$")

    ax.set_title("Steady-State Expression of Gene a vs Input $x_b$")
    ax.set_xlabel(r"$x_b$")
    ax.set_ylabel(r"$x_a^{ss}$")
    ax.set_xlim(X_B_MIN, X_B_MAX)
    ax.set_ylim(-0.02, C / GAMMA + 0.05)
    ax.legend(frameon=False, fontsize=10)
    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(OUTPUT_FIG, dpi=300)
    plt.show()


if __name__ == "__main__":
    main()
