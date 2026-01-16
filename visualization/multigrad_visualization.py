import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

# --- Hyperparameters ---
alpha = 0.05  # Inner loop learning rate
beta = 0.5  # Meta learning rate
num_steps = 5  # Number of inner steps
meta_steps = 500  # Number of meta steps
tol = 1e-3  # Convergence tolerance

# Multi-objective settings
tau = 0.0  # Temperature: 0 = hard argmax, >0 = softmax
num_intrinsic_optima = 1  # How many optima to use from the dictionary

# distinct colors for each intrinsic goal
COLORS = ["#FF0000", "#0000FF", "#00FF00", "#FF00FF", "#00FFFF", "#FFA500"]

# Dictionary of available intrinsic definitions
INTRINSIC_DEFINITIONS = {
    0: (0, -2),
    1: (-2, 2),
    2: (2, 2),
    3: (-2, -2),
    4: (2, -2),
}

# Select active optima based on hyperparameter
active_indices = list(INTRINSIC_DEFINITIONS.keys())[:num_intrinsic_optima]
active_optima = [
    torch.tensor(INTRINSIC_DEFINITIONS[i], dtype=torch.float32) for i in active_indices
]

# --- Loss Functions ---


def get_pseudo_loss(x, target):
    """Computes distance to a specific intrinsic target."""
    return torch.norm(x - target) ** 2


def loss_fn(x):
    """Extrinsic objective (distance to origin)."""
    return torch.norm(x) ** 2


# --- Meta-Gradient Computation ---


def compute_single_path_gradient(theta_init, target_optimum):
    """
    Computes the meta-gradient for a single intrinsic objective.
    Returns: meta_gradient, trajectory_list, final_extrinsic_loss
    """
    theta_list = []
    grad_list = []

    theta = theta_init.clone().detach().requires_grad_(True)
    theta_list.append(theta)

    # 1. Inner Loop Optimization
    for _ in range(num_steps):
        loss = get_pseudo_loss(theta, target_optimum)
        grad = torch.autograd.grad(loss, theta, create_graph=True)[0]
        grad_list.append(grad)
        theta = theta - alpha * grad
        theta_list.append(theta)

    # 2. Compute Meta Loss (Extrinsic)
    loss_meta = loss_fn(theta)

    # 3. Compute Gradient wrt Theta_0 (MAML-style differentiation)
    v_final = torch.autograd.grad(loss_meta, theta, retain_graph=True)[0]

    v = v_final
    for i in reversed(range(num_steps)):
        H_v = torch.autograd.grad(
            grad_list[i], theta_list[i], grad_outputs=v, retain_graph=True
        )[0]
        v = v - alpha * H_v

    return v.detach(), theta_list, loss_meta.item()


def compute_aggregated_gradient(theta_init):
    """
    Aggregates gradients from all active intrinsic optima using softmax/argmax.
    """
    gradients = []
    trajectories = []
    extrinsic_losses = []

    # Run independent inner loops for each intrinsic optimum
    for target in active_optima:
        grad, traj, meta_loss = compute_single_path_gradient(theta_init, target)
        gradients.append(grad)
        trajectories.append(traj)
        extrinsic_losses.append(meta_loss)

    gradients = torch.stack(gradients)
    extrinsic_losses = torch.tensor(extrinsic_losses)

    # Calculate Weights
    # We want to MAXIMIZE the objective, so we treat negative loss as utility.
    scores = -extrinsic_losses

    if tau == 0:
        # Hard Argmax
        best_idx = torch.argmax(scores)
        weights = torch.zeros_like(scores)
        weights[best_idx] = 1.0
    else:
        # Softmax
        # scores_shifted = scores - torch.max(scores)
        # weights = F.softmax(scores_shifted / tau, dim=0)
        weights = F.softmax(scores / tau, dim=0)

    # Aggregate Gradients: sum(w_i * grad_i)
    weighted_grads = gradients * weights.unsqueeze(1)
    meta_grad = torch.sum(weighted_grads, dim=0)

    return meta_grad, trajectories, weights


# --- Main Optimization Loop ---

theta0 = torch.tensor([6.0, -6.0])
meta_trajectory = [theta0.numpy()]
all_step_trajectories = []  # Stores list of lists of trajectories

print(f"Starting Meta-Optimization with {num_intrinsic_optima} optima and tau={tau}...")

for i in range(meta_steps):
    meta_grad, paths, weights = compute_aggregated_gradient(theta0)

    # Store trajectories for visualization
    current_step_paths = []
    for p in paths:
        current_step_paths.append(torch.stack([t.detach() for t in p]).numpy())
    all_step_trajectories.append(current_step_paths)

    # Check convergence
    if torch.linalg.norm(meta_grad, 2) < tol:
        meta_steps = i
        break

    # Meta Update
    theta0 = theta0 - beta * meta_grad
    meta_trajectory.append(theta0.numpy())

print(f"Converged in {meta_steps} steps.")
meta_trajectory = np.stack(meta_trajectory)


# --- Plotting ---

# Prepare background contour
x = np.linspace(-8, 8, 100)
y = np.linspace(-8, 8, 100)
X, Y = np.meshgrid(x, y)
Z = X**2 + Y**2

plt.figure(figsize=(12, 12))

# 1. Background Contours
plt.contourf(
    X, Y, -Z, levels=25, cmap="gray", alpha=0.5
)  # Changed to gray to make colors pop
# plt.contour(X, Y, -Z, levels=10, colors="white", alpha=0.3, linewidths=0.5)

# Start Point
plt.scatter(
    meta_trajectory[0, 0],
    meta_trajectory[0, 1],
    s=500,
    color="white",
    edgecolors="black",
    linewidths=2,
    label=r"Initial base policy $\theta$",
    zorder=7,
)

# End Point
plt.scatter(
    meta_trajectory[-1, 0],
    meta_trajectory[-1, 1],
    s=500,
    color="black",
    edgecolors="white",
    linewidths=2,
    label=r"Resulting base policy $\theta^\dagger$",
    zorder=7,
)


# Intrinsic Optima (Multiple with distinct colors)
for idx, opt in enumerate(active_optima):
    opt_np = opt.numpy()
    c = COLORS[idx % len(COLORS)]
    plt.scatter(
        opt_np[0],
        opt_np[1],
        s=1500,
        color=c,
        edgecolors="black",
        linewidths=2,
        marker="*",
        label=f"Intrinsic optimum ({idx + 1})",
        zorder=5,
    )
    # # Label the stars
    # plt.text(
    #     opt_np[0] - 1.7,
    #     opt_np[1] + 0.5,
    #     f"Opt ({idx + 1})",
    #     color=c,
    #     fontsize=14,
    #     fontweight="bold",
    #     bbox=dict(facecolor="white", alpha=0.5, edgecolor="none", pad=1),
    # )

# Extrinsic Optimum (Target)
plt.scatter(
    0,
    0,
    s=1500,
    color="yellow",
    edgecolors="black",
    linewidths=2,
    marker="*",
    label="Extrinsic optimum",
    zorder=6,
)


# 3. Plot Meta-Gradient Path (Outer Loop)
X_meta = meta_trajectory[:-1, 0]
Y_meta = meta_trajectory[:-1, 1]
U_meta = meta_trajectory[1:, 0] - meta_trajectory[:-1, 0]
V_meta = meta_trajectory[1:, 1] - meta_trajectory[:-1, 1]

plt.quiver(
    X_meta,
    Y_meta,
    U_meta,
    V_meta,
    angles="xy",
    scale_units="xy",
    scale=1,
    color="white",  # Changed to white/black for visibility
    edgecolor="black",
    alpha=1.0,
    linewidth=1.0,
    label="Base policy updates",
    zorder=10,
)


# 2. Plot Inner Loop Trajectories
for i, step_paths in enumerate(all_step_trajectories):
    # Calculate transparency based on time (newer = more opaque)
    alpha_val = 0.9 - 0.6 * (i / meta_steps)
    if alpha_val < 0.1:
        alpha_val = 0.1

    for path_idx, traj in enumerate(step_paths):
        # Pick color corresponding to the intrinsic optimum
        c = COLORS[path_idx % len(COLORS)]

        start = traj[0]
        end = traj[-1]
        dx = end[0] - start[0]
        dy = end[1] - start[1]

        if i % 2 != 0:
            continue  # Plot every 10th for clarity
        plt.quiver(
            start[0],
            start[1],
            dx,
            dy,
            angles="xy",
            scale_units="xy",
            scale=1,
            color=c,
            alpha=alpha_val,
            linewidth=1.5,
            headwidth=3,
            label=f"Exploratory policy updates ({path_idx + 1})" if i == 0 else None,
        )


# 4. Plot Points


# Labels and Styling
plt.title(
    rf"IRPO ($\tau$={tau}, $N$={num_steps}, $K$={num_intrinsic_optima})", fontsize=42
)
plt.xlabel(r"$\theta_1$", fontsize=32)
plt.ylabel(r"$\theta_2$", fontsize=32)
plt.tick_params(axis="both", which="major", labelsize=24)
# plt.axis("equal")
plt.grid(True, linestyle="--", alpha=0.3, color="white")

# Adjust legend to show distinct colors
# plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.05), ncol=4, fontsize=14)

plt.tight_layout()
plt.savefig("multi_objective.svg", bbox_inches="tight", dpi=300)
