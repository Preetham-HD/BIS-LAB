import cv2
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Grey Wolf Optimization (GWO)
# -----------------------------
def gwo_optimize(obj_func, lb, ub, dim, num_agents=10, max_iter=50):
    # Initialize the positions of search agents
    positions = np.random.uniform(lb, ub, (num_agents, dim))

    # Initialize alpha, beta, and delta wolves
    alpha_pos = np.zeros(dim)
    beta_pos = np.zeros(dim)
    delta_pos = np.zeros(dim)
    alpha_score = float('inf')
    beta_score = float('inf')
    delta_score = float('inf')

    # Main loop
    for t in range(max_iter):
        for i in range(num_agents):
            # Keep search agents within bounds
            positions[i] = np.clip(positions[i], lb, ub)

            # Calculate fitness
            fitness = obj_func(positions[i])

            # Update alpha, beta, delta
            if fitness < alpha_score:
                delta_score = beta_score
                delta_pos = beta_pos.copy()
                beta_score = alpha_score
                beta_pos = alpha_pos.copy()
                alpha_score = fitness
                alpha_pos = positions[i].copy()
            elif fitness < beta_score:
                delta_score = beta_score
                delta_pos = beta_pos.copy()
                beta_score = fitness
                beta_pos = positions[i].copy()
            elif fitness < delta_score:
                delta_score = fitness
                delta_pos = positions[i].copy()

        # Coefficients a, A, and C
        a = 2 - t * (2 / max_iter)

        # Update positions
        for i in range(num_agents):
            for j in range(dim):
                r1, r2 = np.random.rand(), np.random.rand()
                A1 = 2 * a * r1 - a
                C1 = 2 * r2
                D_alpha = abs(C1 * alpha_pos[j] - positions[i][j])
                X1 = alpha_pos[j] - A1 * D_alpha

                r1, r2 = np.random.rand(), np.random.rand()
                A2 = 2 * a * r1 - a
                C2 = 2 * r2
                D_beta = abs(C2 * beta_pos[j] - positions[i][j])
                X2 = beta_pos[j] - A2 * D_beta

                r1, r2 = np.random.rand(), np.random.rand()
                A3 = 2 * a * r1 - a
                C3 = 2 * r2
                D_delta = abs(C3 * delta_pos[j] - positions[i][j])
                X3 = delta_pos[j] - A3 * D_delta

                positions[i][j] = (X1 + X2 + X3) / 3.0

    return alpha_pos, alpha_score


# -----------------------------
# Objective Function (Otsu's variance)
# -----------------------------
def otsu_threshold_objective(threshold, gray_img):
    threshold = int(threshold)
    if threshold <= 0 or threshold >= 255:
        return float('inf')

    hist, _ = np.histogram(gray_img, bins=256, range=(0, 256))
    hist = hist.astype(np.float32)
    total = gray_img.size

    w0 = np.sum(hist[:threshold]) / total
    w1 = np.sum(hist[threshold:]) / total
    if w0 == 0 or w1 == 0:
        return float('inf')

    m0 = np.sum(np.arange(threshold) * hist[:threshold]) / (np.sum(hist[:threshold]) + 1e-6)
    m1 = np.sum(np.arange(threshold, 256) * hist[threshold:]) / (np.sum(hist[threshold:]) + 1e-6)

    # Between-class variance (maximize, so we return negative for minimization)
    var_between = w0 * w1 * (m0 - m1) ** 2
    return -var_between


# -----------------------------
# Main function
# -----------------------------
def gwo_image_threshold(image_path):
    # Read and convert to grayscale
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Define objective wrapper
    def obj_func(threshold):
        return otsu_threshold_objective(threshold, gray)

    # Run GWO
    best_threshold, best_score = gwo_optimize(obj_func, lb=0, ub=255, dim=1, num_agents=15, max_iter=50)
    best_threshold = int(best_threshold[0])
    print(f"Optimal Threshold found by GWO: {best_threshold}")

    # Apply threshold
    _, bw_img = cv2.threshold(gray, best_threshold, 255, cv2.THRESH_BINARY)

    # Display results
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    plt.subplot(1, 3, 2)
    plt.title("Grayscale")
    plt.imshow(gray, cmap='gray')

    plt.subplot(1, 3, 3)
    plt.title(f"B&W (GWO Threshold={best_threshold})")
    plt.imshow(bw_img, cmap='gray')
    plt.tight_layout()
    plt.show()

    return bw_img, best_threshold


# Example usage
gwo_image_threshold("download.jpeg")
