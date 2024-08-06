import numpy as np
import pandas as pd
# Good Lattice Points(GLP)
def GLP(n, s, generator=None):
    nvec = np.arange(1, n + 1)  # Create an array from 1 to n
    m = len(generator)  # Length of the generator

    if s > m:
        print("ERROR: s should not be larger than the length of the generator!")
        return None, None

    # Create the initial hypercube sample using the generator
    ih = np.kron(nvec, np.array(generator))  # Ensure generator is a NumPy array
    Um = ih % n  # Modulo operation
    Um[Um == 0] = n  # Replace 0s with n
    Um = Um.reshape(len(nvec), len(generator))
    return Um, generator

# Generalized Good Lattice Points(GGLP)
def GGLP(Um, row_vector, n):
    Um = (Um + row_vector) % n
    Um[Um == 0] = n
    return Um

# Generate samples for a mixture of two normals using GLP method
def generate_mixed_normal_samples_GLP(mu1, sigma1, mu2, sigma2, alpha, n, GLP1):
    A1 = np.linalg.cholesky(sigma1)  # Cholesky decomposition of sigma1
    A2 = np.linalg.cholesky(sigma2)  # Cholesky decomposition of sigma2

    samples = []
    for i in range(n):
        u = GLP1[i]  # Use values from GLP1 instead of generating random numbers

        # Box-Muller transform to generate normal random variables
        r1 = np.sqrt(-2 * np.log(u[0]))
        r2 = np.sqrt(-2 * np.log(u[2]))

        n1 = r1 * np.cos(2 * np.pi * u[1])
        n2 = r1 * np.sin(2 * np.pi * u[1])
        n3 = r2 * np.cos(2 * np.pi * u[3])
        n4 = r2 * np.sin(2 * np.pi * u[3])

        N1 = mu1 + np.dot(A1, np.array([n1, n2]))
        N2 = mu2 + np.dot(A2, np.array([n3, n4]))

        X = N1 if u[4] <= alpha else N2  # Mixture decision

        samples.append(X)

    return np.array(samples)

# Generate samples for a mixture of two normals using GGLP method
def generate_mixed_normal_samples_GGLP(mu1, sigma1, mu2, sigma2, alpha, n, GGLP1):
    A1 = np.linalg.cholesky(sigma1)  # Cholesky decomposition of sigma1
    A2 = np.linalg.cholesky(sigma2)  # Cholesky decomposition of sigma2

    samples = []
    for i in range(n):
        u = GGLP1[i]  # Use values from GGLP1 instead of generating random numbers

        # Box-Muller transform to generate normal random variables
        r1 = np.sqrt(-2 * np.log(u[0]))
        r2 = np.sqrt(-2 * np.log(u[2]))

        n1 = r1 * np.cos(2 * np.pi * u[1])
        n2 = r1 * np.sin(2 * np.pi * u[1])
        n3 = r2 * np.cos(2 * np.pi * u[3])
        n4 = r2 * np.sin(2 * np.pi * u[3])

        N1 = mu1 + np.dot(A1, np.array([n1, n2]))
        N2 = mu2 + np.dot(A2, np.array([n3, n4]))

        X = N1 if u[4] <= alpha else N2  # Mixture decision

        samples.append(X)

    return np.array(samples)

# Generate samples for a mixture of two normals using Monte Carlo method
def generate_mixed_normal_samples_MC(mu1, sigma1, mu2, sigma2, alpha, n):
    A1 = np.linalg.cholesky(sigma1)  # Cholesky decomposition of sigma1
    A2 = np.linalg.cholesky(sigma2)  # Cholesky decomposition of sigma2

    samples = []
    for _ in range(n):
        u = np.random.rand(5)  # Generate random numbers

        # Box-Muller transform to generate normal random variables
        r1 = np.sqrt(-2 * np.log(u[0]))
        r2 = np.sqrt(-2 * np.log(u[2]))

        n1 = r1 * np.cos(2 * np.pi * u[1])
        n2 = r1 * np.sin(2 * np.pi * u[1])
        n3 = r2 * np.cos(2 * np.pi * u[3])
        n4 = r2 * np.sin(2 * np.pi * u[3])

        N1 = mu1 + np.dot(A1, np.array([n1, n2]))
        N2 = mu2 + np.dot(A2, np.array([n3, n4]))

        X = N1 if u[4] <= alpha else N2  # Mixture decision

        samples.append(X)

    return np.array(samples)

# Calculate mean squared error between samples and reference
def calculate_mse(samples, reference):
    return np.mean((samples - reference) ** 2)

# Estimate mean and covariance of samples
def estimate_mean_and_covariance(samples):
    k = len(samples)
    mu_hat = np.mean(samples, axis=0)
    s_hat = np.zeros((samples[0].shape[0], samples[0].shape[0]), dtype=samples[0].dtype)
    for i in range(k):
        s_hat += np.outer(samples[i] - mu_hat, samples[i] - mu_hat)
    s_hat /= k
    return mu_hat, s_hat

# Main Function
def main():
    mixtures = [
        {
            "type": "Scale Mixture",
            "mu1": np.array([0, 1]),
            "sigma1": np.array([[1, 0.5], [0.5, 2]]),
            "mu2": np.array([0, 1]),
            "sigma2": np.array([[1, -0.3], [-0.3, 1.2]]),
            "alpha": 0.7
        },
        {
            "type": "Location Mixture",
            "mu1": np.array([0, 1]),
            "sigma1": np.array([[1, 0.5], [0.5, 1]]),
            "mu2": np.array([3, 4]),
            "sigma2": np.array([[1, 0.5], [0.5, 1]]),
            "alpha": 0.7
        },
        {
            "type": "Special Case",
            "mu1": np.array([-1, 1]),
            "sigma1": np.array([[2, 0.5], [0.5, 1]]),
            "mu2": np.array([1, -1]),
            "sigma2": np.array([[2, 0.5], [0.5, 1]]),
            "alpha": 0.7
        },
        {
            "type": "More Special Case",
            "mu1": np.array([-1, 1]),
            "sigma1": np.array([[1, 0.5], [0.5, 1]]),
            "mu2": np.array([1, -1]),
            "sigma2": np.array([[1, 0.5], [0.5, 1]]),
            "alpha": 0.7
        }
    ]

    n_values = [11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 1069, 1543, 2129, 3001, 4001, 5003, 6007, 8191]  # Number of samples
    B = 5000  # Number of bootstrap iterations

    row_vectors = {
        11: np.array([10, 10, 10, 10, 10]),
        13: np.array([0, 6, 11, 9, 11]),
        15: np.array([0, 9, 0, 8, 6]),
        17: np.array([0, 12, 13, 2, 12]),
        19: np.array([0, 13, 18, 8, 4]),
        21: np.array([0, 5, 7, 13, 14]),
        23: np.array([17, 17, 17, 17, 17]),
        25: np.array([20, 20, 20, 20, 20]),
        27: np.array([0, 1, 21, 25, 8]),
        29: np.array([0, 17, 26, 3, 27]),
        31: np.array([8, 8, 8, 8, 8]),
        1069: np.array([557, 557, 557, 557, 557]),
        1543: np.array([1389, 1389, 1389, 1389, 1389]),
        2129: np.array([1173, 1173, 1173, 1173, 1173]),
        3001: np.array([1558, 1558, 1558, 1558, 1558]),
        4001: np.array([2533, 2533, 2533, 2533, 2533]),
        5003: np.array([0, 4446, 4815, 4068, 4393]),
        6007: np.array([5451, 5451, 5451, 5451, 5451]),
        8191: np.array([6214, 6214, 6214, 6214, 6214])
    }

    h1_values = {
        11: [1, 3, 4, 5, 9],
        13: [1, 6, 8, 9, 10],
        15: [1, 2, 4, 7, 13],
        17: [1, 3, 9, 10, 13],
        19: [1, 4, 6, 7, 17],
        21: [1, 2, 10, 13, 16],
        23: [1, 6, 11, 13, 20],
        25: [1, 4, 6, 9, 11],
        27: [1, 2, 4, 8, 16],
        29: [1, 14, 18, 20, 22],
        31: [1, 2, 4, 8, 16],
        1069: [1, 63, 762, 970, 177],
        1543: [1, 58, 278, 694, 134],
        2129: [1, 618, 833, 1705, 1964],
        3001: [1, 408, 1409, 1681, 1620],
        4001: [1, 1534, 568, 3095, 2544],
        5003: [1, 840, 117, 3593, 1311],
        6007: [1, 509, 780, 558, 1693],
        8191: [1, 1386, 4302, 7715, 3735]
    }

    results = []

    for mixture in mixtures:
        mu1 = mixture["mu1"]
        sigma1 = mixture["sigma1"]
        mu2 = mixture["mu2"]
        sigma2 = mixture["sigma2"]
        alpha = mixture["alpha"]

        true_mu = alpha * mu1 + (1 - alpha) * mu2
        true_sigma = alpha * sigma1 + (1 - alpha) * sigma2

        for n in n_values:
            mean_errors_glp = []
            mean_errors_gglp = []
            mean_errors_mc = []
            var_errors_glp = []
            var_errors_gglp = []
            var_errors_mc = []

            row_vector = row_vectors.get(n, None)
            h1 = h1_values.get(n, None)

            if row_vector is None or h1 is None:
                print(f"Warning: Unknown n value {n}. Skipping...")
                continue

            for b in range(B):
                # Print current n value and iteration count
                print(f"Iteration {b + 1}/{B}, n = {n}, Mixture Type: {mixture['type']}")

                # Generate GLP and GGLP samples
                GLP0, _ = GLP(n, 5, h1)
                GLP1 = (2 * GLP0 - 1) / (2 * n)

                GGLP0 = GGLP(GLP0, row_vector, n)
                GGLP1 = (2 * GGLP0 - 1) / (2 * n)


                # Generate samples using GLP, GGLP, and Monte Carlo
                samples_GLP = generate_mixed_normal_samples_GLP(mu1, sigma1, mu2, sigma2, alpha, n, GLP1)
                samples_GGLP = generate_mixed_normal_samples_GGLP(mu1, sigma1, mu2, sigma2, alpha, n, GGLP1)
                samples_MC = generate_mixed_normal_samples_MC(mu1, sigma1, mu2, sigma2, alpha, n)

                # Bootstrap samples n times
                indices = np.random.choice(n, n, replace=True)
                bootstrap_samples_GLP = samples_GLP[indices]
                bootstrap_samples_GGLP = samples_GGLP[indices]
                bootstrap_samples_MC = samples_MC[indices]

                # Estimate mean and covariance
                mu_hat_GLP, sigma_hat_GLP = estimate_mean_and_covariance(bootstrap_samples_GLP)
                mu_hat_GGLP, sigma_hat_GGLP = estimate_mean_and_covariance(bootstrap_samples_GGLP)
                mu_hat_MC, sigma_hat_MC = estimate_mean_and_covariance(bootstrap_samples_MC)

                # Calculate mean squared error for mean
                mean_errors_glp.append(calculate_mse(mu_hat_GLP, true_mu))
                mean_errors_gglp.append(calculate_mse(mu_hat_GGLP, true_mu))
                mean_errors_mc.append(calculate_mse(mu_hat_MC, true_mu))

                # Calculate mean squared error for variance
                var_errors_glp.append(calculate_mse(sigma_hat_GLP, true_sigma))
                var_errors_gglp.append(calculate_mse(sigma_hat_GGLP, true_sigma))
                var_errors_mc.append(calculate_mse(sigma_hat_MC, true_sigma))

            results.append({
                'mixture_type': mixture['type'],
                'n': n,
                'mean_error_glp': round(np.mean(mean_errors_glp), 6),
                'mean_error_gglp': round(np.mean(mean_errors_gglp), 6),
                'mean_error_mc': round(np.mean(mean_errors_mc), 6),
                'var_error_glp': round(np.mean(var_errors_glp), 6),
                'var_error_gglp': round(np.mean(var_errors_gglp), 6),
                'var_error_mc': round(np.mean(var_errors_mc), 6)
            })

            print(f"Completed n = {n}, Mixture Type: {mixture['type']}")

    # Create DataFrame from results
    df = pd.DataFrame(results)

    # Save DataFrame to Excel
    excel_file = 'QMC_results.xlsx'
    df.to_excel(excel_file, index=False)
    print(f"Results saved to {excel_file}")

if __name__ == "__main__":
    main()
