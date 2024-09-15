# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 19:32:45 2024

@author: HP
"""

import numpy as np
import pandas as pd
from scipy.special import rel_entr  # To compute relative entropy (KL divergence)
from scipy.stats import gaussian_kde
import math
# Good Lattice Points (GLP)
def GLP(n, s, generator=None):
    nvec = np.arange(1, n + 1)  # Create an array from 1 to n
    m = len(generator)  # Length of the generator

    if s > m:
        print("ERROR: s should not be larger than the length of the generator!")
        return None, None

    ih = np.kron(nvec, np.array(generator))  # Ensure generator is a NumPy array
    Um = ih % n  # Modulo operation
    Um[Um == 0] = n  # Replace 0s with n
    Um = Um.reshape(len(nvec), len(generator))
    return Um, generator

# Generalized Good Lattice Points (GGLP)
def GGLP(Um, row_vector, n):
    Um = (Um + row_vector) % n
    Um[Um == 0] = n
    return Um

# Function to calculate entropy using row products as probabilities
# def calculate_entropy(row_products):
#     # Normalize row products to obtain probabilities
#     probabilities = row_products / np.sum(row_products)

#     # Calculate entropy using Shannon's entropy formula: -sum(p * log(p))
#     entropy = -np.sum(probabilities * np.log(probabilities))

#     return entropy
# Kernel Density Estimation (KDE) for data points (samples)
def kernel_density_estimation(data, bandwidth='scott'):
    # Step 1: Fit KDE on the entire dataset (multidimensional KDE if needed)
    kde_function = gaussian_kde(data.T, bw_method=bandwidth)  # Transpose to fit KDE

    # Step 2: Evaluate KDE values for the input data points
    kde_values = kde_function(data.T)

    # Step 3: Sort the data based on KDE values
    # sorted_indices = np.argsort(kde_values)
    # sorted_data = data[sorted_indices]
    # sorted_kde_values = kde_values[sorted_indices]

    # Return the sorted KDE values, sorted data, and KDE function
    return kde_values, kde_function
# KL divergence function using KDE values
def kl_divergence(p):
    n = len(p)
    return (1/n) * np.sum(np.log(p))

# Main function
def main():
    # We store the configurations as a list of dictionaries for each 'n' value
    n_configurations = {
        29: [
            {"s": 4, "row_vector": np.array([0, 1, 1, 21]), "h1": np.array([1, 4, 6, 16])},
            {"s": 4, "row_vector": np.array([28, 28, 28, 28]), "h1": np.array([1, 4, 6, 16])},
            {"s": 5, "row_vector": np.array([0, 17, 26, 3, 27]), "h1": np.array([1, 4, 6, 7, 17])},
            {"s": 5, "row_vector": np.array([28, 28, 28, 28, 28]), "h1": np.array([1, 4, 6, 7, 17])},
            {"s": 15, "row_vector": np.array([23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23]), "h1": np.array([1, 2, 3, 4, 6, 7, 8, 9, 12, 14, 16, 18, 19, 24, 28])},
            {"s": 15, "row_vector": np.array([0, 13, 16, 10, 22, 9, 12, 21, 18, 28, 12, 4, 8, 14, 7]), "h1": np.array([1, 2, 3, 4, 6, 7, 8, 9, 12, 14, 16, 18, 19, 24, 28])}
        ],
        13: [
            {"s": 5, "row_vector": np.array([0, 7, 9, 8, 7]), "h1": np.array([1, 6, 8, 9, 10])},
            {"s": 5, "row_vector": np.array([0, 6, 11, 9, 11]), "h1": np.array([1, 6, 8, 9, 10])}
        ],
        # 15: [
        #     {"s": 5, "row_vector": np.array([0, 14, 7, 14, 2]), "h1": np.array([1, 2, 4, 7, 13])}
        # ],
        17: [
            {"s": 8, "row_vector": np.array([0, 1, 12, 1, 5, 12, 2, 5]), "h1": np.array([1, 3, 5, 9, 10, 11, 13, 15])},
            {"s": 8, "row_vector": np.array([16, 16, 16, 16, 16, 16, 16, 16]), "h1": np.array([1, 3, 5, 9, 10, 11, 13, 15])}
        ],
        31: [
            {"s": 8, "row_vector": np.array([7, 7, 7, 7, 7, 7, 7, 7]), "h1": np.array([1, 5, 7, 8, 10, 14, 16, 19])},
            {"s": 8, "row_vector": np.array([0, 3, 25, 26, 9, 13, 18, 14]), "h1": np.array([1, 5, 7, 8, 10, 14, 16, 19])},
            {"s": 15, "row_vector": np.array([7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7]), "h1": np.array([1, 2, 4, 5, 7, 8, 9, 10, 14, 16, 18, 19, 20, 25, 28])},
            {"s": 15, "row_vector": np.array([22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22]),"h1": np.array([1, 2, 4, 5, 7, 8, 9, 10, 14, 16, 18, 19, 20, 25, 28])}
        ],
        25: [
            {"s": 9, "row_vector": np.array([0, 17, 21, 11, 9, 7, 4, 14, 18]), "h1": np.array([1, 2, 3, 4, 6, 9, 11, 12, 18])},
            {"s": 9, "row_vector": np.array([0, 14, 5, 17, 5, 23, 5, 20, 12]), "h1": np.array([1, 2, 3, 4, 6, 9, 11, 12, 18])}
        ],
        27: [
            {"s": 10, "row_vector": np.array([0, 9, 21, 17, 24, 21, 22, 9, 21, 24]), "h1": np.array([1, 2, 4, 5, 8, 10, 13, 16, 20, 26])},
            {"s": 10, "row_vector": np.array([0, 23, 14, 8, 19, 24, 9, 11, 10, 25]), "h1": np.array([1, 2, 4, 5, 8, 10, 13, 16, 20, 26])}
        ],
        1069: [
            {"s": 5, "row_vector": np.array([557, 557, 557, 557, 557]), "h1": np.array([1, 63, 762, 970, 177])},
            {"s": 5, "row_vector": np.array([549, 549, 549, 549, 549]), "h1": np.array([1, 63, 762, 970, 177])},
            {"s": 5, "row_vector": np.array([1068, 1068, 1068, 1068, 1068]), "h1": np.array([1, 63, 762, 970, 177])}
        ],
        3997: [
            {"s": 8, "row_vector": np.array([0, 2896, 272, 2747, 2776, 2399, 3598, 516]), "h1": np.array([1, 3888, 3564, 3034, 2311, 1417, 375, 3211])},
            {"s": 8, "row_vector": np.array([0, 2888, 3448, 685, 555, 2157, 560, 3164]), "h1": np.array([1, 3888, 3564, 3034, 2311, 1417, 375, 3211])},
            {"s": 8, "row_vector": np.array([3085, 3085, 3085, 3996, 3085, 3085, 3085, 3085]),"h1": np.array([1, 3888, 3564, 3034, 2311, 1417, 375, 3211])},
            {"s": 8, "row_vector": np.array([3996, 3996, 3996, 3996, 3996, 3996, 3996, 3996]),"h1": np.array([1, 3888, 3564, 3034, 2311, 1417, 375, 3211])}
        ],
        4661: [
            {"s": 10, "row_vector": np.array([0, 4647, 3322, 1819, 3264, 2552, 3289, 4447, 3926, 2730]), "h1": np.array([1, 4574, 4315, 3889, 3304, 2570, 1702, 715, 4289, 3122])},
            {"s": 10, "row_vector": np.array([4660, 4660, 4660, 4660, 4660, 4660, 4660, 4660, 4660, 4660]), "h1": np.array([1, 4574, 4315, 3889, 3304, 2570, 1702, 715, 4289, 3122])},
            {"s": 10, "row_vector": np.array([0, 4341, 3652, 1461, 1551, 3665, 3348, 3082, 167, 3343]), "h1": np.array([1, 4574, 4315, 3889, 3304, 2570, 1702, 715, 4289, 3122])},
            {"s": 10, "row_vector": np.array([0, 3781, 3492, 598, 4249, 3434, 3656, 1060, 2896, 2627]), "h1": np.array([1, 4574, 4315, 3889, 3304, 2570, 1702, 715, 4289, 3122])}
        ]
    }
    # all_results = []
    kl_results = []  # Initialize the list for storing KL results
    i = 0
    for n, configurations in n_configurations.items():
        for config in configurations:
            i += 1
            print(i)
            row_vector = config["row_vector"]
            h1 = config["h1"]
            s = config["s"]

            # Generate GLP and GGLP samples using s for each n
            GLP0, _ = GLP(n, s, h1)
            GLP1 = (2 * GLP0 - 1) / (2 * n)

            GGLP0 = GGLP(GLP0, row_vector, n)
            GGLP1 = (2 * GGLP0 - 1) / (2 * n)

            # Calculate KDE for GLP and GGLP
            kde_glp_values, _ = kernel_density_estimation(GLP1)
            kde_gglp_values, _ = kernel_density_estimation(GGLP1)

            # Calculate KL divergence using KDE values (P = kde_values, Q = 1)
            kl_glp = kl_divergence(kde_glp_values)  # Compare GLP with uniform distribution
            kl_gglp = kl_divergence(kde_gglp_values)  # Compare GGLP with uniform distribution

            # Store KL results for saving
            kl_results.append({
                "n": n,
                "s": s,
                "KL Divergence GLP": kl_glp,
                "KL Divergence GGLP": kl_gglp
            })

            # # Prepare the data for saving
            # for i in range(sorted_glp.shape[0]):
            #     all_results.append({
            #         "n": n,
            #         "Type": "GLP",
            #         "Row": sorted_glp[i],
            #         "KDE": kde_glp_values[i],  # Now it's KDE
            #         "KL Divergence": kl_glp
            #     })

            # for i in range(sorted_gglp.shape[0]):
            #     all_results.append({
            #         "n": n,
            #         "Type": "GGLP",
            #         "Row": sorted_gglp[i],
            #         "KDE": kde_gglp_values[i],  # Now it's KDE
            #         "KL Divergence": kl_gglp
            #     })

    # # Convert the results into a DataFrame
    # df = pd.DataFrame(all_results)

    # # Expand the Row column to multiple columns to represent the individual elements
    # row_df = pd.DataFrame(df['Row'].tolist(), index=df.index)
    # df = df.drop(columns=['Row']).join(row_df)

    # Save KL divergence results to Excel
    kl_df = pd.DataFrame(kl_results)
    kl_excel_file = 'KL_Divergence_Results_KDE.xlsx'
    kl_df.to_excel(kl_excel_file, index=False)
    print(f"KL Divergence results saved to {kl_excel_file}")


if __name__ == "__main__":
    main()