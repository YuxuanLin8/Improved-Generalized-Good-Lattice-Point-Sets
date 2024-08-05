# -*- coding: utf-8 -*-
"""
Created on Fri May 24 16:46:38 2024

@author: HP
"""

import time 
import pandas as pd
from TAGLP import generate_Um, Algorithm1, MD, lp_distance, WDvar, Frobenius_Distance



# For saving the results
def save_initial_data(Um, generator):
    # Save the Um matrix to a CSV file
    pd.DataFrame(Um).to_csv('Um_matrix_appendix.csv', index=False, header=False)
    # Save the generator to a CSV file
    pd.DataFrame(generator).to_csv('generator_appendix.csv', index=False, header=['Generator'])
def save_algorithm_results(U, rankU, md, best_us):
    # Open the file for writing
    # with open('Algorithm1_matrices_appendix.csv', 'w') as f:
    #     for i, matrix in enumerate(U):
    #         # Write a header line indicating the current matrix number
    #         f.write(f"Matrix U{i+1}\n")
    #         # Use pandas to_csv to write the matrix to the file in append mode
    #         pd.DataFrame(matrix).to_csv(f, index=False, header=False)
    #         # Write the corresponding u vector used for this matrix
    #         f.write(f"u vector: {best_us[i][0]}\n")
    #         # Write a blank line as a separator only between matrices
    #         f.write("\n")
    pd.DataFrame(U).to_csv('U_matrix_appendix.csv', index=False, header=False)
    # Save other results to another CSV file
    result_data = pd.DataFrame({
        'RankU': rankU,
        'MD': md,  # Save the MD value
        'uVector': str(best_us)
            #[str(u[0]) for u in best_us]  # Save the u vectors as a string
    })
    result_data.to_csv('Algorithm1_results_appendix.csv', index=False)
# main function
def main():
    

    # Example usage:
    # h-values according to the appendix A.2 pp278-280 in Wang and Fang's book
    # the generating vectors h_values are calculated by KH Yuan wrt MSE
    # for s=3
    h_values_3 = {
        5: [1, 2, 4],
        7: [1, 2, 4],
        9: [1, 2, 4],
        11: [1, 3, 5],
        13: [1, 3, 9],
        15: [1, 2, 7],
        17: [1, 3, 9],
        19: [1, 3, 9],
        21: [1, 4, 10],
        23: [1, 15, 18],
        25: [1, 8, 14],
        27: [1, 20, 22],
        29: [1, 16, 24],
        31: [1, 11, 28]
    }
    # for s=4
    h_values_4 = {
        7: [1, 2, 3, 6],
        9: [1, 2, 4, 7],
        11: [1, 2, 5, 7],
        13: [1, 6, 8, 10],
        15: [1, 2, 4, 8],
        17: [1, 2, 4, 8],
        19: [1, 5, 7, 9],
        21: [1, 2, 10, 17],
        23: [1, 2, 5, 10],
        25: [1, 4, 6, 9],
        27: [1, 5, 17, 25],
        29: [1, 4, 6, 16],
        31: [1, 15, 19, 22], 
        
        307: [1, 42, 229, 101], 
        562: [1, 53, 89, 221], 
        701: [1, 82, 415, 382], 
        1019: [1, 71, 765, 865],
        2129: [1, 766, 1281, 1906],
        3001: [1, 174, 266, 1269]
    }
    # for s=5
    h_values_5 = {
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
        8191: [1, 1386, 4302, 7715, 3735],
        10007: [1, 198, 9183, 6967, 8507],
        15019: [1, 10641, 2640, 6710, 784],
        20039: [1, 11327, 11251, 12076, 18677],
        33139: [1, 32133, 17866, 21281, 32247],
        51097: [1, 44672, 45346, 7044, 14242],
        71053: [1, 33755, 65170, 12740, 6878],
        100063: [1, 90036, 77477, 27253, 6222],
        374181: [1, 343867, 255381, 310881, 115892]
    }
    # for s=6
    h_values_6 = {
        11: [1, 2, 4, 5, 8, 10],
        13: [1, 2, 3, 4, 6, 8],
        17: [1, 3, 5, 9, 10, 13],
        19: [1, 4, 5, 6, 7, 17],
        21: [1, 2, 4, 8, 11, 16],
        23: [1, 5, 6, 8, 13, 20],
        25: [1, 8, 12, 14, 18, 21],
        27: [1, 2, 4, 5, 8, 16],
        29: [1, 14, 18, 19, 20, 22],
        31: [1, 6, 9, 11, 28, 29]
    }
    # for s=7
    h_values_7 = {
        13: [1, 2, 3, 4, 6, 8, 12],
        17: [1, 3, 5, 9, 10, 13, 15],
        19: [1, 2, 4, 7, 8, 13, 16],
        21: [1, 2, 4, 5, 8, 10, 19],
        23: [1, 2, 4, 8, 9, 16, 18],
        25: [1, 8, 12, 14, 18, 19, 21],
        27: [1, 2, 4, 5, 8, 10, 16],
        29: [1, 5, 14, 18, 19, 20, 22],
        31: [1, 5, 7, 8, 10, 14, 16],
        
        3997: [1, 3888, 3564, 3034, 2311, 1417, 375],
        11215: [1, 10909, 10000, 8512, 6485, 3976, 1053],
        15019: [1, 12439, 2983, 8807, 7041, 7210, 6741]
    }
    # for s=8
    h_values_8 = {
        17: [1, 3, 5, 9, 10, 11, 13, 15],
        19: [1, 2, 4, 7, 8, 13, 14, 16],
        23: [1, 2, 4, 8, 9, 13, 16, 18],
        25: [1, 2, 3, 4, 6, 9, 12, 18],
        27: [1, 2, 4, 5, 8, 10, 16, 20],
        29: [1, 2, 3, 4, 6, 8, 12, 16],
        31: [1, 5, 7, 8, 10, 14, 16, 19],
        
        3997: [1, 3888, 3564, 3034, 2311, 1417, 375, 3211]
    }
    # for s=9
    h_values_9 = {
        17: [1, 3, 5, 9, 10, 11, 13, 15, 16],
        19: [1, 2, 4, 7, 8, 9, 13, 14, 16],
        23: [1, 2, 3, 4, 8, 9, 13, 16, 18],
        25: [1, 2, 3, 4, 6, 9, 11, 12, 18],
        27: [1, 4, 7, 10, 13, 16, 19, 22, 25],
        29: [1, 2, 3, 4, 6, 8, 12, 16, 24],
        31: [1, 5, 7, 8, 10, 14, 16, 18, 19]
    }
    
    h_values_10 = {
        19: [1, 2, 4, 7, 8, 9, 13, 14, 16, 18],
        23: [1, 2, 3, 4, 6, 8, 9, 12, 13, 16],
        25: [1, 2, 3, 4, 6, 8, 9, 11, 12, 18],
        27: [1, 2, 4, 5, 8, 10, 13, 16, 20, 26],
        29: [1, 2, 3, 4, 6, 8, 12, 16, 19, 24],
        31: [1, 4, 5, 7, 8, 10, 14, 16, 18, 19],
        
        4661: [1, 4574, 4315, 3889, 3304, 2570, 1702, 715, 4289, 3122]
        }
    
    h_values_11 = {
        23: [1, 2, 3, 4, 6, 8, 9, 12, 13, 16, 18],
        25: [1, 2, 3, 4, 6, 8, 9, 11, 12, 18, 24],
        29: [1, 2, 3, 4, 6, 8, 9, 12, 16, 19, 24],
        31: [1, 4, 5, 6, 9, 11, 13, 19, 23, 28, 29],
        
        4661: [1, 4574, 4315, 3889, 3304, 2570, 1702, 715, 4289, 3122, 1897],
        13587: [1, 13334, 12579, 11337, 9631, 7492, 4961, 2084, 12502, 9100, 5529],
        24076: [1, 23628, 22290, 20090, 17066, 13276, 8790, 3692, 22153, 16125, 9797]
        }
    
    h_values_12 = {
        23: [1, 2, 4, 5, 8, 9, 10, 11, 16, 17, 20, 22],
        29: [1, 6, 11, 12, 13, 15, 19, 21, 22, 24, 25, 27],
        31: [1, 4, 5, 7, 8, 9, 10, 14, 16, 18, 19, 25]
        }
    
    h_values_13 = {
        29: [1, 2, 3, 4, 6, 7, 8, 9, 12, 16, 18, 19, 24],
        31: [1, 2, 4, 5, 7, 8, 10, 14, 16, 18, 19, 20, 25]
        }
    
    h_values_14 = {
        29: [1, 2, 3, 4, 6, 7, 8, 9, 12, 14, 16, 18, 19, 24],
        31: [1, 2, 4, 5, 7, 8, 10, 14, 16, 18, 19, 20, 25, 28]
        }
    
    h_values_15 = {
        29: [1, 2, 3, 4, 6, 7, 8, 9, 12, 14, 16, 18, 19, 24, 28],
        31: [1, 2, 4, 5, 7, 8, 9, 10, 14, 16, 18, 19, 20, 25, 28]
        }
    
    
    n = 24076
    s = 11
    K = 30000
    p = 2 # use the l2_distance
    a = 0.1
    c = 0.03
    I = 10
    constant = -(4/3)**s
    
    # K0 = n**(s-1)
    
    # if K0 < K:  # The while-loop in TA for unique u vector will be dead end
    #     K = int(K0/2)
    
    Um, generator = generate_Um(n, s, h_values=h_values_11)
    MD_start = time.time()
    mdUm = MD(Um)
    MD_end = time.time()
    
    L2_start = time.time()
    L2Um = lp_distance(Um, 2)
    L2_end = time.time()
    
    WD_start = time.time()
    wdUm = WDvar(Um) + constant
    WD_end = time.time()
    
    FD_start = time.time()
    fdUm = Frobenius_Distance(Um)
    FD_end = time.time()
    
    print('The initial GLP in appendix:', '\nMD = ', mdUm, '\nWD = ', wdUm, '\nFrobenius_Distance = ', fdUm, '\nMaxmin = ', L2Um, '\n')
    
    print('The runtime of MD once for n = ' + str(n), 's = ' + str(s), 'Time Elapse: ' + str(MD_end - MD_start) + ' in seconds')
    print('The runtime of Maxmin d2 once for n = ' + str(n), 's = ' + str(s), 'Time Elapse: ' + str(L2_end - L2_start) + ' in seconds')
    print('The runtime of WD once for n = ' + str(n), 's = ' + str(s), 'Time Elapse: ' + str(WD_end - WD_start) + ' in seconds')
    print('The runtime of Frobenius_Distance once for n = ' + str(n), 's = ' + str(s), 'Time Elapse: ' + str(FD_end - FD_start) + ' in seconds')
    print('The search limit K = ', K)
    if Um is None:
        print("Invalid parameters or no valid combinations can be generated.")
    
    else:
        # Applying algorithm1 wrt maximin
        #U, rankU, d2, best_u = Algorithm1(Um, K, a, c, I, False, False, p) 
        #print('\nMD value = ', MD(U), '\nWD value = ', WDvar(U) + constant, '\nFrobenius_Distance value = ', Frobenius_Distance(U), '\n', 'Maxmin distance = ', lp_distance(U, p), '\nbest_u = ', best_u)
        
        
        
        # Applying algorithm1 wrt Frobenius_Distance
        U, rankU, md, best_u = Algorithm1(Um, K, a, c, I, True, True, p) 
        print('\nMD value = ', MD(U), '\nWD value = ', WDvar(U) + constant, '\nFrobenius_Distance value = ', Frobenius_Distance(U), '\n', 'Maxmin distance = ', lp_distance(U, p), '\nbest_u = ', best_u)
        
        
        
        #save_initial_data(Um, generator) # Output the initial matrix and generate vectors
        #save_algorithm_results(U, rankU, md, best_u) # Output the optimal matrix and corresponding row vectors after transformation under the criteria
    
if __name__ == '__main__':
    main()