# -*- coding: utf-8 -*-
"""
Created on Fri May 10 15:57:46 2024

@author: HP
"""
import numpy as np
import random
import copy
import math
                 
def MD(x):
    # elements range from 1 to n
    # The input x is an n*s GLP set indicating q's of x are all equal to n.
    # The GLP set design matrix containing elements in {1, ... , n} needs to be linearly transformed into the unit hypercube C^s.
    n, s = np.shape(x)
    x = (x - 0.5) / n
    x0 = copy.deepcopy(x)
    x1 = abs(x0.T - 0.5)
    
    main1 = 5/3 - 0.25 * x1 - 0.25 * np.multiply(x1, x1)
    part1 = (2/n) * np.sum( np.prod(main1, axis=0) )
    part = np.zeros((n, 1))
    for i in range(n):
        xi = np.dot(np.ones((n, 1)), np.reshape(x0[i,:], (1, len(x0[i,:]) )))
        xi1 = abs(xi.T - 0.5)
        xm = abs(xi.T - x0.T)
        main2 = 15/8 - 0.25 * xi1 - 0.25 * x1 - 0.75 * xm + 0.5 * np.multiply(xm, xm)
        part[i] = np.sum(np.prod(main2, axis = 0))
    part2 = np.sum(part)
    # constant = (19/12)^s;
    # MDvar = -part1 + part2/(n^2);
    MD = (19/12)**s - part1 + part2 / (n**2)
    return MD  

def WDvar(x):
    # elements range from 1 to n
    # The input x is an n*s GLP set indicating q's of x are all equal to n.
    # The GLP set design matrix containing elements in {1, ... , n} needs to be linearly transformed into the unit hypercube C^s.
    n, s = np.shape(x)
    x = (x - 0.5) / n
    x0 = copy.deepcopy(x)
    
    part = np.zeros((n, 1))
    for i in range(n):
        xi = np.dot(np.ones((n, 1)), np.reshape(x0[i,:], (1, len(x0[i,:]) )))
        xw = abs(xi.T - x0.T)
        mainw = 3/2 - xw + np.multiply(xw, xw)
        part[i] = np.sum(np.prod(mainw, axis = 0))
    cumpart = np.sum(part)
    #constant = -(4/3)**s
    WDvar = cumpart / (n**2)
    return WDvar


def Frobenius_Distance(C):
    """
    Calculate the Frobenius distance between the covariance matrix of C
    and a scaled identity matrix I_s/12.
    """
    # Calculate the mean vector and center the matrix C
    mean_vector = np.mean(C, axis=0)
    C_centered = C - mean_vector

    # Calculate the covariance matrix of C (use unbiased estimates to calculate the covariance)
    covariance_matrix = np.cov(C_centered, rowvar=False, bias=False)

    # Identity matrix I_s scaled by 1/12
    s = C.shape[1]  # Number of dimensions
    I_s = np.identity(s) / 12

    # Calculate the Frobenius distance between the covariance matrix of C and I_s/12
    diff = covariance_matrix - I_s
    distance = np.sqrt(np.trace(diff.T @ diff))

    return distance
    

def neigh(x0, u = False):
    # Randomly generate a neighborhood of design x, 
    # that is randomly choosing one element of the current u to alternate except the first element
    # u = (0, u2, ..., us), ui in {0, 1, ..., n-1} is a row vector.
    
    #####Sample codes of TA
    # n, s = x.shape
    # xn = copy.deepcopy(x)
    # col = s - 1
    # rows = random.sample(list(range(n)), 2)
    # while xn[rows[0], col] == xn[rows[1],col]:
    #     rows = random.sample(list(range(n)), 2)
    # xn[rows[0],col], xn[rows[1],col] = xn[rows[1], col], xn[rows[0], col]
    
    ####### TAGLP alternates one randomly chosen element from u. xn = x0 + 1 * un (mod n), un is a neighborhood of the current u
    n, s = x0.shape
    if type(u) is np.ndarray:
        un = copy.deepcopy(u)
        
        # Keep the first element of u is unchanged.
        col = random.randint(1, s - 1)   # Randomly choose the element index [1, s-1]
        while un[0, col] == u[0, col]:    # Avoid inefficient update
            un[0, col] = random.choice(list(range(n)))
    else:
        un = [0] + [random.randint(0, n - 1) for i in range(s - 1)]
        un = np.array(un)
        un = np.reshape(un, (1, s))
    xn = np.mod( x0 + np.dot(np.ones([n, 1]), un), n )
    xn[xn == 0] = n
    return xn, un

def Tseq(x0, a, c, I, J, disc = True, p = 2):
    # Retreive a T sequence for given a, I, J and the initial GLP design.
    # a = 0.1, proportion of MD function values
    # I = 10, number of thresholds
    # J = 5000, number of searches
    # If disc = True, MD function is adopted. Otherwise, L2-distance is adopted.
    # c = 0.03 is a parameter controling the mixture diminishing procedure
    # When the first Ti <= cT0 occurs, the sequence cools down in a linear manner.
    indisc = []
    n, s = x0.shape
    MD = Frobenius_Distance
    for j in range(J):
        uc = [0] + random.sample(list(range(n)), s - 1)
        uc = np.array(uc)
        uc = np.reshape(uc, (1, s))
        xc = np.mod( x0 + np.dot(np.ones([n, 1]), uc), n )
        xc[xc == 0] = n
        if disc == True:    # MD is adopted.
            indisc.append(MD(xc))
        else:    # L2-distance is adopted.
            indisc.append(lp_distance(xc, p))
            ############################## Lp-distance
    T0 = (max(indisc) - min(indisc)) * a
    T = [T0]
    for i in range(1, I):
        if T[i-1] <= c * T0:
            T.append( (I-i-1)/(I-i) * T[i-1] )    # linearly diminishing
        else:
            T.append( ( (I-i-1)/I ) * T[i-1] )    # expponentially diminishing
    return T

def TAGLP(x0, a, c, I, J, disc = True, FD = True, p = 2):
    # Threshold Accepting algorithm
    # x0: initial design
    # a, c: parameters of threshold sequence T, a = 0.1, c = 0.03
    # I: number of thresholds
    # J: number of searchs
    # If disc = True, MD function is adopted. Otherwise, L2-distance is adopted.
    n, s = x0.shape
    #if FD == True:
        #MD = WDvar
        #constant = -(4/3)**s
    #else:
    #    MD = Frobenius_Distance
    constant = 0
    
    
    T = Tseq(x0, a, c, I, J, disc)
    count = 0
    
    indisc = []
    if disc == True:    # MD is adopted.
        if FD == True:
            indisc.append(Frobenius_Distance(x0))
        else:
            indisc.append(MD(x0))
    else:    # L2-distance is adopted.
        indisc.append(lp_distance(x0, p))
        ############################## Lp-distance
    
    
    x = x0
    u = False
    uniqueUs = set() # Ensure that the row vector u is unique
    xall = []
    xall.append(x0)
    L2UB = calculate_d_p(n, s, p)
    
    for i in range(I):
        # xn, u = neigh(x0, u)
        # xneigh = np.array([xn])
        # mdn = []
        # mdn.append(MD(xn))
        # for t in range(1, N):
        #     xn, un = neigh(x0, u)
        #     xneigh = np.vstack((xneigh, np.array([xn])))
        #     mdn.append(MD(xn))
        # xn_opt_MD = min(mdn)
        # loc = mdn.index(xn_opt_MD)
        
        # x = xneigh[loc]
        # disc.append(xn_opt_MD)
        # xall.append(x)
        for j in range(J):
            count += 1
#            print(count)
            xn, ucand = neigh(x0, u)
            strU = str(ucand.tolist())
            while strU in uniqueUs:    # if encountering identical u, resample a neighborhood
                xn, ucand = neigh(x0, u)
                strU = str(ucand.tolist())
            u = ucand
            uniqueUs.add(strU)
            
            if disc == True:    # MD is adopted.
                if FD == True:
                    mdxn = Frobenius_Distance(xn)
                else:
                    mdxn = MD(xn)
                if mdxn - indisc[-1] <= T[i]:    # Accept the neighborhood as the current solution.
                    x = xn
                    indisc.append(mdxn)
                    xall.append(x)
            else:    # L2-distance is adopted.
                mdxn = lp_distance(xn, p)

                if indisc[-1] - mdxn <= T[i]:    # Accept the neighborhood as the current solution.
                    x = xn
                    indisc.append(mdxn)
                    xall.append(x)
                
                if mdxn == L2UB:
                    #The current matrix is optimal wrt L2-distance.
                    mdnew = mdxn
                    print('Found optimal GGLP wrt L2-distance, achieving the UB.')
                    print('Upper bound of L2-distance', L2UB)
                    print(x)
                    return x, u, mdnew
            
        if disc == True:    # MD is adopted.
            mdnew = min(indisc)
            print('Discrepancy difference with the initial GLP', mdnew + constant - indisc[0])
        else:    # L2-distance is adopted.
            mdnew = max(indisc)
            # print('L2-distance difference with the upper bound', L2UB - mdnew)
            # print('L2-distance difference with the initial GLP', mdnew - indisc[0])
            ############################## Lp-distance
            
        optloc = indisc.index(mdnew)
        x = xall[optloc]
    
    return x, u, mdnew + constant

# From TY Yan with modification of YX Lin
# Caculate the lp_distance
def lp_distance(D, p):
    """
    Compute the Lp distance (LD) for matrix D where each row is compared to every other row
    using vectorized operations to improve efficiency.
    """
    N = D.shape[0]  # Number of runs
    minDist = np.inf  # Initialize minimum distance with infinity

    # Ensure D is of floating point type to handle inf and floating point operations
    D = D.astype(np.float64)

    # Loop over each row and calculate the lp distance to every other row
    for i in range(N):
        # Create a matrix where every row is the ith row of D
        D_i = np.tile(D[i, :], (N, 1))

        # Calculate the element-wise p-th power of the absolute difference
        dist_matrix = np.abs(D - D_i) ** p

        # Sum over columns to get the Lp distance for each pair of rows
        row_distances = np.sum(dist_matrix, axis=1, dtype=np.float64)

        # We do not consider the distance of the row with itself, which will be zero
        row_distances[i] = np.inf  # Set the self-distance to infinity to exclude it

        # Find the minimum distance in this iteration
        current_min = np.min(row_distances)

        # Update the global minimum distance if the current minimum is smaller
        if current_min < minDist:
            minDist = current_min

    return minDist
# The upper bound of lp_distance
def calculate_d_p(n, s, p):
    """
    Calculate the upper bound of d_p(D) for a U-type (n, n^s) design D.
    """
    if p == 1:
        result = (n + 1) * s / 3
    elif p == 2:
        result =  n * (n + 1) * s / 6
    else:
        raise ValueError("Unsupported value of p; p should be either 1 or 2.")

    return math.floor(result)

# Function to generate Um matrix as initial
def generate_Um(n, s, generator=None, h_values=None):
    nvec = np.arange(1, n + 1)

    # Use the provided generator if available
    if generator is None:
        # If generator is not provided, generate it based on n
        generator = h_values.get(n, None)
        if generator is None:
            print('Please check h_values_s includes the current n as a key.')
            return None, None
            # Incorporate PGLP GLP with power generator
            # generator = nvec[np.fromiter((math.gcd(i, n) == 1 for i in nvec), dtype=bool)]

    m = len(generator)

    if s > m:
        return None, None, None

    ih = np.kron(nvec, generator)
    Um = ih % n
    Um[Um == 0] = n
    Um = Um.reshape(len(nvec), len(generator))
    return Um, generator
# Algorithm1 function with TA assist
def Algorithm1(Um, K, a, c, I, disc = True, FD = True, p = 2):
    n, s = Um.shape
    #if FD == True:
    #     MD = WDvar 
    #     constant = -(4/3)**s
    # else:
    #    MD = Frobenius_Distance
    constant = 0
    
    upper_bound = calculate_d_p(n, s, p)
    Ucandidate = []
    mds = []
    if disc == True:    # MD is adopted.
        benchmark = [np.inf, 0]
    else:    # L2-distance is adopted.
        benchmark = [-np.inf, 0]
    us = []  # List to store all used u vectors
    idx = 0    # Count the number of current steps
    # N-1 prior searches: Select a row vector u = (i, i, ..., i) in the size of s, i = 1, ..., n-1
    for j in range(1, n): # Ensure that the value of each component is 1 to n-1
        u = np.array([j] * s)
        u = np.reshape(u, (1, s))
        U_temp = (Um + np.dot(np.ones((n, 1)), u)) % n
        U_temp[U_temp == 0] = n
        Ucandidate.append(U_temp)
        if disc == True:    # MD is adopted.
            if FD == True:
                md = Frobenius_Distance(U_temp)
            else:
                md = MD(U_temp)
        else:    # L2-distance is adopted.
            md = lp_distance(U_temp, p)
            mds.append(md)
        us.append((u, md))  # Store u vector with its mds value
        if disc == False:    # L2-distance is adopted.
            if md >= upper_bound:  # Check if reached the upper bound
                print(f"Stopping early, reached upper bound at i = {j}")
                return U_temp, np.linalg.matrix_rank(U_temp), md, u
            elif md >= benchmark[0]:
                benchmark = [md, u]    # Record the best solution
        elif md <= benchmark[0]:    # MD is adopted and the current solution is better than the record
            benchmark = [md, u]
        idx += 1
        
        if idx >= K:
            break
    
    
    benchmark[0] = benchmark[0] + constant
    # TA searches
    J = int((K - n + 1)/I)
    if J > 0:
        x, u, mdnew = TAGLP(Um, a, c, I, J, disc, FD, p)
    
    # Output competes between TA and n-1 prior searches
    if disc == False:    # L2-distance is adopted.
        if mdnew > benchmark[0]:    # TA is conducted only if the upper bound is not achieved in the previous n-1 searches
            print('TA beats n-1 prior searches')
            if mdnew >= upper_bound:
                print('TA achieves the upper bound')
                return x, np.linalg.matrix_rank(x), mdnew, u
            else:
                print('TA does not achieve the upper bound')
                return x, np.linalg.matrix_rank(x), mdnew, u
        else:
            print('TA is no better than the n-1 prior searches with maxmin distance = ', mdnew)
            print('Benchmark maxmin distance of n-1 searches = ', benchmark[0], 'and the corresponding u = ', benchmark[1], '\n', 'the upper bound is', upper_bound)
            Ubenchmark = (Um + np.dot(np.ones((n, 1)), benchmark[1])) % n
            Ubenchmark[Ubenchmark == 0] = n
            return Ubenchmark, np.linalg.matrix_rank(Ubenchmark), benchmark[0], benchmark[1]
    elif mdnew < benchmark[0]:    # MD is adopted and the TA output is better than the n-1 prior searches
        print('TA beats n-1 prior searches')
        return x, np.linalg.matrix_rank(x), mdnew, u
    else:
        print('TA is no better than the n-1 prior searches with discrepancy value = ', mdnew)
        print('Benchmark discrepancy value of n-1 searches = ', benchmark[0], 'and the corresponding u = ', benchmark[1])
        Ubenchmark = (Um + np.dot(np.ones((n, 1)), benchmark[1])) % n
        Ubenchmark[Ubenchmark == 0] = n
        return Ubenchmark, np.linalg.matrix_rank(Ubenchmark), benchmark[0], benchmark[1]
    
    
    # # Randomly select K-(n-1) row vectors, ensuring that no two vectors are the same
    # uniqueUs = set() # Ensure that the row vector u is unique
    # while idx < K:
    #     u_tail = np.random.randint(0, n, size= s - 1)  # Generate s-1 random elements
    #     u = np.insert(u_tail, 0, 0)  # Ensure that the first element of the row vector is 0
    #     strU = str(u.tolist())
    #     if strU not in uniqueUs:
    #         uniqueUs.add(strU)
            
    #         U_temp = (Um + np.dot(np.ones((n, 1)), u)) % n
    #         U_temp[U_temp == 0] = n
    #         Ucandidate[idx] = U_temp
    #         mds[idx] = lp_distance(U_temp, p)
    #         us.append((u, mds[idx]))  # Store u vector with its mds value
    #         if mds[idx] >= upper_bound:  # Check if reached the upper bound
    #             print(f"Stopping early, reached upper bound at index {idx}")
    #             return Ucandidate[:idx+1], [np.linalg.matrix_rank(U) for U in Ucandidate[:idx+1]], mds[:idx+1], us[:idx+1]
    #         idx += 1

    #     if idx >= K:
    #         break

    # maxd = np.max(mds)
    # indices = np.where(mds == maxd)[0]
    # best_us = [us[i] for i in indices]  # Select u vectors corresponding to max mds
    # U = [Ucandidate[i] for i in indices]
    # rankU = [np.linalg.matrix_rank(U[i]) for i in range(len(indices))]
    # md = maxd

    #return U, rankU, md, best_us  # Return best u vectors alongside other results