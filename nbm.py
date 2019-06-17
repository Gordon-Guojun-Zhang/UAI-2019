import numpy as np
np.set_printoptions(precision=4, suppress=True)
import random
''' a library for computing Naive Bayes model, such as Bernoulli dist.
predictions, giving samplings, using gradient descent, computing ells'''

# Bernoulli distribution, return y^x*(1-y)^{1-x}
def bern(x, y):
    return y if x == 1 else 1 - y

def bern_n(x, y, n):  # compute vector version B(x|y)
    prod = 1.0
    for _ in range(n):
        prod *= bern(x[_], y[_])
    return prod

def helper_bernoulli(n, c, Phi, x):    # n independent bernoulli's 
    ct = 0   # the number of zero terms
    prod = 1.0
    for k in range(n):
        single = bern(x[k], Phi[c, k])
        if single == 0:
            ct += 1
        if ct == 1:
            prod *= 1
            j = k    # the missing index
            continue
        if ct == 2:     # at least two zero's
            return [0.0, -1]       # j = -1 means that there are two 0's
        prod *= single
    if ct == 1:
        return [prod, j]   # only one zero
    else:
        return [prod, -2]         # j = -2 means that there are no 0's

def bernoulli(prod, j):
    ''' takes prod j, the output of helper_bernoulli'''
    return 0.0 if j >= -1 else prod

def new_bern(n, c, Phi, x):
    prod = 1.0
    for k in range(n):
        single = bern(x[k], Phi[c, k])
        prod *= single
    return prod

def covariance(Phi, c):
    n = len(Phi[0, :])
    Sigma = np.diag(np.multiply(Phi[c, :], np.ones(n) - Phi[c, :]))
    return Sigma


def randInt(m, n):    # randomly initialize theta and Phi
    theta = np.random.rand(m)
    theta /= np.sum(theta)
    Phi = np.random.rand(m, n)
    return [theta, Phi]

def bernoulliJ(prod, j, c, k, Phi, x): # N independent bernoulli's w/o j
# also uses bernoulli(n, c, Phi, x) as an auxillary
    if j == -1:
        return 0.0
    elif j == -2:
        return prod / bern(x[k], Phi[c, k])
    else:
        return prod if j == k else 0.0


# naive bayes model
def nbm(n, Phi, theta, x): # P_O(x) = sum_c th_c prod_i bern(phi_c,i x_i)
    ''' theta = (th_1, th_2, ..., th_m), n = number of features
        Phi: m x n, m is the number of classes
        x: a bit string of size n''' 
    m = len(theta)
    total = 0
    for c in range(m):
        [prod, j] = helper_bernoulli(n, c, Phi, x)
        total += theta[c] * bernoulli(prod, j)
    return total

def ell(n, Phi, theta, S, samples):
    ''' compute the ELL given the samples '''
    ell = 0
    for t in S:
        ct = S[t]
        x = decToBits(t, n)
        ell -= ct * np.log(nbm(n, Phi, theta, x) + 1e-30)
    ell /= samples
    return ell


def sampling(n, Phi, theta):
    x = list(range(n))
    m = len(theta)
    eps = np.random.uniform(0, 1)
    total = 0
    for i in range(m):
        total += theta[i]
        if total > eps:
            choice = i
            break
    for j in range(n):
        eps = np.random.uniform(0, 1)
        x[j] = 1 if eps < Phi[choice, j] else 0
    return x


def decToBits(t, n):   
    # transform a decimal t to a bit array of size n
    a = []
    cur = t
    index = 0
    while index != n:
        a.append(cur % 2)
        cur //= 2
        index += 1
    return list(reversed(a))

def bitsToDec(x):   # x is a list, turn it into a number
    n = len(x)
    number = 0
    for i in range(n):
        number += 2**(n - i - 1) * x[i]
    return number

def sample_size(n, S):
    ''' return the sample size of the dictionary S '''
    total = 0
    for t in S:
        total += S[t]
    return total

def sample_gen(gtheta, gPhi, samples, n):
    ''' generate a sample set based on gtheta and gPhi'''
    S = dict()
    for _ in range(samples):
        x = sampling(n, gPhi, gtheta)
        t = bitsToDec(x)
        if t not in S:
            S[t] = 0
        S[t] += 1
    return S

def project(theta):   
    ''' project theta onto the simplex, see eg. https://web.stanford.edu/~jduchi/projects/DuchiShSiCh08.pdf, use the simplist one for now'''
    m = len(theta)
    v = np.flip(np.sort(theta), axis = 0)
    total = 0
    for j in range(m):
        total += v[j]
        if (j + 1) * v[j] - total + 1 > 0:
            continue
        else:
            break
    if (j + 1) * v[j] - total + 1 <= 0:
        rho = j
        total -= v[j]
    else:
        rho = m
    shift = (total - 1) / rho
    w = np.zeros(m)
    for _ in range(m):
        w[_] = max(theta[_] - shift, 0)
    return w


def eff_num_pur(n, theta, Phi, s_size, S, cluster):
    ''' returns gamma_c as defined in the UAI submission'''
    m = len(theta)
    N_c = 0.0
    for t in S:
        ct = S[t]
        x = decToBits(t, n)
        Bern = np.zeros(m)  # B(x|mu_c)
        for c in range(m):
            [prod, index] = helper_bernoulli(n, c, Phi, x)
            Bern[c] = bernoulli(prod, index)
        P_omega = np.dot(theta, Bern) 
        resp = (1 / P_omega) * Bern[cluster]
        N_c += resp * ct
    return N_c

def eff_num(n, theta, Phi, s_size, S, cluster):
    ''' returns N_c'''
    return theta[cluster] * eff_num_pur(n, theta, Phi, s_size, S, cluster)

def eff_mean(n, theta, Phi, s_size, S, cluster):
    ''' returns overline x_c '''
    m = len(theta)
    x_c = np.zeros(n)
    for t in S:
        ct = S[t]
        x = decToBits(t, n)
        Bern = np.zeros(m)  # the increment of Phi_{c, j}
        for c in range(m):
            [prod, index] = helper_bernoulli(n, c, Phi, x)
            Bern[c] = bernoulli(prod, index)
        Px = np.dot(theta, Bern)  # regularize
        x_c += np.multiply((ct / Px) * Bern[cluster] * theta[cluster], x)
    N_c = eff_num(n, theta, Phi, s_size, S, cluster)
    x_c = np.divide(x_c, N_c)
    return x_c

def gradient(n, theta, Phi, samples, S):
    ''' compute the gradients'''
    m = len(theta)
    grad = np.zeros_like(Phi)
    theta_inc = np.zeros(m)
    for t in S:
        ct = S[t]
        x = decToBits(t, n)
        Bern = np.zeros(m)  # the increment of Phi_{c, j}
        Phi_inc = np.zeros_like(Phi)
        for c in range(m):
            [prod, index] = helper_bernoulli(n, c, Phi, x)
            Bern[c] = bernoulli(prod, index)
            for j in range(n):
                Phi_inc[c, j] = theta[c] * ((-1)**(1 - x[j])) * bernoulliJ(prod, index, c, j, Phi, x)
        P_omega = np.dot(theta, Bern)
        Phi_inc = ct * np.divide(Phi_inc, P_omega)
        grad -= Phi_inc
    for c in range(m):
        theta_inc[c] = eff_num_pur(n, theta, Phi, samples, S, c)
    return [-theta_inc/samples, grad/samples]


def gradientDescent(n, sample_size, S, theta, Phi, numIterations, alpha, tolerance=1e-6):
    ''' gradient descent for unsupervised naive bayes 
    model update in a batch '''
    m = len(theta)
    ll = 0
    iterId = 0
    for i in range(numIterations):
        iterId += 1
        oldll = ll
        # update the Likelihood for each iteration
        oldtheta = np.copy(theta)
        oldPhi = np.copy(Phi)
        [theta_inc, grad] = gradient(n, theta, Phi, sample_size, S)
        theta -= alpha * theta_inc
        theta = project(theta) 
        Phi -= alpha * grad
        # project phi_{c,j} to be in between [0, 1]
        for c in range(m):
            for j in range(n):
                ''' impose soft constraints on Phi'''
                Phi[c, j] = max(Phi[c, j], 0)
                Phi[c, j] = min(Phi[c, j], 1)
        diff1 = np.linalg.norm(oldtheta - theta)
        diff2 = np.linalg.norm(oldPhi - Phi)
        if diff1 < tolerance and diff2 < tolerance:
            print("GD converges at ", i, "steps")
            break
    ll = 0.0
    for t in S:
        ct = S[t]
        x = decToBits(t, n)
        ll -= ct * np.log(nbm(n, Phi, theta, x))
    ll /= sample_size
    return [ll, theta, Phi, iterId]

def em(n, m, samples, S, theta, Phi, numIterations=20000, tolerance=1e-5):
    ''' see PRML pg 446-447 '''
    theta1 = np.copy(theta)
    Phi1 = np.copy(Phi)
    for _ in range(numIterations):
        oldtheta = np.copy(theta1)
        oldPhi = np.copy(Phi1)
        effnum = np.zeros(m)
        for c in range(m):
            effnum[c] = eff_num(n, theta1, Phi1, samples, S, c)
            Phi1[c, :] = eff_mean(n, theta1, Phi1, samples, S, c)
        theta1 = np.divide(effnum, samples)
        if np.linalg.norm(oldtheta - theta1) < tolerance and np.linalg.norm(oldPhi - Phi1) < tolerance:
            break
#    print("EM converges at ", _, "steps")
#    print("ell of EM: ", ell(n, Phi1, theta1, S, samples))
    return [theta1, Phi1]

def bestEll(samples, S):
    ''' return - sum p_S log p_S '''
    optimal = 0
    for t in S:
        ct = S[t] / samples
        optimal += -ct * np.log(ct)
    return optimal

def zeroInfoEll(n, samples, S):
    ''' return - sum_i sum P_S(X_i) log P_S(X_i) '''
    p0 = np.zeros(n)  # an array: [P(X_i = 0)]
    # compute p0 and sum P_S log P_S
    for t in S:
        ct = S[t] / samples
        x = decToBits(t, n)
        for i in range(n):
            if x[i] == 0:
                p0[i] += ct
    zeroInfo = 0
    for i in range(n):
        if p0[i] == 0 or p0[i] == 1:
            continue
        else:
            zeroInfo += -p0[i]* np.log(p0[i])
            zeroInfo += -(1 - p0[i])* np.log((1 - p0[i]))
    return zeroInfo

def twoInfoElls(n, samples, S):
    # create all pairs
    pairs = dict()
    twoInfos = dict()
    for i in range(n):
        for j in range(i + 1, n):
            pairs[i, j] = np.zeros((2, 2))
            twoInfos[i, j] = 0.0
    p0 = np.zeros(n)  # an array: [P(X_i = 0)]

    # compute 2-info points
    for t in S:
        ct = S[t] / samples
        x = decToBits(t, n)
        for i in range(n):
            if x[i] == 0:
                p0[i] += ct
        for i in range(n):
            for j in range(i + 1, n):
                pairs[i, j][x[i], x[j]] += S[t] / samples
    
    # zero_info points
    zeroInfo = 0
    for i in range(n):
        if p0[i] == 0 or p0[i] == 1:
            continue
        else:
            zeroInfo += -p0[i]* np.log(p0[i])
            zeroInfo += -(1 - p0[i])* np.log((1 - p0[i]))

    # two_info points
    for i in range(n):
        for j in range(i + 1, n):
            m = pairs[i, j]
            twoInfos[i, j] = zeroInfo
            twoInfos[i, j] += two_ell(m) 
            twoInfos[i, j] -= -p0[i]* np.log(p0[i]) -(1 - p0[i])* np.log((1 - p0[i]))
            twoInfos[i, j] -= -p0[j]* np.log(p0[j]) -(1 - p0[j])* np.log((1 - p0[j]))
    return twoInfos.values()
            
