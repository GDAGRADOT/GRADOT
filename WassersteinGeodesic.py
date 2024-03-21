import numpy as np
from scipy.spatial.distance import cdist
import warnings
import sys
sys.path.append('../')
warnings.filterwarnings("ignore")


DBL_MAX = np.finfo('float').max
DBL_MIN = np.finfo('float').min



def algo0(a, b, M, epsilon = .3, param='primal', max_iter=50):
    lamda = 1/epsilon
    n = M.shape[0]
    l_b = M.shape[1]
    K = np.zeros((n, l_b))
    K_til = np.zeros((n, l_b))
    for i in range(l_b):
        for j in range(n):
            tmp = np.exp(-lamda*M[j,i])
            K[j, i] = tmp
            tmp = tmp/a[j]
            if np.isinf(tmp) or np.isnan(tmp):
                K_til[j, i] = DBL_MAX
            else:
                K_til[j, i] = tmp
    it = 0
    u = np.ones(n)/n
    temp_v = np.zeros(l_b)
    while it < max_iter:
        for i in range(l_b):
            tmp = 0
            for j in range(n):
                tmp = tmp + K[j,i]*u[j]
            tmp = b[i]/tmp
            if np.isinf(tmp):
                temp_v[i] = DBL_MAX
            elif np.isnan(tmp):
                temp_v[i] = 0
            else:    
                temp_v[i] = tmp # check for zero
        for j in range(n):
            tmp = 0
            for i in range(l_b):
                tmp = tmp + K_til[j,i] * temp_v[i]
            if tmp < DBL_MIN:
                u[j] = DBL_MAX
            else:
                u[j] = 1/tmp # check for zero
        it = it + 1
    t = np.zeros((n, l_b))
    if param=='primal':
        for i in range(l_b):
            for j in range(n):
                t[j,i] = K[j,i] * u[j] * temp_v[i]
        return t
    else: # param=='dual'
        alpha = np.zeros(n)
        tmp = 0
        for j in range(n):
            if u[j]!=0:
                u[j] = np.log(u[j])
                tmp = tmp + u[j]
        tmp = tmp/(lamda*n)
        for j in range(n):
            alpha[j] = +(tmp - u[j]/lamda)
        return alpha
        



def algo1(X, Y, b, M, weight=None, max_iter=[10,50]):
    d,n = X.shape
    N = len(Y)
    # Initializing importance weights and weights of barycenter unless provided
    if weight is None:
        weight = np.repeat(1./N, N)
    a_hat = a_til = np.ones(n)/n
    t = t_0 = 1
    while t< max_iter[0]:
        beta = (t+1)/2
        a = (1-(1/beta))*a_hat+(1/beta)*a_til
        alpha_list = [algo0(a, b[i], M[i], param='dual', 
                            max_iter=max_iter[1]) for i in range(N)]
        alpha = [weight[i]*alpha_list[i] for i in range(N)]
        alpha = np.sum(alpha, axis=0)
        a_til_n = a_til * np.exp(-t_0*beta*alpha)
        # Solving potential numeric issues
        if np.sum(np.isinf(a_til_n)) == 1:
            a_til = np.zeros((n,))
            a_til[np.isinf(a_til_n)] = 1.
        elif np.all(a_til_n==0):
            a_til = np.ones((n,))/n
        else:
            a_til = a_til_n/a_til_n.sum()
        a_hat = (1-1/beta)*a_hat + a_til/beta
        if np.any(np.isnan(a_hat)):
            print('Something is wrong in Algo1')
        t = t+1
    return a_hat




def algo2(Y, b, n, weight=None, max_iter=[5, 10, 50]):
    N = len(Y)
    #d = Y[0].shape[0]    
    # Initializing importance weights, atoms of barycenter and 
    # weights of barycenter unless provided
    if weight is None:
        weight = np.repeat(1./N, N)
    tmp_Y0=Y[0].T.copy()
    #np.random.shuffle(tmp_Y0)
    X=tmp_Y0.T[:,:n]
    a = np.ones(n)/n
    t = 1
    # Running optimization
    while t < max_iter[0]:
        teta = 1/3
        M = [cdist(X.T,Y[i].T, metric='sqeuclidean') for i in range(N)]
        a = algo1(X, Y, b, M, weight=weight, max_iter=max_iter[1:])
        T_list = [algo0(a, b[i], M[i], max_iter=max_iter[2]) for i in range(N)]
        g = [weight[i]*np.dot(Y[i],T_list[i].T) for i in range(N)]
        g = np.sum(g, axis=0)/a[None,:]
        X = (1-teta)*X + (teta)*g
        t = t+1
        if np.any(np.isnan(X)):
            print('Something is wrong in Algo2')
    return X.T, a