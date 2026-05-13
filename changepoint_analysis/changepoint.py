
import numpy as np



def CUSUM_mean(Y, threshold, variance):
    """
    Estimates points of change in mean in a time series.
    
    """

    N = len(Y)
    C_2 = np.zeros(N)
    C_max = 0
    tau = 0
    sum_tot = np.sum(Y)
    sum = 0

    for t in range(1, N):
        sum += Y[t]
        y_1_t = sum/t
        y_t_n = (sum_tot - sum)/(N - t)

        C_2[t] = t*(N-t) / (N) * (y_1_t - y_t_n)**2

        if C_2[t] > C_max:     # if a new maximum found
            C_max = C_2[t]      # update maximum
            tau = t             # update change point
        
    if C_max/variance > threshold:
        return tau, C_max, C_2          # changepiont above threshold
    else:
        return None, C_max, C_2         # no changepoint detected