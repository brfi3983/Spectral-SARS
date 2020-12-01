import numpy as np
import numpy.random as rand

# A is the adjacency matrix as a numpy array.
# Vaccination not yet implemented

def begin_epidemic(A, vaccinate = False):
    n = A.shape[0]
    #find v_0
    return rand.randint(0, n-1)


def SIR(A, beta, mu, delta_t, T, v_0, vaccinated = False):
    # Initialize variables
    num_timesteps = T/delta_t
    q = mu * delta_t
    p = beta * delta_t

    susceptible_nodes = np.r_[0:(A.shape[0] - 1):1] # all nodes susceptible at first
    susceptible_nodes.remove(v_0)   # remove first infected

    infectious_nodes = [v_0]
    recovered_nodes = np.empty(0)

    # Main loop
    for t in (1,num_timesteps):
        for v in infectious_nodes:
            # Neighbors of v become infected with probability q
            for s in susceptible_nodes:
                if A[v,s].equals(1):
                    r = rand.uniform(0,1)
                    if r > p:
                        infectious_nodes.append(s)
                        susceptible_nodes.remove(s)
            r = rand.uniform(0,1)
            if r > q:
                recovered_nodes.append(v)
                infectious_nodes.remove(v)
    







