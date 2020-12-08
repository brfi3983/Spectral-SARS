import numpy as np
import numpy.random as rand

# A is the adjacency matrix as a numpy array.
# Vaccination not yet implemented
class SIR_class():
    def __init__(self, A):
        self.A = A
        self.n = A.shape[0]

    def remove_nodes(self, Y, X):

        # Y is array you will be removing from (S/I), X is array where the value has been moved to (I/R)
        ind = np.full(Y.shape[0], True)

        # If it is in the old array, change the mask to False so that it is not in the final output
        for i in range(Y.shape[0]):
            if Y[i] in X:
                ind[i] = False
        Y_new = Y[ind]

        return Y_new

    def SIR(self, beta, mu, delta_t, T, vaccinated=False):

        # Initialize variables
        num_timesteps = T/delta_t
        q = mu * delta_t
        p = beta * delta_t

        v_0 = rand.randint(0, self.n - 1)

        # Initializing Arrays
        S = np.arange(0,self.n)  # all nodes susceptible at first
        S = S[S != v_0]   # remove first infected
        I = np.array([v_0])
        R = np.empty(0)

        # Main loop
        for t in (1,num_timesteps):

            # For each infected node
            for v in I:

                for s in S:
                    # For each susceptible node that is a neighbor of the infected one
                    if self.A[v, s] == 1:

                        # Determining if node's neighbor is infected
                        r = rand.uniform(0,1)
                        if r < p:
                            I = np.append(I, s)

                # clean up old array
                S = self.remove_nodes(S, I)

                # Determining if each node recovers
                r = rand.uniform(0,1)
                if r < q:
                    R = np.append(R, v)

            # clean up old array
            I = self.remove_nodes(I, R)

        return [S, I, R]








