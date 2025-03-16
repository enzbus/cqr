import numpy as np
import scipy as sp

def equilibrate(
    A: sp.sparse.csc_matrix,
    b: np.array,
    c: np.array,
    P: sp.sparse.csc_matrix,
    lnorm=2,
    niter=25):
    
    n = len(c)
    m = len(b)

    full_mat = sp.sparse.block_array(
        [
            [P, A.T, c.reshape((n, 1))],
            [A, None, b.reshape((m, 1))],
            [c.reshape((1, n)), b.reshape((1, m)), None]
        ])
    # plt.imshow(full_mat.todense())
    # plt.colorbar()

    full_scaler = np.ones(m+n+1)

    for _ in range(niter):
        if lnorm == 2:
            norm = sp.sparse.linalg.norm(full_mat, axis=0)
        elif lnorm == np.inf:
            norm = full_mat.max(axis=0).todense().flatten()
        else:
            raise SyntaxError('l-norm not supported!')
        norm[norm == 0.] = 1.
        print(max(norm)/min(norm))
        scaler = norm ** (-0.5)
        full_scaler *= scaler
        full_mat = (full_mat * scaler).T * scaler

    e = full_scaler[:n]
    d = full_scaler[n:-1]
    sigma = full_scaler[-1]
    return e, d, sigma, full_mat
    # print(full_mat)


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # create data
    n = 20
    nonneg = 20
    zero = 20
    m = nonneg + zero

    A = sp.sparse.rand(m,n, density=.1)
    b = np.random.randn(m)
    c = np.random.randn(n)
    P = sp.sparse.rand(n,n, density=.05)
    P += P.T

    e, d, sigma, _ = equilibrate(
        A=A, b=b, c=c, P=P,
        lnorm=np.inf,
        niter=100,
        # lnorm=np.inf
        )
    
    

    # plt.figure()
    # plt.imshow(full_mat.todense())
    # plt.colorbar()
    # plt.show()