def sliced_wasserstein_barycenter(X1, X2, a=None, b=None, K=3, time_init=None, proj="normal",
                                  pos_reg=1, color_reg=1):
    
    if len(X1) < len(X2):
        X2, X1 = X1, X2 #  We always have len(X1) > len(X2).
        swaped = True
    else:
        swaped = False
        
    m = len(X1)
    n = len(X2)
#     print(n,m)
    if a is None:
        a = np.ones(m) / m
    if b is None:
        b = np.ones(n) / n
        
    # Permutation to apply in order to know the mapping.
    # Each point is send to 2 location per projection k.
    perm = np.zeros((2, K, m), dtype=int)
    weights = np.zeros((2, m))
    
    ratio = n/m
    
    # fix array that depend on the value of n and m. We are only interested at the order and at the
    # number of point that should be send from one distribution to the other.
    # n = 2, m = 5 give [0, 0, 0, 1, 1] for the perm_global and [., ., 1, ., . ] for 2
    # All the masse of the the first 2 points are send to the points 0.
    # and only a fraction (computed in weights) of the 3rd points are send to 0 because perm_global_2
    # has a value.
    # In fact perm_global_2 have some value at every position. But it is associated with 0 in weights[1].
    
    # I recomande to take example (n = 7, m=16 for instance) to understand all this function.
    arange_1 = np.int_(np.arange(1, n+1, ratio)) / n
    arange_2 = np.arange(1, m + 1) / m
    perm_global = np.int_(np.arange(0, n, ratio))
    perm_global_2 = np.minimum(perm_global + 1, n - 1)
    
    a1_a2 = arange_1 - arange_2
    # weights[0, i] associated to the transport of the points i in X1 to the points X2 define in perm_global
    # weights[1, i] associated to the transport of the points i in X1 to the points X2 define in perm_global_2
    weights[0] = (1/m) * (0 <= a1_a2) + (1/m + a1_a2) * (0 > a1_a2)
    weights[1] = np.abs(a1_a2) * (0 > a1_a2)

    X1_return = np.array([])
    X2_return = np.array([])
    for k in range(K):
        if proj == "normal":
            projector = np.random.rand(X1.shape[1])
            if color_reg is not None:
                projector[2:] *= color_reg
            x1 = X1 @ projector
            x2 = X2 @ projector
        elif proj == "middle":
            projector = ((np.random.rand(X1.shape[1]) - 0.5) * 2)
            if color_reg is not None:
                projector[2:] *= color_reg
            x1 = (X1 - X1.mean(axis=0, keepdims=True))@ projector
            x2 = (X2 - X2.mean(axis=0, keepdims=True))@ projector
        elif proj == "rdm":
            projector = ((np.random.rand(X1.shape[1]) - 0.5) * 2)
            if color_reg is not None:
                projector[2:] *= color_reg
            b = np.random.rand(1, X1.shape[1])
            x1 = (X1 / X1.max() - b) @ projector
            x2 = (X2 / X2.max() - b) @ projector
#         elif proj == "middle":
#             projector = np.random.rand(5)
#             x1 = X1 @ projector
#             x2 = X2 @ projector
        
        
        projector = projector / np.sum(projector) #  Not necessary usefull
#         print(x1)
        argsort_x1 = np.argsort(x1)
#         print(argsort_x1)
        argsort_x2 = np.argsort(x2)
#         print("arg", argsort_x1, argsort_x2)
        full_argsort_x2 = argsort_x2[perm_global]
#         print("what we want", full_argsort_x2)
#         arg_argsort_x1 = np.argsort(argsort_x1)
#         perm[0, k] = full_argsort_x2[arg_argsort_x1]

        full_argsort_x2_ = argsort_x2[perm_global_2]
#         print("what we want", full_argsort_x2)
#         arg_argsort_x1 = np.argsort(argsort_x1)
#         perm[1, k] = full_argsort_x2[argsort_x1]
        if X1_return.size:
            X1_return = np.concatenate((X1_return, np.tile(argsort_x1, 2)))
            X2_return = np.concatenate((X2_return, full_argsort_x2, full_argsort_x2_))
        else:
            X1_return = np.tile(argsort_x1, 2)
            X2_return = np.concatenate((full_argsort_x2, full_argsort_x2_))

    return X1_return, X2_return, weights, swaped


def slices_the_slides(X1, X2,
                      K=10,
                      solver="loop",
                      proj="normal"):
    np.random.seed(123456789)
    """
    X1 : source
    X2 : target
    K : number of slices
    proj : type of projection of the wasserstein slice
    """
    X1_return, X2_return, weights, swaped = sliced_wasserstein_barycenter(X1, X2, K=K, proj=proj, color_reg=None)

    weights = (np.tile(weights.reshape(-1), K) / K) * len(X2)
#     print("weights", weights)
#     print("X1_return", X1_return)
#     print("X2_return", X2_return)
    if not swaped:
        T = [X1_return, X2_return, weights]
    else:
        T = [X2_return, X1_return, weights]
#     print(weights.sum())
#     print("sum", [np.sum(T[2][T[0]==i]) for i in range(len(X1))])
#     print("sum", [np.sum(T[2][T[1]==i]) for i in range(len(X2))])
    return T
