import pystable


# TODO: test
def gaussian():
    return pystable.create(alpha=2.0, beta=0.0, mu=0.0,
                           sigma=1.0, parameterization=1)


# TODO: test
def rescale(dist: pystable.STABLE_DIST, t: float) -> pystable.STABLE_DIST:
    """
    Rescales stable distribution using scaling property.
    For iid X_i ~ S(a, b, mu_i, sigma_i)

    sum_{i=1}^{t} X_i ~ S(a, b, mu, sigma)

    where
      mu = sum_{i=1}^{t} mu_i
      |sigma| = ( sum_{i=1}^{t} |sigma|**(a) ) ** (1/a)

    and `t` input for function is number of iids.
    """
    mu = dist.contents.mu_1 * t
    sigma = dist.contents.sigma * t**(1/dist.contents.alpha)

    return pystable.create(
        alpha=dist.contents.alpha,
        beta=dist.contents.beta,
        mu=mu,
        sigma=sigma,
        parameterization=1
    )
