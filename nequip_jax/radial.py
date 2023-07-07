import e3nn_jax as e3nn


def default_radial_basis(r, n: int):
    """Default radial basis function."""
    return e3nn.bessel(r, n) * e3nn.poly_envelope(5, 2)(r)[:, None]


def simple_smooth_radial_basis(r, n: int):
    return e3nn.soft_one_hot_linspace(
        r,
        start=0.0,
        end=1.0,
        number=n,
        basis="smooth_finite",
        start_zero=False,
        end_zero=True,
    )
