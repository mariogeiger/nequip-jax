from typing import List, Optional

import e3nn_jax as e3nn


def filter_layers(
    layer_irreps: List[e3nn.Irreps], max_ell: Optional[int]
) -> List[e3nn.Irreps]:
    filtered = [e3nn.Irreps(layer_irreps[-1])]
    for irreps in reversed(layer_irreps[:-1]):
        irreps = e3nn.Irreps(irreps)
        if max_ell is not None:
            lmax = max_ell
        else:
            lmax = max(irreps.lmax, filtered[0].lmax)
        irreps = irreps.filter(
            keep=e3nn.tensor_product(
                filtered[0],
                e3nn.Irreps.spherical_harmonics(lmax=lmax),
            ).regroup()
        )
        filtered.insert(0, irreps)
    return filtered
