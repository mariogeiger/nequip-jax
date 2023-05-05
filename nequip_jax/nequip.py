from typing import Callable, Optional, Union

import e3nn_jax as e3nn
import flax
import haiku as hk
import jax
import jax.numpy as jnp


class NEQUIPLayerFlax(flax.linen.Module):
    avg_num_neighbors: float
    num_species: int = 1
    max_ell: int = 3
    output_irreps: e3nn.Irreps = 64 * e3nn.Irreps("0e + 1o + 2e")
    even_activation: Callable[[jnp.ndarray], jnp.ndarray] = jax.nn.swish
    odd_activation: Callable[[jnp.ndarray], jnp.ndarray] = jax.nn.tanh
    mlp_activation: Callable[[jnp.ndarray], jnp.ndarray] = jax.nn.swish
    mlp_n_hidden: int = 64
    mlp_n_layers: int = 2
    n_radial_basis: int = 8

    @flax.linen.compact
    def __call__(
        self,
        vectors: e3nn.IrrepsArray,
        node_feats: e3nn.IrrepsArray,
        node_specie: jnp.ndarray,
        senders: jnp.ndarray,
        receivers: jnp.ndarray,
    ):
        return _impl(
            e3nn.flax.Linear,
            e3nn.flax.MultiLayerPerceptron,
            self,
            vectors,
            node_feats,
            node_specie,
            senders,
            receivers,
        )


class NEQUIPLayerHaiku(hk.Module):
    def __init__(
        self,
        avg_num_neighbors: float,
        num_species: int = 1,
        max_ell: int = 3,
        output_irreps: e3nn.Irreps = 64 * e3nn.Irreps("0e + 1o + 2e"),
        even_activation: Callable[[jnp.ndarray], jnp.ndarray] = jax.nn.swish,
        odd_activation: Callable[[jnp.ndarray], jnp.ndarray] = jax.nn.tanh,
        mlp_activation: Callable[[jnp.ndarray], jnp.ndarray] = jax.nn.swish,
        mlp_n_hidden: int = 64,
        mlp_n_layers: int = 2,
        n_radial_basis: int = 8,
        name: Optional[str] = None,
    ):
        super().__init__(name)
        self.avg_num_neighbors = avg_num_neighbors
        self.num_species = num_species
        self.max_ell = max_ell
        self.output_irreps = output_irreps
        self.even_activation = even_activation
        self.odd_activation = odd_activation
        self.mlp_activation = mlp_activation
        self.mlp_n_hidden = mlp_n_hidden
        self.mlp_n_layers = mlp_n_layers
        self.n_radial_basis = n_radial_basis

    def __call__(
        self,
        vectors: e3nn.IrrepsArray,
        node_feats: e3nn.IrrepsArray,
        node_specie: jnp.ndarray,
        senders: jnp.ndarray,
        receivers: jnp.ndarray,
    ):
        return _impl(
            e3nn.haiku.Linear,
            e3nn.haiku.MultiLayerPerceptron,
            self,
            vectors,
            node_feats,
            node_specie,
            senders,
            receivers,
        )


def _impl(
    Linear: Callable,
    MultiLayerPerceptron: Callable,
    self: Union[NEQUIPLayerFlax, NEQUIPLayerHaiku],
    vectors: e3nn.IrrepsArray,  # [n_edges, 3]
    node_feats: e3nn.IrrepsArray,  # [n_nodes, irreps]
    node_specie: jnp.ndarray,  # [n_nodes] int between 0 and num_species-1
    senders: jnp.ndarray,  # [n_edges]
    receivers: jnp.ndarray,  # [n_edges]
):
    n_edge = vectors.shape[0]
    n_node = node_feats.shape[0]
    assert vectors.shape == (n_edge, 3)
    assert node_feats.shape == (n_node, node_feats.irreps.dim)
    assert node_specie.shape == (n_node,)
    assert senders.shape == (n_edge,)
    assert receivers.shape == (n_edge,)

    # we regroup the target irreps to make sure that gate activation
    # has the same irreps as the target
    output_irreps = e3nn.Irreps(self.output_irreps).regroup()

    # target irreps plus extra scalars for the gate activation
    num_nonscalar = output_irreps.filter(drop="0e + 0o").num_irreps
    irreps = output_irreps + e3nn.Irreps(f"{num_nonscalar}x0e").simplify()

    self_connection = Linear(
        irreps, num_indexed_weights=self.num_species, name="skip_tp"
    )(
        node_specie, node_feats
    )  # [n_nodes, feature * output_irreps]

    node_feats = Linear(node_feats.irreps, name="linear_up")(node_feats)

    messages = node_feats[senders]

    # Angular part
    messages = e3nn.concatenate(
        [
            messages.filter(irreps),
            e3nn.tensor_product(
                messages,
                e3nn.spherical_harmonics(
                    [l for l in range(1, self.max_ell + 1)],
                    vectors,
                    normalize=True,
                    normalization="component",
                ),
                filter_ir_out=irreps,
            ),
        ]
    ).regroup()  # [n_edges, irreps]

    # Radial part
    lengths = e3nn.norm(vectors).array  # [n_edges, 1]
    mix = MultiLayerPerceptron(
        self.mlp_n_layers * (self.mlp_n_hidden,) + (messages.irreps.num_irreps,),
        self.mlp_activation,
        output_activation=False,
    )(
        e3nn.bessel(lengths[:, 0], self.n_radial_basis)
        * e3nn.poly_envelope(5, 2)(lengths),
    )  # [n_edges, num_irreps]

    # Discard 0 length edges that come from graph padding
    mix = jnp.where(lengths == 0.0, 0.0, mix)

    # Product of radial and angular part
    messages = messages * mix  # [n_edges, irreps]

    # Message passing
    zeros = e3nn.IrrepsArray.zeros(
        messages.irreps, node_feats.shape[:1], messages.dtype
    )
    node_feats = zeros.at[receivers].add(messages)  # [n_nodes, irreps]
    node_feats = node_feats / jnp.sqrt(self.avg_num_neighbors)

    node_feats = Linear(irreps, name="linear_down")(node_feats)

    node_feats = node_feats + self_connection  # [n_nodes, irreps]

    node_feats = e3nn.gate(
        node_feats,
        even_act=self.even_activation,
        even_gate_act=self.even_activation,
        odd_act=self.odd_activation,
        odd_gate_act=self.odd_activation,
    )

    assert node_feats.irreps == output_irreps
    assert node_feats.shape == (n_node, output_irreps.dim)
    return node_feats
