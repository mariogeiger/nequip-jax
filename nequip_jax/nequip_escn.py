from typing import Callable, Optional, Union

import e3nn_jax as e3nn
import flax
import haiku as hk
import jax
import jax.numpy as jnp
from e3nn_jax.experimental.linear_shtp import LinearSHTP


class NEQUIPESCNLayerFlax(flax.linen.Module):
    avg_num_neighbors: float
    num_species: int = 1
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


class NEQUIPESCNLayerHaiku(hk.Module):
    def __init__(
        self,
        avg_num_neighbors: float,
        num_species: int = 1,
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
    self: Union[NEQUIPESCNLayerFlax, NEQUIPESCNLayerHaiku],
    vectors: e3nn.IrrepsArray,  # [n_edges, 3]
    node_feats: e3nn.IrrepsArray,  # [n_nodes, irreps]
    node_specie: jnp.ndarray,  # [n_nodes] int between 0 and num_species-1
    senders: jnp.ndarray,  # [n_edges]
    receivers: jnp.ndarray,  # [n_edges]
):
    num_nodes = node_feats.shape[0]
    num_edges = vectors.shape[0]
    assert vectors.shape == (num_edges, 3)
    assert node_feats.shape == (num_nodes, node_feats.irreps.dim)
    assert node_specie.shape == (num_nodes,)
    assert senders.shape == (num_edges,)
    assert receivers.shape == (num_edges,)

    # we regroup the target irreps to make sure that gate activation
    # has the same irreps as the target
    output_irreps = e3nn.Irreps(self.output_irreps).regroup()

    # target irreps plus extra scalars for the gate activation
    irreps = output_irreps + output_irreps.filter(
        drop="0e + 0o"
    ).num_irreps * e3nn.Irreps("0e")

    self_connection = Linear(
        irreps, num_indexed_weights=self.num_species, name="skip_tp"
    )(
        node_specie, node_feats
    )  # [n_nodes, feature * output_irreps]

    node_feats = Linear(node_feats.irreps, name="linear_up")(node_feats)

    messages = node_feats[senders]

    conv = LinearSHTP(irreps, mix=False)
    w_unused = conv.init(jax.random.PRNGKey(0), messages[0], vectors[0])
    w_unused_flat = flatten(w_unused)

    # Radial part
    lengths = e3nn.norm(vectors).array  # [n_edges, 1]
    mix = MultiLayerPerceptron(
        self.mlp_n_layers * (self.mlp_n_hidden,) + (w_unused_flat.size,),
        self.mlp_activation,
        output_activation=False,
    )(
        e3nn.bessel(lengths[:, 0], self.n_radial_basis)
        * e3nn.poly_envelope(5, 2)(lengths),
    )  # [n_edges, mul * w_unused_flat.size]

    # Discard 0 length edges that come from graph padding
    mix = jnp.where(lengths == 0.0, 0.0, mix)

    w = jax.vmap(unflatten, (0, None))(mix, w_unused)
    messages = jax.vmap(conv.apply)(w, messages, vectors)

    # Message passing
    node_feats = e3nn.scatter_sum(messages, dst=receivers, output_size=num_nodes)
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
    assert node_feats.shape == (num_nodes, output_irreps.dim)
    return node_feats


def flatten(w):
    return jnp.concatenate([x.ravel() for x in jax.tree_util.tree_leaves(w)])


def unflatten(array, template):
    lst = []
    start = 0
    for x in jax.tree_util.tree_leaves(template):
        lst.append(array[start : start + x.size].reshape(x.shape))
        start += x.size
    return jax.tree_util.tree_unflatten(jax.tree_util.tree_structure(template), lst)