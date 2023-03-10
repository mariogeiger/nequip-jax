from typing import Callable

import e3nn_jax as e3nn
import flax
import jax
import jax.numpy as jnp


class NEQUIPLayer(flax.linen.Module):
    avg_num_neighbors: float
    num_species: int = 1
    sh_lmax: int = 3
    target_irreps: e3nn.Irreps = 64 * e3nn.Irreps("0e + 1o + 2e")
    even_activation: Callable[[jnp.ndarray], jnp.ndarray] = jax.nn.swish
    odd_activation: Callable[[jnp.ndarray], jnp.ndarray] = jax.nn.tanh
    mlp_activation: Callable[[jnp.ndarray], jnp.ndarray] = jax.nn.swish
    mlp_n_hidden: int = 64
    mlp_n_layers: int = 2
    n_radial_basis: int = 8

    @flax.linen.compact
    def __call__(
        self,
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
        target_irreps = e3nn.Irreps(self.target_irreps).regroup()

        # target irreps plus extra scalars for the gate activation
        irreps = target_irreps + target_irreps.filter(
            drop="0e + 0o"
        ).num_irreps * e3nn.Irreps("0e")

        self_connection = e3nn.flax.Linear(
            irreps, num_indexed_weights=self.num_species, name="skip_tp"
        )(
            node_specie, node_feats
        )  # [n_nodes, feature * target_irreps]

        node_feats = e3nn.flax.Linear(node_feats.irreps, name="linear_up")(node_feats)

        node_feats = MessagePassingConvolution(
            self.avg_num_neighbors,
            irreps,
            self.mlp_activation,
            self.mlp_n_hidden,
            self.mlp_n_layers,
            self.n_radial_basis,
            self.sh_lmax,
        )(vectors, node_feats, senders, receivers)

        node_feats = e3nn.flax.Linear(irreps, name="linear_down")(node_feats)

        node_feats = node_feats + self_connection  # [n_nodes, irreps]

        node_feats = e3nn.gate(
            node_feats,
            even_act=self.even_activation,
            even_gate_act=self.even_activation,
            odd_act=self.odd_activation,
            odd_gate_act=self.odd_activation,
        )

        assert node_feats.irreps == target_irreps
        assert node_feats.shape == (n_node, target_irreps.dim)
        return node_feats


class MessagePassingConvolution(flax.linen.Module):
    avg_num_neighbors: float
    target_irreps: e3nn.Irreps
    activation: Callable[[jnp.ndarray], jnp.ndarray] = jax.nn.swish
    mlp_n_hidden: int = 64
    mlp_n_layers: int = 2
    n_radial_basis: int = 8
    sh_lmax: int = 3

    @flax.linen.compact
    def __call__(
        self,
        vectors: e3nn.IrrepsArray,  # [n_edges, 3]
        node_feats: e3nn.IrrepsArray,  # [n_nodes, irreps]
        senders: jnp.ndarray,  # [n_edges, ]
        receivers: jnp.ndarray,  # [n_edges, ]
    ) -> e3nn.IrrepsArray:
        messages = node_feats[senders]

        # Angular part
        messages = e3nn.concatenate(
            [
                messages.filter(self.target_irreps),
                e3nn.tensor_product(
                    messages,
                    e3nn.spherical_harmonics(
                        [l for l in range(1, self.sh_lmax + 1)],
                        vectors,
                        normalize=True,
                        normalization="component",
                    ),
                    filter_ir_out=self.target_irreps,
                ),
            ]
        ).regroup()  # [n_edges, irreps]

        # Radial part
        lengths = e3nn.norm(vectors).array  # [n_edges, 1]
        mix = e3nn.flax.MultiLayerPerceptron(
            self.mlp_n_layers * (self.mlp_n_hidden,) + (messages.irreps.num_irreps,),
            self.activation,
            output_activation=False,
        )(
            jnp.where(
                lengths == 0.0,  # discard 0 length edges that come from graph padding
                0.0,
                e3nn.bessel(lengths[:, 0], self.n_radial_basis)
                * e3nn.poly_envelope(5, 2)(lengths),
            )
        )  # [n_edges, num_irreps]

        # Product of radial and angular part
        messages = messages * mix  # [n_edges, irreps]

        # Message passing
        zeros = e3nn.IrrepsArray.zeros(
            messages.irreps, node_feats.shape[:1], messages.dtype
        )
        node_feats = zeros.at[receivers].add(messages)  # [n_nodes, irreps]

        return node_feats / jnp.sqrt(self.avg_num_neighbors)
