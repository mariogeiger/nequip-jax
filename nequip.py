from typing import Callable

import e3nn_jax as e3nn
import flax
import jax.numpy as jnp


class NEQUIPLayer(flax.linen.Module):
    sh_lmax: int = 3
    num_features: int = 64
    hidden_irreps: e3nn.Irreps = e3nn.Irreps("0e + 1o + 2e")
    num_species: int = 1
    avg_num_neighbors: float
    activation: Callable[[jnp.ndarray], jnp.ndarray]

    @flax.linen.compact
    def __call__(
        self,
        vectors: e3nn.IrrepsArray,  # [n_edges, 3]
        node_feats: e3nn.IrrepsArray,  # [n_nodes, irreps]
        node_specie: jnp.ndarray,  # [n_nodes] int between 0 and num_species-1
        senders: jnp.ndarray,  # [n_edges]
        receivers: jnp.ndarray,  # [n_edges]
    ):
        lengths = safe_norm(vectors.array, axis=-1)

        basis = e3nn.bessel(lengths, 8)  # [n_edges, num_basis]
        cutoff = e3nn.poly_envelope(5, 2)(lengths)  # [n_edges]
        radial_embedding = basis * cutoff[:, None]  # [n_edges, num_basis]

        edge_attrs = e3nn.concatenate(
            [
                radial_embedding,
                e3nn.spherical_harmonics(
                    [l for l in range(1, self.sh_lmax + 1)],
                    vectors / lengths[..., None],
                    normalize=False,
                    normalization="component",
                ),
            ]
        )  # [n_edges, irreps]

        # TODO: check that it's equivariant to what is done in NEQUIP
        sc = e3nn.haiku.Linear(
            self.num_features * self.hidden_irreps,
            num_indexed_weights=self.num_species,
            name="skip_tp",
        )(
            node_specie, node_feats
        )  # [n_nodes, feature * hidden_irreps]

        node_feats = e3nn.flax.Linear(node_feats.irreps, name="linear_up")(node_feats)

        target_irreps: e3nn.Irreps = self.num_features * self.hidden_irreps
        target_irreps += target_irreps.filter(drop="0e").num_irreps * e3nn.Irreps("0e")

        node_feats = MessagePassingConvolution(
            self.avg_num_neighbors, target_irreps, self.activation
        )(node_feats, edge_attrs, senders, receivers)

        node_feats = e3nn.flax.Linear(target_irreps, name="linear_down")(node_feats)

        node_feats = e3nn.gate(
            node_feats, even_act=self.activation, even_gate_act=self.activation
        )

        node_feats = node_feats + sc  # [n_nodes, feature * hidden_irreps]

        return node_feats


class MessagePassingConvolution(flax.linen.Module):
    avg_num_neighbors: float
    target_irreps: e3nn.Irreps
    activation: Callable[[jnp.ndarray], jnp.ndarray]

    @flax.linen.compact
    def __call__(
        self,
        node_feats: e3nn.IrrepsArray,  # [n_nodes, irreps]
        edge_attrs: e3nn.IrrepsArray,  # [n_edges, irreps]
        senders: jnp.ndarray,  # [n_edges, ]
        receivers: jnp.ndarray,  # [n_edges, ]
    ) -> e3nn.IrrepsArray:
        assert node_feats.ndim == 2
        assert edge_attrs.ndim == 2

        messages = node_feats[senders]

        messages = e3nn.concatenate(
            [
                messages.filter(self.target_irreps),
                e3nn.tensor_product(
                    messages,
                    edge_attrs.filter(drop="0e"),
                    filter_ir_out=self.target_irreps,
                ),
            ]
        ).regroup()  # [n_edges, irreps]

        mix = e3nn.haiku.MultiLayerPerceptron(
            3 * [64] + [messages.irreps.num_irreps],
            self.activation,
            output_activation=False,
        )(
            edge_attrs.filter(keep="0e")
        )  # [n_edges, num_irreps]

        messages = messages * mix  # [n_edges, irreps]

        zeros = e3nn.IrrepsArray.zeros(
            messages.irreps, node_feats.shape[:1], messages.dtype
        )
        node_feats = zeros.at[receivers].add(messages)  # [n_nodes, irreps]

        return node_feats / jnp.sqrt(self.avg_num_neighbors)


def safe_norm(x: jnp.ndarray, axis: int = None, keepdims=False) -> jnp.ndarray:
    """nan-safe norm."""
    x2 = jnp.sum(x**2, axis=axis, keepdims=keepdims)
    return jnp.where(x2 == 0, 1, x2) ** 0.5
