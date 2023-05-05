import e3nn_jax as e3nn
import flax
import haiku as hk
import jax
import jax.numpy as jnp
import jraph

from nequip_jax import (
    NEQUIPLayerFlax,
    NEQUIPLayerHaiku,
    NEQUIPESCNLayerFlax,
    NEQUIPESCNLayerHaiku,
)


def dummy_graph():
    return jraph.GraphsTuple(
        nodes={
            "positions": jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
            "species": jnp.array([0, 1]),
        },
        edges=None,
        globals=None,
        senders=jnp.array([0, 1]),
        receivers=jnp.array([1, 0]),
        n_node=jnp.array([2]),
        n_edge=jnp.array([2]),
    )


def test_nequip_flax():
    class NEQUIP(flax.linen.Module):
        @flax.linen.compact
        def __call__(self, graph: jraph.GraphsTuple) -> jraph.GraphsTuple:
            positions = graph.nodes["positions"]
            species = graph.nodes["species"]

            vectors = e3nn.IrrepsArray(
                "1o", positions[graph.receivers] - positions[graph.senders]
            )
            node_feats = flax.linen.Embed(num_embeddings=5, features=32)(species)
            node_feats = e3nn.IrrepsArray(f"{node_feats.shape[1]}x0e", node_feats)

            for _ in range(2):
                layer = NEQUIPLayerFlax(
                    avg_num_neighbors=1.0, output_irreps="16x0e + 16x1o"
                )
                node_feats = layer(
                    vectors,
                    node_feats,
                    species,
                    graph.senders,
                    graph.receivers,
                )

            return node_feats

    graph = dummy_graph()

    model = NEQUIP()
    w = model.init(jax.random.PRNGKey(0), graph)

    apply = jax.jit(model.apply)
    apply(w, graph)
    apply(w, graph)


def test_nequip_haiku():
    @hk.without_apply_rng
    @hk.transform
    def model(graph: jraph.GraphsTuple) -> jraph.GraphsTuple:
        positions = graph.nodes["positions"]
        species = graph.nodes["species"]

        vectors = e3nn.IrrepsArray(
            "1o", positions[graph.receivers] - positions[graph.senders]
        )
        node_feats = hk.Embed(vocab_size=5, embed_dim=32)(species)
        node_feats = e3nn.IrrepsArray(f"{node_feats.shape[1]}x0e", node_feats)

        for _ in range(3):
            layer = NEQUIPLayerHaiku(
                avg_num_neighbors=1.0,
                output_irreps="16x0e + 16x1o",
            )
            node_feats = layer(
                vectors,
                node_feats,
                species,
                graph.senders,
                graph.receivers,
            )

        return node_feats

    graph = dummy_graph()

    w = model.init(jax.random.PRNGKey(0), graph)

    apply = jax.jit(model.apply)
    apply(w, graph)
    apply(w, graph)


def test_nequip_escn_flax():
    class NEQUIP(flax.linen.Module):
        @flax.linen.compact
        def __call__(self, graph: jraph.GraphsTuple) -> jraph.GraphsTuple:
            positions = graph.nodes["positions"]
            species = graph.nodes["species"]

            vectors = e3nn.IrrepsArray(
                "1o", positions[graph.receivers] - positions[graph.senders]
            )
            node_feats = flax.linen.Embed(num_embeddings=5, features=32)(species)
            node_feats = e3nn.IrrepsArray(f"{node_feats.shape[1]}x0e", node_feats)

            for _ in range(2):
                layer = NEQUIPESCNLayerFlax(
                    avg_num_neighbors=1.0,
                    output_irreps="16x0e + 16x1o",
                )
                node_feats = layer(
                    vectors,
                    node_feats,
                    species,
                    graph.senders,
                    graph.receivers,
                )

            return node_feats

    graph = dummy_graph()

    model = NEQUIP()
    w = model.init(jax.random.PRNGKey(0), graph)

    apply = jax.jit(model.apply)
    apply(w, graph)
    apply(w, graph)


def test_nequip_escn_haiku():
    @hk.without_apply_rng
    @hk.transform
    def model(graph: jraph.GraphsTuple) -> jraph.GraphsTuple:
        positions = graph.nodes["positions"]
        species = graph.nodes["species"]

        vectors = e3nn.IrrepsArray(
            "1o", positions[graph.receivers] - positions[graph.senders]
        )
        node_feats = hk.Embed(vocab_size=5, embed_dim=32)(species)
        node_feats = e3nn.IrrepsArray(f"{node_feats.shape[1]}x0e", node_feats)

        for _ in range(3):
            layer = NEQUIPESCNLayerHaiku(
                avg_num_neighbors=1.0,
                output_irreps="16x0e + 16x1o",
            )
            node_feats = layer(
                vectors,
                node_feats,
                species,
                graph.senders,
                graph.receivers,
            )

        return node_feats

    graph = dummy_graph()

    w = model.init(jax.random.PRNGKey(0), graph)

    apply = jax.jit(model.apply)
    apply(w, graph)
    apply(w, graph)
