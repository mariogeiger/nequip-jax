import time

import e3nn_jax as e3nn
import flax
import jax
import jax.numpy as jnp
import jraph

from nequip_jax import NEQUIPLayerFlax


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

        for _ in range(3):
            layer = NEQUIPLayerFlax(
                avg_num_neighbors=1.0,
                output_irreps=64 * e3nn.Irreps("0e + 1o + 2e + 3o"),
            )
            node_feats = layer(
                vectors,
                node_feats,
                species,
                graph.senders,
                graph.receivers,
            )

        return node_feats


def main():
    model = NEQUIP()

    n_nodes = 256
    n_edges = 4096

    graph = jraph.GraphsTuple(
        nodes={
            "positions": jax.random.normal(jax.random.PRNGKey(0), (n_nodes, 3)),
            "species": jax.random.randint(jax.random.PRNGKey(1), (n_nodes,), 0, 5),
        },
        edges=None,
        globals=None,
        senders=jax.random.randint(jax.random.PRNGKey(2), (n_edges,), 0, n_nodes),
        receivers=jax.random.randint(jax.random.PRNGKey(3), (n_edges,), 0, n_nodes),
        n_node=jnp.array([n_nodes]),
        n_edge=jnp.array([n_edges]),
    )

    w = jax.jit(model.init)(jax.random.PRNGKey(0), graph)
    print("number of parameters:", sum(x.size for x in jax.tree_util.tree_leaves(w)))

    apply = jax.jit(model.apply)

    print("compiling forward pass")
    apply(w, graph)

    print("running forward pass")
    t0 = time.time()
    apply(w, graph).array.block_until_ready()
    t1 = time.time()
    print(f"took {t1 - t0} seconds")

    bwr = jax.jit(jax.grad(lambda w, graph: apply(w, graph).array.sum()))
    print("compiling backward pass")
    bwr(w, graph)

    print("running backward pass")
    t0 = time.time()
    g = bwr(w, graph)
    jax.tree_util.tree_map(lambda x: x.block_until_ready(), g)
    t1 = time.time()
    print(f"took {t1 - t0} seconds")


if __name__ == "__main__":
    main()
