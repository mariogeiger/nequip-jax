import e3nn_jax as e3nn
import jax
import jax.numpy as jnp

from nequip import NEQUIPLayer


def main():
    layer = NEQUIPLayer(
        avg_num_neighbors=10.0,
    )

    vectors = e3nn.IrrepsArray("1o", jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))
    node_feats = e3nn.IrrepsArray(
        "0e + 1o", jnp.array([[1.0, 0.0, 0.0, 0.0], [-1.0, 1.0, 0.0, 0.0]])
    )
    node_specie = jnp.array([0, 0])

    senders = jnp.array([0, 1])
    receivers = jnp.array([1, 0])

    w = layer.init(
        jax.random.PRNGKey(0),
        vectors,
        node_feats,
        node_specie,
        senders,
        receivers,
    )

    print(jax.tree_util.tree_map(lambda x: x.shape, w))

    apply = jax.jit(layer.apply)

    print("compiling")
    apply(w, vectors, node_feats, node_specie, senders, receivers)

    print("running")
    apply(w, vectors, node_feats, node_specie, senders, receivers)

    print("done")


if __name__ == "__main__":
    main()
