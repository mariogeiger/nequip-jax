import e3nn_jax as e3nn
import jax
import jax.numpy as jnp

from nequip import NEQUIPLayer


def main():
    layer = NEQUIPLayer(
        sh_lmax=3,
        num_features=64,
        hidden_irreps=e3nn.Irreps("0e + 1o + 2e"),
        num_species=1,
        avg_num_neighbors=10.0,
        activation=jax.nn.gelu,
    )

    vectors = e3nn.IrrepsArray("1o", jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))
    node_feats = e3nn.IrrepsArray("0e", jnp.array([[1.0], [-1.0]]))
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


if __name__ == "__main__":
    main()
