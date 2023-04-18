## Installation

Install directly from GitHub with:

```
pip install git+https://github.com/mariogeiger/nequip-jax
```


## Usage

### Original Nequip

```python
from nequip_jax import NEQUIPLayerFlax  # Flax version
from nequip_jax import NEQUIPLayerHaiku  # Haiku version
```

Look at [test.py](test.py) for an example of how to stack the layers.

### Optimization using ESCN

Optimization for large `L` using [https://arxiv.org/pdf/2302.03655.pdf](paper).
With extra support of parity.

```python
from nequip_jax import NEQUIPESCNLayerFlax  # Flax version
from nequip_jax import NEQUIPESCNLayerHaiku  # Haiku version
```
