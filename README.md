[![Python package](https://github.com/tnbar/tednet/actions/workflows/python-package.yml/badge.svg)](https://github.com/tnbar/tednet/actions/workflows/python-package.yml)
[![Documentation Status](https://readthedocs.org/projects/tednet/badge/?version=latest)](https://tednet.readthedocs.io/en/latest/?badge=latest)
![PyPI - License](https://img.shields.io/pypi/l/tednet)
[![PyPI](https://img.shields.io/pypi/v/tednet)](https://pypi.org/project/tednet/)

# TedNet: A Pytorch Toolkit for Tensor Decomposition Networks
`tednet` is a toolkit for tensor decomposition networks. Tensor decomposition networks are neural networks whose layers are decomposed by tensor decomposition, including CANDECOMP/PARAFAC, Tucker2, Tensor Train, Tensor Ring and so on. For a convenience to do research on it, ``tednet`` provides excellent tools to deal with tensorial networks.


Now, **tednet** is easy to be installed by `pip`:

```shell script
pip install tednet
```

More information could be found in [Document](https://tednet.readthedocs.io/en/latest/index.html).


---

### Quick Start

##### Operation
There are some operations supported in `tednet`, and it is convinient to use them. First, import it:

```python
import tednet as tdt
```

Create matrix whose diagonal elements are ones:
```python
diag_matrix = tdt.eye(5, 5)
```

A way to transfer the Pytorch tensor into numpy array:

```python
diag_matrix = tdt.to_numpy(diag_matrix)
```

Similarly, the numpy array can be taken into Pytorch tensor by:

```python
diag_matrix = tdt.to_tensor(diag_matrix)
```

##### Tensor Decomposition Networks (Tensor Ring for Sample)
To use tensor ring decomposition models, simply calling the tensor ring module is enough.

```python
import tednet.tnn.tensor_ring as tr

# Define a TR-LeNet5
model = tr.TRLeNet5(10, [6, 6, 6, 6])
```
