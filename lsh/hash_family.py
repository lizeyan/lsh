import numpy as np


class E2Family:
    """
    Hash family for euclidean distance
    """

    def __init__(self, n_dims: int, w: float = 1., k: int = 10):
        """
        return floor( (q dot a) / w)
        :param n_dims: n_dims of the input vectors
        :param w:
        """
        assert w > 0, f"w should be positive, but {w} received"
        assert n_dims > 0, f"n_dims should be positive, but {n_dims} received"
        assert k > 0, f"k should be positive, but {k} received"
        self.w = w
        self.n_dims = n_dims
        self.k = k

    def _hash_generator(self, a, b):
        def _hash(q):
            # assert np.shape(q) == np.shape(a)[-1:], f'query should be in shape {np.shape(a)[-1:]}'
            _ret = np.floor_divide(np.tensordot(q, a, axes=[-1, -1]) + b, self.w).astype(np.int32)
            if np.ndim(q) == 1:
                return tuple(_ret)
            elif np.ndim(q) == 2:
                return list(map(tuple, _ret))
            else:
                raise RuntimeError(f"Unsupport query in shape {np.shape(q)}")

        return _hash

    def sample(self):
        """
        return a random sample from this hash family
        :return:
        """
        a = np.random.normal(loc=0, scale=1, size=(self.k, self.n_dims))
        b = np.random.uniform(low=0, high=self.w, size=(self.k,))
        return self._hash_generator(a, b)
