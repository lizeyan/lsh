import numpy as np


class E2Family:
    """
    Hash family for euclidean distance
    """
    def __init__(self, n_dims: int, w: float = 1.):
        """
        return floor( (q dot a) / w)
        :param n_dims: n_dims of the input vectors
        :param w:
        """
        assert w > 0, f"w should be positive, but {w} received"
        assert n_dims > 0, f"n_dims should be positive, but {n_dims} received"
        self.w = w
        self.n_dims = n_dims

    def _hash_generator(self, a):
        return lambda q: np.floor_divide(np.tensordot(q, a, axes=[-1, -1]), self.w)

    def sample(self):
        """
        return a random sample from this hash family
        :return:
        """
        a = np.random.normal(loc=0, scale=1, size=(self.n_dims,))
        return self._hash_generator(a)


class CompoundE2Family(E2Family):
    """
    Compound Hash family for euclidean distance
    g(q) = (h_1(q), h_2(q), ..., h_k(q))
    """
    def __init__(self, n_dims: int, w: float = 1., k: int = 20):
        """
        return floor( (q dot a) / w)
        :param n_dims: n_dims of the input vectors
        :param w:
        """
        super().__init__(n_dims=n_dims, w=w)
        assert k > 0, f"k should be positive, but {k} received"
        self.k = k

    def sample(self):
        """
        return a random sample from this hash family
        :return:
        """
        a = np.random.normal(loc=0, scale=1, size=(self.k, self.n_dims))
        return self._hash_generator(a)
