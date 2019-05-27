from lsh import BasicE2LSH
import numpy as np
from heapq import heappush, heappop
from loguru import logger


class MultiProbeE2LSH(BasicE2LSH):
    def __init__(self, n_dims: int, n_hash_table: int = 1, n_compounds: int = 1, w: float = 1., max_t: int = 4096):
        super().__init__(n_dims=n_dims, n_hash_table=n_hash_table, n_compounds=n_compounds, w=w)
        self.max_t = max_t

        self.perturbation_sequence = self.generate_perturbation_sequence()
        logger.debug(f"finish generating perturbation sequence for {self}")

    def query(self, q, t: int = 1):
        # TODO
        pass

    def generate_perturbation_sequence(self):
        def _is_valid(_a):
            _ = np.zeros(shape=(2 * self.n_compounds + 1, ))
            _[list(_a)] = 1
            return np.max(_[1:self.n_compounds + 1] + _[self.n_compounds + 1:]) <= 1

        indices = np.arange(2 * self.n_compounds + 1)
        z = indices * (indices + 1) * (self.w ** 2) / (4 * (self.n_compounds + 1) * (self.n_compounds + 2))
        heap = [(z[1], (1,))]
        perturbation_sequence = []
        for i in range(self.max_t):
            while True:
                a = heappop(heap)[1]
                if a[-1] < 2 * self.n_compounds:
                    a_e = a + (a[-1]+1,)
                    a_s = a[:-1] + (a[-1]+1,)
                    heappush(heap, (np.sum(z[list(a_e)]), a_e))
                    heappush(heap, (np.sum(z[list(a_s)]), a_s))
                if _is_valid(a):
                    break
            perturbation_sequence.append(a)
        return perturbation_sequence

