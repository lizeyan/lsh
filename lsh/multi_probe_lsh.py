import copy

from lsh import BasicE2LSH, HashTable
import numpy as np
from heapq import heappush, heappop
from loguru import logger


class MultiProbeE2LSH(BasicE2LSH):
    def __init__(self, n_dims: int, n_hash_table: int = 1, n_compounds: int = 1, w: float = 1., max_t: int = 4096):
        super().__init__(n_dims=n_dims, n_hash_table=n_hash_table, n_compounds=n_compounds, w=w)
        self.max_t = min(max_t, 4 ** self.n_compounds - 1)

        self.perturbation_sequence = self.generate_perturbation_sequence()
        logger.debug(f"finish generating perturbation sequence for {self}")

    def query(self, q, t: int = 1):
        assert t >= 1, 'at least query one time'
        results = list(filter(
            lambda x: len(x) > 0,
            map(lambda hash_table: self.query_hash_table(q, hash_table, t), self.hash_tables))
        )
        if len(results) > 0:
            return np.unique(np.concatenate(results), axis=0)
        else:
            return np.asarray([])

    def generate_perturbation_sequence(self):
        indices = np.arange(2 * self.n_compounds + 1)
        z = indices * (indices + 1) * (self.w ** 2) / (4 * (self.n_compounds + 1) * (self.n_compounds + 2))
        heap = [(z[1], (1,))]
        perturbation_sequence = [np.asarray((), dtype=np.int32)]
        for i in range(self.max_t):
            a = heappop(heap)[1]
            if a[-1] < 2 * self.n_compounds:
                a_e = a + (a[-1]+1,)
                a_s = a[:-1] + (a[-1]+1,)
                heappush(heap, (np.sum(z[list(a_e)]), a_e))
                heappush(heap, (np.sum(z[list(a_s)]), a_s))
            perturbation_sequence.append(np.asarray(a, dtype=np.int32))
        return perturbation_sequence

    def query_hash_table(self, q, hash_table: HashTable, t: int = 1):
        score_ele_negative = np.asarray(hash_table.h.x_negative(q))
        score_ele_positive = self.w - score_ele_negative
        score_ele = np.concatenate([score_ele_negative, score_ele_positive])
        map_indices = np.argsort(score_ele)
        basic_query = hash_table.h(q)
        results = []
        for perturbation in self.perturbation_sequence[:t]:
            real_perturbation = map_indices[perturbation - 1]  # [0, self.n_compounds)
            query = list(copy.copy(basic_query))
            for ele_perturb in real_perturbation:
                sign = 2 * ele_perturb // self.n_compounds - 1
                loc = ele_perturb % self.n_compounds
                query[loc] += sign
            ret = hash_table.entries.get(tuple(query), [])
            if ret:
                results.append(ret)
        if results:
            return np.concatenate(results)
        else:
            return np.asarray([])



