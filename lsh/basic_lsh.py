from typing import Callable, List, Dict
import numpy as np

from lsh import LSH
from lsh.hash_family import CompoundE2Family


class HashTable:
    def __init__(self, h: Callable):
        """
        :param h: hash function of this hash table
        """
        self.h = h
        self.entries = {}  # type: Dict[int, List]

    def add_batch(self, q_list):
        """
        add a entry q
        """
        bid_list = self.h(q_list)
        for bid, q in zip(bid_list, q_list):
            if bid not in self.entries:
                self.entries[bid] = []
            self.entries[bid].append(q)

    def query(self, q):
        return self.entries.get(self.h(q), [])


class BasicE2LSH(LSH):
    def add_batch(self, q_list):
        list(map(lambda hash_table: hash_table.add_batch(q_list), self.hash_tables))

    def __init__(self, n_dims: int, n_hash_table: int = 20, n_compounds: int = 20, w: float = 1.):
        super().__init__()
        self.n_hash_table = n_hash_table
        self.n_compounds = n_compounds
        self.w = w
        self.hash_family = CompoundE2Family(n_dims=n_dims, k=n_compounds, w=w)
        self.hash_tables = [HashTable(self.hash_family.sample()) for _ in range(self.n_hash_table)]

    def query(self, q):
        """
        :return:
        """
        results = np.concatenate([hash_table.query(q) for hash_table in self.hash_tables])
        return np.unique(results, axis=0)
