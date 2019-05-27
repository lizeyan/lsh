import json
import copy


class LSH:
    def __init__(self):
        pass

    def add(self, q, **kwargs):
        return self.add_batch([q])

    def query(self, q, **kwargs):
        raise NotImplementedError()

    def add_batch(self, q_list):
        raise NotImplementedError()

    def __str__(self):
        x = copy.copy(self.__dict__)
        x['name'] = self.__class__.__name__
        return "&".join(f"{key}={value}" for key, value in sorted(filter(
            lambda item: isinstance(item[1], (str, float, int)), x.items()
        ), key=lambda item: item[0]))

